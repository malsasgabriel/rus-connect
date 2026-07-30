package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/segmentio/kafka-go"
)

const (
	// writeWait is the maximum time allowed to write a message to a peer.
	writeWait = 10 * time.Second
	// pongWait is how long we wait for the next pong before dropping the peer.
	pongWait = 60 * time.Second
	// pingPeriod must be shorter than pongWait.
	pingPeriod = (pongWait * 9) / 10
	// maxMessageSize caps inbound frames; clients are not expected to send data.
	maxMessageSize = 4096

	kafkaMinBackoff = 100 * time.Millisecond
	kafkaMaxBackoff = 30 * time.Second
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		origin := r.Header.Get("Origin")
		if origin == "" {
			return true
		}
		allowed := os.Getenv("WS_ALLOWED_ORIGINS")
		if allowed == "" {
			allowed = "http://localhost:3000,http://127.0.0.1:3000"
		}
		for _, v := range strings.Split(allowed, ",") {
			if strings.TrimSpace(v) == origin {
				return true
			}
		}
		return false
	},
}

type Client struct {
	conn *websocket.Conn
	send chan []byte
}

type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.Mutex
}

func newHub() *Hub {
	return &Hub{
		broadcast:  make(chan []byte),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		clients:    make(map[*Client]bool),
	}
}

func (h *Hub) run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()
		case message := <-h.broadcast:
			// Slow clients are evicted from the map below, so this branch mutates
			// shared state and must hold the write lock. Using RLock here was a
			// data race that could panic with "concurrent map iteration and map
			// write".
			h.mu.Lock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					close(client.send)
					delete(h.clients, client)
				}
			}
			h.mu.Unlock()
		}
	}
}

func serveWs(hub *Hub, w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println(err)
		return
	}
	client := &Client{conn: conn, send: make(chan []byte, 256)}
	hub.register <- client

	go client.writePump()
	go client.readPump(hub)
}

func (c *Client) readPump(hub *Hub) {
	defer func() {
		hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(maxMessageSize)
	_ = c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		return c.conn.SetReadDeadline(time.Now().Add(pongWait))
	})

	for {
		if _, _, err := c.conn.ReadMessage(); err != nil {
			break
		}
	}
}

func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			_ = c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				_ = c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}
		case <-ticker.C:
			_ = c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func main() {
	log.Println("\U0001F680 Starting WebSocket Server...")

	hub := newHub()
	go hub.run()

	// Kafka Setup
	kafkaBrokers := []string{"kafka:9092"}
	if brokers := os.Getenv("KAFKA_BROKERS"); brokers != "" {
		kafkaBrokers = strings.Split(brokers, ",")
	}

	// Use a new reader with a unique group ID for the WebSocket server
	r := kafka.NewReader(kafka.ReaderConfig{
		Brokers: kafkaBrokers,
		Topic:   "direction_signals",
		GroupID: "websocket-server-group",
	})
	defer r.Close()

	// Consume Kafka messages in a background goroutine
	go func() {
		backoff := kafkaMinBackoff
		for {
			m, err := r.ReadMessage(context.Background())
			if err != nil {
				// Without a delay a permanent broker failure turns this loop into
				// a CPU-burning spin that floods the logs.
				log.Printf("Error reading kafka message: %v (retrying in %v)", err, backoff)
				time.Sleep(backoff)
				if backoff < kafkaMaxBackoff {
					backoff *= 2
					if backoff > kafkaMaxBackoff {
						backoff = kafkaMaxBackoff
					}
				}
				continue
			}
			backoff = kafkaMinBackoff
			// Broadcast the message value (JSON signal) to all connected clients
			hub.broadcast <- m.Value
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/ws", func(w http.ResponseWriter, req *http.Request) {
		serveWs(hub, w, req)
	})
	mux.HandleFunc("/health", healthzHandler)
	mux.HandleFunc("/healthz", healthzHandler)
	mux.HandleFunc("/readyz", readyzHandler(r))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8082" // Default to 8082 to avoid conflict with API Gateway (8080) and Analytics (8081)
	}

	// No WriteTimeout: it would kill long-lived websocket connections.
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	log.Printf("Listening on :%s", port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}

func healthzHandler(w http.ResponseWriter, _ *http.Request) {
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
	})
}

func readyzHandler(reader *kafka.Reader) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		ready := reader != nil
		if !ready {
			w.WriteHeader(http.StatusServiceUnavailable)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"status": "not_ready",
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "ready",
		})
	}
}
