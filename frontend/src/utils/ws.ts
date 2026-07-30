export type ReconnectingSocketHandlers = {
  onOpen?: (socket: WebSocket) => void;
  onMessage?: (event: MessageEvent) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  minDelayMs?: number;
  maxDelayMs?: number;
};

/**
 * Same-origin websocket URL.
 *
 * The previous logic hardcoded `ws://<host>:8080/ws` for localhost:3000, which
 * bypassed the dev proxy, broke behind nginx and could never work over https.
 */
export function resolveWsUrl(path = "/ws"): string {
  const configured = (import.meta.env.VITE_WS_URL as string | undefined)?.trim();
  if (configured) {
    return configured.replace(/\/$/, "");
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

/**
 * Opens a websocket that reconnects with exponential backoff + jitter.
 * Returns a disposer that stops reconnecting and closes the socket.
 */
export function createReconnectingSocket(
  url: string,
  handlers: ReconnectingSocketHandlers = {}
): () => void {
  const minDelay = handlers.minDelayMs ?? 1000;
  const maxDelay = handlers.maxDelayMs ?? 30000;

  let socket: WebSocket | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let delay = minDelay;
  let disposed = false;

  const connect = () => {
    if (disposed) return;

    socket = new WebSocket(url);

    socket.onopen = () => {
      delay = minDelay;
      if (socket) handlers.onOpen?.(socket);
    };

    socket.onmessage = (event) => handlers.onMessage?.(event);

    socket.onerror = (event) => handlers.onError?.(event);

    socket.onclose = (event) => {
      handlers.onClose?.(event);
      if (disposed) return;
      const jitter = Math.random() * 250;
      timer = setTimeout(connect, delay + jitter);
      delay = Math.min(delay * 2, maxDelay);
    };
  };

  connect();

  return () => {
    disposed = true;
    if (timer) clearTimeout(timer);
    if (socket) {
      socket.onclose = null;
      socket.close();
    }
  };
}
