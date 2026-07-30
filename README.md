# rus-connect (PredPump Radar)

Real-time crypto pump/direction radar: market data is pulled from Bybit, pushed
through Kafka, scored by an online-learning ML engine and streamed to a React
dashboard over websockets.

## Architecture

```
Bybit REST ──> data-fetcher ──> Kafka (ticker, orderbook, trades, candle_1m)
                    │                     │
                    v                     v
              ClickHouse            analytics-engine ──> Kafka (direction_signals, pump_signals)
                                                              │
                                          api-gateway <───────┘
                                               │  REST /api/v1/* + WS /ws
                                               v
                                    frontend (nginx + React)
```

| Service | Port | Purpose |
| --- | --- | --- |
| `frontend` | 3000 | React dashboard, nginx proxies `/api` and `/ws` |
| `api-gateway` | 8080 | REST API + websocket fan-out, proxies analytics-engine |
| `analytics-engine` | 8081 | Feature engineering, online learning, signals |
| `websocket-server` | 8082 | Standalone Kafka -> websocket bridge |
| `data-fetcher` | 8083 | Bybit polling, Kafka producing, ClickHouse writes |
| `clickhouse` | 8123 / 9000 | Candle storage |
| `kafka` | 9092 (29092 from host) | Message bus |
| `redis` | internal | Caching |

## Quick start

```bash
cp env.example .env      # optional: override CLICKHOUSE_PASSWORD, symbols, ...
docker compose up -d --build
```

Then open http://localhost:3000.

Startup order is enforced with healthchecks: zookeeper -> kafka -> `kafka-init`
(creates all topics) -> clickhouse -> services -> frontend. The first boot takes
~1-2 minutes, mostly Kafka.

Useful checks:

```bash
docker compose ps
curl -s localhost:8080/healthz
curl -s localhost:8080/readyz     # 503 until analytics-engine answers
curl -s localhost:8081/healthz
docker compose logs -f analytics-engine
```

## Local development

```bash
# frontend (proxies /api and /ws to localhost:8080 via Vite)
cd frontend && npm ci && npm run dev

# any Go service
cd api-gateway && go build ./... && go test -race ./...
```

Each Go service is its own module (`api-gateway`, `analytics-engine`,
`data-fetcher`, `websocket-server`), so run Go commands from inside the service
directory. CI (`.github/workflows/ci.yml`) runs `go vet`, `go build` and
`go test -race` for all four.

## Environment variables

| Variable | Default | Used by |
| --- | --- | --- |
| `KAFKA_BROKERS` | `kafka:9092` | all Go services |
| `CH_DSN` | `clickhouse://app:app_password@clickhouse:9000/default` | data-fetcher, analytics-engine |
| `CLICKHOUSE_USER` / `CLICKHOUSE_PASSWORD` / `CLICKHOUSE_DB` | `app` / `app_password` / `default` | clickhouse |
| `ANALYTICS_ENGINE_URL` | `http://analytics-engine:8081` | api-gateway |
| `WS_ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | api-gateway, websocket-server |
| `API_GATEWAY_PORT` / `ANALYTICS_ENGINE_PORT` / `DATA_FETCHER_PORT` / `PORT` | `8080` / `8081` / `8083` / `8082` | respective service |
| `CONFIDENCE_THRESHOLD` | `0.10` | analytics-engine |
| `ML_MIN_CANDLES_BOOTSTRAP` / `ML_MIN_CANDLES` | `20` / `60` | analytics-engine |
| `GIN_MODE` | `release` | api-gateway |
| `VITE_WS_URL` | same origin `/ws` | frontend (build time) |

The frontend is built as a static bundle, so `VITE_*` variables only take effect
at build time. In Docker it talks to the gateway through nginx on the same
origin, which is why no API URL needs to be configured.

## Websockets

`GET /ws` on the api-gateway. Message envelope:

```json
{ "type": "market_update" | "initial_data" | "pump_signal_update" | "direction_signal", "data": {} }
```

The browser client reconnects automatically with exponential backoff. Origins
must be listed in `WS_ALLOWED_ORIGINS`, otherwise the handshake is rejected.
