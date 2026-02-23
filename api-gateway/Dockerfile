# Stage 1: Builder
FROM golang:1.24-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /build

# Copy pkg/validator first to maintain relative path structure for replace directive
COPY pkg/validator ./pkg/validator

# Copy go.mod and go.sum from api-gateway
COPY api-gateway/go.mod api-gateway/go.sum ./api-gateway/

# Change to api-gateway directory and download dependencies
WORKDIR /build/api-gateway
RUN go mod download && go mod verify

# Copy the rest of api-gateway source code
COPY api-gateway/ ./

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -a -installsuffix cgo \
    -o api-gateway .

# Stage 2: Runtime
FROM alpine:3.19

RUN apk --no-cache add ca-certificates tzdata curl && \
    addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser

WORKDIR /app

COPY --from=builder /build/api-gateway .

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/infrastructure/metrics || exit 1

CMD ["./api-gateway"]