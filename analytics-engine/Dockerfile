FROM golang:1.23-alpine

RUN apk add --no-cache git curl

WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN go build -o /bin/analytics-engine ./...

CMD ["/bin/analytics-engine"]