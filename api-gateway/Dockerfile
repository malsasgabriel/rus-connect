FROM golang:1.23-alpine

RUN apk add --no-cache git

WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN go build -o /bin/api-gateway ./...

CMD ["/bin/api-gateway"]