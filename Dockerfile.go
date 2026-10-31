FROM golang:1.25-alpine AS build
WORKDIR /app
COPY go/go.mod go/go.sum ./
RUN go mod download
COPY go/ ./
RUN CGO_ENABLED=0 go build -o /gateway .
FROM alpine:3.21
COPY --from=build /gateway /gateway
COPY web /web
EXPOSE 8080
CMD ["/gateway"]
