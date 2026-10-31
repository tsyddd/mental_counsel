#!/usr/bin/env bash
set -euo pipefail

cat > docker-compose.override.yml <<'YAML'
services:
  gateway:
    volumes:
      - /home/dashan/interview-platform-build/gateway:/gateway:ro
      - ./web:/web:ro
YAML

docker compose up -d --no-build --force-recreate gateway
docker compose ps gateway
