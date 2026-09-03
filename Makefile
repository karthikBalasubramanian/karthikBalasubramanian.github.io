# ==============================================================================
# Makefile - Local Development & Docker Environment (Mimicking GitHub Actions)
# ==============================================================================

.PHONY: help build serve up down logs clean restart

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker container for Jekyll site + all Vite microsites"
	@echo "  make serve    - Start local server at http://localhost:4000"
	@echo "  make down     - Stop local Docker container"
	@echo "  make logs     - View live container logs"
	@echo "  make restart  - Rebuild & restart container"
	@echo "  make clean    - Remove build artifacts & containers"

build:
	@echo "Building local Docker container..."
	docker compose build

serve: up

up:
	@echo "Starting local website container..."
	docker compose up -d
	@echo "================================================================="
	@echo "🚀 Local site running at: http://localhost:4000"
	@echo "================================================================="

down:
	@echo "Stopping local container..."
	docker compose down

logs:
	docker compose logs -f

restart: down build up

clean:
	docker compose down -v --remove-orphans
	rm -rf _site
	@echo "Cleaned build artifacts."
