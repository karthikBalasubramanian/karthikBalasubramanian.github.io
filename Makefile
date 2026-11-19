# ==============================================================================
# Makefile - Local Development & Docker Environment (Mimicking GitHub Actions)
# ==============================================================================

.PHONY: help build serve up down logs clean restart site check tokens

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker container for Jekyll site + all Vite microsites"
	@echo "  make serve    - Start local server at http://localhost:4000"
	@echo "  make down     - Stop local Docker container"
	@echo "  make logs     - View live container logs"
	@echo "  make restart  - Rebuild & restart container"
	@echo "  make clean    - Remove build artifacts & containers"
	@echo "  make site     - Build without Docker (needs Ruby + Node): Jekyll (local URL) + all microsites into _site"
	@echo "  make check    - Run html-proofer on _site (internal links)"
	@echo "  make tokens   - Regenerate/sync shared microsite design tokens + header/footer"

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

# ------------------------------------------------------------------------------
# Native (non-Docker) build, same steps as GitHub Actions
# ------------------------------------------------------------------------------
site:
	bundle install
	JEKYLL_ENV=production bundle exec jekyll build --config _config.yml,_config.local.yml
	scripts/build-microsites.sh _site

check:
	bundle exec htmlproofer _site --disable-external --no-enforce-https \
	  --ignore-urls "/^http:\/\/127.0.0.1/,/^http:\/\/0.0.0.0/,/^http:\/\/localhost/"

tokens:
	node shared/site-chrome/build.mjs
