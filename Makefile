.PHONY: help build up down restart logs ps clean test model

help:
	@echo "AI Telemetry Anomaly Detection - Docker Compose Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build       - Build all Docker images"
	@echo "  up          - Start all services"
	@echo "  down        - Stop all services"
	@echo "  restart     - Restart all services"
	@echo "  logs        - View logs (all services)"
	@echo "  ps          - List running services"
	@echo "  clean       - Remove all containers and volumes"
	@echo "  test        - Run system health checks"
	@echo "  model       - Generate dummy model file"
	@echo "  scale-ingest- Scale ingestion service (replicas=3)"
	@echo ""

build:
	@echo "Building all services..."
	docker-compose build

up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 5
	@make ps

down:
	@echo "Stopping all services..."
	docker-compose down

restart:
	@echo "Restarting all services..."
	docker-compose restart

logs:
	docker-compose logs -f

ps:
	docker-compose ps

clean:
	@echo "Removing all containers and volumes..."
	docker-compose down -v
	@echo "Cleaning up dangling images..."
	docker image prune -f

test:
	@echo "Running health checks..."
	@echo ""
	@echo "API Gateway:"
	@curl -s http://localhost:8080/health | python -m json.tool || echo "API not ready"
	@echo ""
	@echo "Simulator:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Simulator not ready"
	@echo ""
	@echo "Model:"
	@curl -s http://localhost:8002/health | python -m json.tool || echo "Model not ready"
	@echo ""
	@echo "Threshold:"
	@curl -s http://localhost:8003/health | python -m json.tool || echo "Threshold not ready"
	@echo ""
	@echo "Ingestion:"
	@curl -s http://localhost:8001/health | python -m json.tool || echo "Ingestion not ready"

model:
	@echo "Generating dummy model..."
	cd services/model && python train_dummy_model.py

scale-ingest:
	@echo "Scaling ingestion service to 3 replicas..."
	docker-compose up -d --scale ingestion=3
