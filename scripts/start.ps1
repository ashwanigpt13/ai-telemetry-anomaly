# Quick Start Script for AI Telemetry Anomaly Detection System
# Run this script to build and start the entire system

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AI Telemetry Anomaly Detection System" -ForegroundColor Cyan
Write-Host "Quick Start Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
Write-Host "Checking Docker Compose..." -ForegroundColor Yellow
try {
    docker-compose version | Out-Null
    Write-Host "✓ Docker Compose is available" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker Compose not found. Please install Docker Compose." -ForegroundColor Red
    exit 1
}

# Generate dummy model if needed
Write-Host ""
Write-Host "Checking for model file..." -ForegroundColor Yellow
if (-Not (Test-Path "services\model\model.pt")) {
    Write-Host "Model file not found. Generating dummy model..." -ForegroundColor Yellow
    Push-Location services\model
    python train_dummy_model.py
    Pop-Location
    Write-Host "✓ Dummy model generated" -ForegroundColor Green
} else {
    Write-Host "✓ Model file exists" -ForegroundColor Green
}

# Build services
Write-Host ""
Write-Host "Building Docker images..." -ForegroundColor Yellow
docker-compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Build complete" -ForegroundColor Green

# Start services
Write-Host ""
Write-Host "Starting services..." -ForegroundColor Yellow
docker-compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to start services" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Services started" -ForegroundColor Green

# Wait for services to be healthy
Write-Host ""
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service health
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Service Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Endpoints" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Test API Gateway
Write-Host ""
Write-Host "API Gateway (http://localhost:8080):" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method Get -TimeoutSec 5
    Write-Host "✓ API Gateway is healthy" -ForegroundColor Green
    $response | ConvertTo-Json
} catch {
    Write-Host "✗ API Gateway not responding yet" -ForegroundColor Red
}

# Test Simulator
Write-Host ""
Write-Host "Simulator (http://localhost:8000):" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -TimeoutSec 5
    Write-Host "✓ Simulator is healthy" -ForegroundColor Green
    $response | ConvertTo-Json
} catch {
    Write-Host "✗ Simulator not responding yet" -ForegroundColor Red
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "System Ready!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Available Services:" -ForegroundColor Cyan
Write-Host "  • API Gateway:  http://localhost:8080" -ForegroundColor White
Write-Host "  • Simulator:    http://localhost:8000" -ForegroundColor White
Write-Host "  • Ingestion:    http://localhost:8001" -ForegroundColor White
Write-Host "  • Model:        http://localhost:8002" -ForegroundColor White
Write-Host "  • Threshold:    http://localhost:8003" -ForegroundColor White
Write-Host "  • MQTT Broker:  localhost:1883" -ForegroundColor White
Write-Host "  • Redis:        localhost:6379" -ForegroundColor White
Write-Host ""
Write-Host "Quick Commands:" -ForegroundColor Cyan
Write-Host "  View logs:      docker-compose logs -f" -ForegroundColor White
Write-Host "  Stop system:    docker-compose down" -ForegroundColor White
Write-Host "  Restart:        docker-compose restart" -ForegroundColor White
Write-Host ""
Write-Host "Try it out:" -ForegroundColor Cyan
Write-Host '  Invoke-RestMethod -Uri "http://localhost:8080/health"' -ForegroundColor White
Write-Host ""
