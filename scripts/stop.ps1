# Stop Script for AI Telemetry Anomaly Detection System# Stop Script for AI Telemetry Anomaly Detection System






















Write-Host ""Write-Host "  docker-compose down -v" -ForegroundColor WhiteWrite-Host "To remove all data volumes, run:" -ForegroundColor YellowWrite-Host ""}    exit 1    Write-Host "✗ Failed to stop services" -ForegroundColor Red} else {    Write-Host "✓ All services stopped successfully" -ForegroundColor Greenif ($LASTEXITCODE -eq 0) {docker-compose downWrite-Host "Stopping all services..." -ForegroundColor Yellow# Stop all servicesWrite-Host ""Write-Host "========================================" -ForegroundColor CyanWrite-Host "Stopping AI Telemetry Anomaly Detection System" -ForegroundColor CyanWrite-Host "========================================" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Stopping All Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Stop services
Write-Host "Stopping containers..." -ForegroundColor Yellow
docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All services stopped" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to stop some services" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "To remove volumes (data will be lost), run:" -ForegroundColor Yellow
Write-Host "  docker-compose down -v" -ForegroundColor White
Write-Host ""
