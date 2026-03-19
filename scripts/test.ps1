# Test Script for AI Telemetry Anomaly Detection System

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing System Endpoints" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test all service health endpoints
$services = @(
    @{Name="API Gateway"; Port=8080},
    @{Name="Simulator"; Port=8000},
    @{Name="Ingestion"; Port=8001},
    @{Name="Model"; Port=8002},
    @{Name="Threshold"; Port=8003}
)

Write-Host "Health Checks:" -ForegroundColor Yellow
Write-Host ""

foreach ($service in $services) {
    $url = "http://localhost:$($service.Port)/health"
    try {
        $response = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 5
        Write-Host "✓ $($service.Name) (port $($service.Port))" -ForegroundColor Green
        Write-Host "  $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
    } catch {
        Write-Host "✗ $($service.Name) (port $($service.Port)) - Not responding" -ForegroundColor Red
    }
    Write-Host ""
}

# Test simulator configuration
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Simulator Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    $config = Invoke-RestMethod -Uri "http://localhost:8000/config" -Method Get -TimeoutSec 5
    Write-Host "Current configuration:" -ForegroundColor Yellow
    $config | ConvertTo-Json -Depth 3
    Write-Host ""
} catch {
    Write-Host "✗ Cannot retrieve simulator config" -ForegroundColor Red
    Write-Host ""
}

# Test prediction endpoint with sample data
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Prediction Endpoint" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$sampleWindow = @()
for ($i = 0; $i -lt 50; $i++) {
    $sampleWindow += ,@(
        [Math]::Round((Get-Random -Minimum -1 -Maximum 1), 3),
        [Math]::Round((Get-Random -Minimum -1 -Maximum 1), 3),
        [Math]::Round((Get-Random -Minimum -1 -Maximum 1), 3)
    )
}

$predictRequest = @{
    entity_id = "test_engine"
    window = $sampleWindow
} | ConvertTo-Json -Depth 10

try {
    Write-Host "Sending prediction request for entity: test_engine" -ForegroundColor Yellow
    $prediction = Invoke-RestMethod -Uri "http://localhost:8080/predict" -Method Post -Body $predictRequest -ContentType "application/json" -TimeoutSec 10
    Write-Host "✓ Prediction successful" -ForegroundColor Green
    Write-Host ""
    Write-Host "Result:" -ForegroundColor Cyan
    $prediction | ConvertTo-Json
    Write-Host ""
} catch {
    Write-Host "✗ Prediction failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# Test entity state endpoint
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing Entity State Endpoint" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

try {
    $entityState = Invoke-RestMethod -Uri "http://localhost:8080/entity/test_engine" -Method Get -TimeoutSec 5
    Write-Host "✓ Entity state retrieved" -ForegroundColor Green
    Write-Host ""
    Write-Host "State:" -ForegroundColor Cyan
    $entityState | ConvertTo-Json
    Write-Host ""
} catch {
    Write-Host "✗ Entity state retrieval failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# Show container status
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Container Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
docker-compose ps

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view logs: docker-compose logs -f [service-name]" -ForegroundColor White
Write-Host "To restart:   docker-compose restart [service-name]" -ForegroundColor White
Write-Host ""
