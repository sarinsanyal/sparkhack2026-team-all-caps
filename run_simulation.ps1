Write-Host "Starting Central FL Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit -Command `"python -m server.server`""

# Wait 3 seconds to let the server fully boot up before clients try to connect
Start-Sleep -Seconds 3

Write-Host "Starting Hospital Clients..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"python -m clients.client --hospital-id 1`""
Start-Process powershell -ArgumentList "-NoExit -Command `"python -m clients.client --hospital-id 2`""
Start-Process powershell -ArgumentList "-NoExit -Command `"python -m clients.client --hospital-id 3`""

# Wait 2 seconds for clients to connect and start training
Start-Sleep -Seconds 2

Write-Host "Opening Live Dashboard..." -ForegroundColor Magenta
# This will automatically open the HTML file in your default web browser
Start-Process "dashboard\dashboard.html"