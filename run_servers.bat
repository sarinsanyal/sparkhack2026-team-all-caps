@echo off
TITLE Central FL System Launcher

echo Cleaning up old logs...
if exist "logs\rounds.json" del /f /q "logs\rounds.json"

echo Starting Central FL Server...
start "FL Server" cmd /k "python -m server.server"

timeout /t 3 /nobreak > nul

echo Starting Hospital Clients...
start "Hospital Client 1" cmd /k "python -m clients.client --hospital-id 1"
start "Hospital Client 2" cmd /k "python -m clients.client --hospital-id 2"
start "Hospital Client 3" cmd /k "python -m clients.client --hospital-id 3"

echo.
echo Waiting for Round 1 to complete and "rounds.json" to be generated...
echo (This may take a minute depending on your training epochs)

:WAIT_FOR_FILE
if not exist "logs\rounds.json" (
    :: Wait 2 seconds before checking again to save CPU
    timeout /t 10 /nobreak > nul
    goto WAIT_FOR_FILE
)

echo.
echo Data detected! Opening Live Dashboard...
start "" "http://127.0.0.1:3000/dashboard/dashboard.html"

echo All systems initiated.
pause