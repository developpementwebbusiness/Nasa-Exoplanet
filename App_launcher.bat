start cmd /k "docker compose up --build" 
:waitloop
timeout /t 10 >nul
powershell -Command "(Invoke-WebRequest -Uri http://localhost:3000 -UseBasicParsing).StatusCode" >nul 2>nul
if errorlevel 1 goto waitloop
start "" http://localhost:3000/