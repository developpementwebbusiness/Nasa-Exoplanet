start cmd /k docker compose up --build
timeout /t 20 >nul
start "" http://localhost:3000/