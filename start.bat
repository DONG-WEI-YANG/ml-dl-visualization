@echo off
title ML/DL Visualization Platform - Launcher

echo ============================================
echo   ML/DL Visualization Platform
echo ============================================
echo.

:: -- Start Backend (FastAPI) --
echo [1/2] Starting Backend (FastAPI) ...
start "Backend - FastAPI" cmd /k "cd /D %~dp0platform\backend && call .venv\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Wait for backend to initialize
timeout /t 3 /nobreak >nul

:: -- Start Frontend (Vite) --
echo [2/2] Starting Frontend (Vite) ...
start "Frontend - Vite" cmd /k "cd /D %~dp0platform\frontend && npm run dev"

echo.
echo ============================================
echo   Frontend : http://localhost:5173
echo   Backend  : http://localhost:8000
echo   API Docs : http://localhost:8000/docs
echo ============================================
echo.
echo Closing this window will NOT stop the services.
echo To stop, close each cmd window or run stop.bat
echo.
pause
