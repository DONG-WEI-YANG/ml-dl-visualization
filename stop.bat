@echo off
title ML/DL Visualization Platform - Stop

echo ============================================
echo   Stopping all services
echo ============================================
echo.

:: Stop uvicorn (Python)
taskkill /f /fi "WINDOWTITLE eq Backend - FastAPI*" >nul 2>&1
echo [OK] Backend stopped

:: Stop Vite (Node)
taskkill /f /fi "WINDOWTITLE eq Frontend - Vite*" >nul 2>&1
echo [OK] Frontend stopped

echo.
echo All services stopped.
pause
