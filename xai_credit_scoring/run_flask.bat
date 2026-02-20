@echo off
echo ======================================
echo   Starting FinTrust AI Flask API
echo ======================================
call "%~dp0..\.venv\Scripts\activate.bat"
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" flask_api.py
pause
