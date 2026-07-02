@echo off
REM Windows startup script for Local RAG Agent
REM Ensures proper Python path setup before running Streamlit

echo Starting Local RAG Agent...
echo.

REM Change to script directory
cd /d "%~dp0"

REM Set PYTHONPATH to include project root
set PYTHONPATH=%~dp0;%PYTHONPATH%

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Warning: No virtual environment found. Using system Python.
)

REM Check Ollama
echo Checking Ollama status...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama does not appear to be running!
    echo Please start Ollama first: ollama serve
    echo.
    pause
)

REM Start Streamlit
echo Launching Streamlit UI...
streamlit run app.py

pause
