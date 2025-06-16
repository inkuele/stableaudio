@echo off
cd /d %~dp0\..\

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run Python script
python run_gradio_offline.py

pause
