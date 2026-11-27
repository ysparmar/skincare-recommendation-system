@echo off
REM run_backend.bat — Start the Flask backend server
echo Starting SkinSenseAI Flask backend...
cd /d "d:\Anti Gravity\ML Project"
.venv\Scripts\python.exe backend/app.py
pause
