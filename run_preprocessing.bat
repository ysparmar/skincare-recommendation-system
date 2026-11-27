@echo off
REM run_preprocessing.bat — Run all preprocessing pipelines
echo Running all preprocessing pipelines...
cd /d "d:\Anti Gravity\ML Project"
.venv\Scripts\python.exe ML/preprocessing/preprocess_acne04.py
IF %ERRORLEVEL% NEQ 0 goto :error
.venv\Scripts\python.exe ML/preprocessing/preprocess_products.py
IF %ERRORLEVEL% NEQ 0 goto :error
.venv\Scripts\python.exe ML/preprocessing/preprocess_treatment.py
IF %ERRORLEVEL% NEQ 0 goto :error
echo All preprocessing complete!
goto :end
:error
echo Error during preprocessing. Check output above.
:end
pause
