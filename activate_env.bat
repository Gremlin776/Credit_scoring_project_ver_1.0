@echo off
chcp 65001 >nul
echo ========================================
echo  Credit Scoring Project - Activation
echo ========================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo To deactivate run: deactivate
echo To run project: python run_project.py
echo.
cmd /k
