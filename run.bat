@echo off
echo Starting GeoIntel System...
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
