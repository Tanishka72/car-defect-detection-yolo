@echo off
REM Launch IMAGE ANALYZER Streamlit App

cd /d "c:\Users\sharm\OneDrive\Documents\Projects\IMAGE ANALYZER"

REM Activate virtual environment
call .venv-3\Scripts\activate.bat

REM Run Streamlit app
python -m streamlit run app.py

pause
