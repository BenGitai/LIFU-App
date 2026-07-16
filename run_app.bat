@echo off
REM One-click launcher (Windows). Assumes the "lifu" conda env exists (see README).
call conda activate lifu
python "%~dp0app.py"
pause
