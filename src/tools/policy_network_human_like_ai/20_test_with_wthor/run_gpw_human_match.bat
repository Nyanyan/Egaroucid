@echo off
setlocal EnableExtensions DisableDelayedExpansion
title GPW 30 players - WTHOR human move agreement
chcp 65001 >nul
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%POLICY_PYTHON%" -u "%~dp0evaluate_gpw_human_match.py" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [ERROR] Evaluation stopped. Check the error above and run_status.json. Run the same command to resume.
if not defined POLICY_NO_PAUSE pause
exit /b %RESULT%
