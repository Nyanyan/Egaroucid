@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Loss plus one - WTHOR blend accuracy
chcp 65001 >nul
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%POLICY_PYTHON%" -u "%~dp0evaluate_loss_plus_one_blends.py" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [ERROR] Evaluation stopped. Check the error above and run_status.json if created.
if not defined POLICY_NO_PAUSE pause
exit /b %RESULT%
