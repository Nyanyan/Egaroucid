@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Loss plus one - standalone strength matches
chcp 65001 >nul
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%POLICY_PYTHON%" -u "%~dp0run_loss_plus_one_strength.py" %*
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" echo [ERROR] Matches stopped. Run the same command again to resume after checking the error.
if not defined POLICY_NO_PAUSE pause
exit /b %RESULT%
