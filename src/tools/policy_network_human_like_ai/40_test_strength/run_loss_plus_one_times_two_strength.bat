@echo off
setlocal EnableExtensions DisableDelayedExpansion
title (disc loss + 1) x 2 - standalone strength matches
chcp 65001 >nul
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%POLICY_PYTHON%" -u "%~dp0run_loss_plus_one_times_two_strength.py" %*
set "RESULT=%ERRORLEVEL%"
echo Matches exit code: %RESULT%
if not "%RESULT%"=="0" echo [ERROR] Matches stopped. Check the error before resuming with the same command.
if not defined POLICY_NO_PAUSE pause
exit /b %RESULT%
