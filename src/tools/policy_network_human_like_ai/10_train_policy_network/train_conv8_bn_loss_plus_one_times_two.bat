@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Conv8+BN training - (disc loss + 1) x 2
chcp 65001 >nul
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%POLICY_PYTHON%" -u "%~dp0train_loss_plus_one_times_two.py" %*
set "RESULT=%ERRORLEVEL%"
echo Training exit code: %RESULT%
if not "%RESULT%"=="0" echo [ERROR] Training stopped. See the run directory and logs above.
if not defined POLICY_NO_PAUSE pause
exit /b %RESULT%
