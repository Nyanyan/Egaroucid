@echo off
setlocal
cd /d "%~dp0"
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUNBUFFERED=1"
title WTHOR to Othello Quest fine-tuning
"%POLICY_PYTHON%" -u fine_tune_policy.py %*
set "TRAIN_RESULT=%ERRORLEVEL%"
echo.
echo Fine-tuning exit code: %TRAIN_RESULT%
if not defined POLICY_NO_PAUSE pause
exit /b %TRAIN_RESULT%
