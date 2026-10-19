@echo off
setlocal EnableExtensions DisableDelayedExpansion
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
set "EXIT_CODE=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1
title WTHOR inference dropout - 5 rates and 7 Quest test bands
"%POLICY_PYTHON%" -u run_dropout_sweep.py %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Dropout sweep finished with exit code %EXIT_CODE%.
echo Evaluation logs, progress and results are in the output directory.
popd
if not defined POLICY_NO_PAUSE pause
exit /b %EXIT_CODE%
