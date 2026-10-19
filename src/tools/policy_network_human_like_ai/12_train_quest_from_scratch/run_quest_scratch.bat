@echo off
setlocal
cd /d "%~dp0"
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
title Quest under 800 - scratch training and test evaluation
"%POLICY_PYTHON%" -u train_quest_scratch.py --resume
if errorlevel 1 goto failed
"%POLICY_PYTHON%" -u evaluate_scratch.py
if errorlevel 1 goto failed
echo.
echo COMPLETE: scratch training and both test evaluations succeeded.
if not defined POLICY_NO_PAUSE pause
exit /b 0
:failed
set "RUN_RESULT=%ERRORLEVEL%"
echo.
echo Experiment failed with exit code %RUN_RESULT%. Saved checkpoints and predictions can be resumed.
if not defined POLICY_NO_PAUSE pause
exit /b %RUN_RESULT%
