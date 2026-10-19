@echo off
setlocal
cd /d "%~dp0"
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
title Quest under 800 - resume cached evaluation
"%POLICY_PYTHON%" -u ..\10_train_policy_network\run_with_resource_log.py --tee --log output\quest_under800_20261010_analysis\quest_baselines.log --summary output\quest_under800_20261010_analysis\quest_baselines_resource.json -- "%POLICY_PYTHON%" -u resume_quest_evaluation.py %*
set "RUN_RESULT=%ERRORLEVEL%"
echo.
echo Evaluation exit code: %RUN_RESULT%
if not defined POLICY_NO_PAUSE pause
exit /b %RUN_RESULT%
