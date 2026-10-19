@echo off
setlocal
cd /d "%~dp0"
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
title Quest under 800 - verification, Elo and two graphs
"%POLICY_PYTHON%" -u ..\10_train_policy_network\run_with_resource_log.py --tee --log output\quest_under800_20261010_analysis\finalization.log --summary output\quest_under800_20261010_analysis\finalization_resource.json -- "%POLICY_PYTHON%" -u finish_experiment.py
set "RUN_RESULT=%ERRORLEVEL%"
echo.
echo Analysis exit code: %RUN_RESULT%
if not defined POLICY_NO_PAUSE pause
exit /b %RUN_RESULT%
