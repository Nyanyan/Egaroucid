@echo off
setlocal
cd /d "%~dp0"
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
title Quest under 800 fine-tuned model vs 30 players
"%POLICY_PYTHON%" -u ..\10_train_policy_network\run_with_resource_log.py --tee --log output\quest_under800_20261010_analysis\strength.log --summary output\quest_under800_20261010_analysis\strength_resource.json -- "%POLICY_PYTHON%" -u run_strength_vs_30.py %*
set "RUN_RESULT=%ERRORLEVEL%"
echo.
echo Tournament exit code: %RUN_RESULT%
if not defined POLICY_NO_PAUSE pause
exit /b %RUN_RESULT%
