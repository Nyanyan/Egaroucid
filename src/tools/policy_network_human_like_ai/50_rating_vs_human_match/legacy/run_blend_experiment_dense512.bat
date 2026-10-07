@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Model runs and default settings are pinned in run_model_blend_experiment.py.
rem Arguments are forwarded, e.g. --dry-run or --stage report.
rem Use the same command again to resume. Set POLICY_NO_PAUSE=1 for scripts.
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
pushd "%~dp0.."
if errorlevel 1 exit /b 1
"%POLICY_PYTHON%" -u "%~dp0..\run_model_blend_experiment.py" --models dense512 %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [ERROR] Experiment stopped. Check the console and output logs.
popd
if not defined POLICY_NO_PAUSE pause
exit /b %EXIT_CODE%
