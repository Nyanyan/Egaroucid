@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%POLICY_PYTHON%" -u "%~dp0run_gpw_policy4_strength.py" %*
exit /b %ERRORLEVEL%
