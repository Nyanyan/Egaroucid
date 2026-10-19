@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
"%POLICY_PYTHON%" -u "%~dp0calculate_avg_disc_loss.py" %*
exit /b %ERRORLEVEL%
