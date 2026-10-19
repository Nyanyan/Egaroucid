@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Render saved report and bootstrap sources. No inference or tournament is started.
rem POLICY_PYTHON may point to a Python interpreter with numpy and matplotlib.
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONIOENCODING=utf-8"
"%POLICY_PYTHON%" -u "%~dp0plot_rating_vs_human_match.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [ERROR] Plotting failed. Check the input JSON and Python dependencies.
if not defined POLICY_NO_PAUSE pause
exit /b %EXIT_CODE%
