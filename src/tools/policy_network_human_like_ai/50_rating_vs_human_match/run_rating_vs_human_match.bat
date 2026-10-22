@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Default: five verified 24-model paper figures (PNG, SVG, PDF).
rem Explicit --input keeps the legacy report-JSON renderer available.
rem POLICY_PYTHON may point to a Python interpreter with numpy and matplotlib.
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "PYTHONIOENCODING=utf-8"
"%POLICY_PYTHON%" -u "%~dp0plot_rating_vs_human_match.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo [ERROR] Plotting failed. Check the input JSON and Python dependencies.
if not defined POLICY_NO_PAUSE pause
exit /b %EXIT_CODE%
