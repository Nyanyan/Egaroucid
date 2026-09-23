@echo off
setlocal
set "DEEP_ROOT=%~dp0..\ignored\tmp\endgame_rearchitecture_20260922"
set "DEEP_PYTHON=C:\Users\yaman\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "DEEP_CHECK="
if "%~1"=="--check-only" set "DEEP_CHECK=--check-only"
if not "%~1"=="" if not "%~1"=="--check-only" goto usage
if not "%~2"=="" goto usage
if not exist "%DEEP_PYTHON%" (
    echo Python runtime is missing: %DEEP_PYTHON%
    exit /b 2
)
if not exist "%DEEP_ROOT%\compare_deep.py" (
    echo The prepared comparison files are missing: %DEEP_ROOT%
    exit /b 2
)
pushd "%DEEP_ROOT%" || exit /b 2
if "%DEEP_CHECK%"=="" (
    echo Deep endgame speed comparison: baseline / all changes / Edax 4.5.5
    echo FFO 60-79: all 20 positions, exact endgame search.
    echo MPC39 and MPC40: all 5 positions each, settings 0 and 1.
    echo Edax: level 60 for FFO; 32 and 33 for MPC39; 36 for MPC40.
    echo 28 search threads, hash 25. Engines run one at a time.
    echo 3 repetitions; extend unstable comparisons to 10, then at most 20.
    echo Warm-up runs are excluded. No MPC accuracy judgment is made.
    echo Keep the PC awake and avoid other heavy workloads.
    echo The comparison can take many hours; cumulative limit is 72 hours.
    echo Ctrl+C interrupts. Run this BAT again to resume completed runs.
    echo Results: %DEEP_ROOT%\deep_comparison_summary.csv
    echo.
)
"%DEEP_PYTHON%" -B -u compare_deep.py %DEEP_CHECK%
set "DEEP_EXIT=%ERRORLEVEL%"
popd
if not "%DEEP_CHECK%"=="" exit /b %DEEP_EXIT%
echo.
if "%DEEP_EXIT%"=="0" echo COMPLETE. Results and raw output have been saved.
if "%DEEP_EXIT%"=="130" echo INTERRUPTED. Run this BAT again to resume.
if not "%DEEP_EXIT%"=="0" if not "%DEEP_EXIT%"=="130" echo STOPPED on an error. Keep the files and ask Codex to inspect them.
pause
exit /b %DEEP_EXIT%
:usage
echo Usage: %~nx0 [--check-only]
exit /b 2
