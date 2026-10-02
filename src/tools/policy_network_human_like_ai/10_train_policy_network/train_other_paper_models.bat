@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Train the remaining paper architectures sequentially: Conv4, Conv6, Conv8.
rem Each child BAT controls its epochs and batch size and saves a separate run.
rem Stop immediately on failure. Set POLICY_NO_PAUSE=1 for unattended use.
set "PAUSE_AT_END=1"
if defined POLICY_NO_PAUSE set "PAUSE_AT_END="
set "POLICY_NO_PAUSE=1"
set "EXIT_CODE=0"

echo [1/3] Training Conv4
call "%~dp0train_conv4.bat"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto finish

echo [2/3] Training Conv6
call "%~dp0train_conv6.bat"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto finish

echo [3/3] Training Conv8
call "%~dp0train_conv8.bat"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto finish

echo All three paper models completed.

:finish
if not "%EXIT_CODE%"=="0" echo [ERROR] Training stopped with exit code %EXIT_CODE%. Later models were not started.
if defined PAUSE_AT_END pause
exit /b %EXIT_CODE%
