@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "MODEL_CONFIG=conv6"
set "MODEL_NAME=conv6"
set "DEFAULT_BATCH_SIZE=256"
call "%~dp0train_prepared.bat" %*
exit /b %ERRORLEVEL%
