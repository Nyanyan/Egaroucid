@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "MODEL_CONFIG=conv8"
set "MODEL_NAME=conv8"
set "DEFAULT_BATCH_SIZE=256"
call "%~dp0train_prepared.bat" %*
exit /b %ERRORLEVEL%
