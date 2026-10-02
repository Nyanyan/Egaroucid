@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "MODEL_CONFIG=conv4"
set "MODEL_NAME=conv4"
set "DEFAULT_BATCH_SIZE=512"
call "%~dp0train_prepared.bat" %*
exit /b %ERRORLEVEL%
