@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "MODEL_CONFIG=conv8_bn"
set "MODEL_NAME=conv8_bn"
set "DEFAULT_BATCH_SIZE=256"
call "%~dp0train_prepared.bat" %*
exit /b %ERRORLEVEL%
