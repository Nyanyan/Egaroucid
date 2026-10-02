@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "MODEL_CONFIG=w512_d4_a0.03:512:4:0.03:0.0:0.0:0.001"
set "MODEL_NAME=dense512"
set "DEFAULT_BATCH_SIZE=8192"
call "%~dp0train_prepared.bat" %*
exit /b %ERRORLEVEL%
