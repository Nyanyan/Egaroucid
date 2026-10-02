@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Shared launcher. Data paths are resolved by Python from environment variables.
if not defined MODEL_CONFIG set "MODEL_CONFIG=512x4"
if not defined MODEL_NAME set "MODEL_NAME=dense512"
if not defined DEFAULT_BATCH_SIZE set "DEFAULT_BATCH_SIZE=8192"
if not defined EPOCHS set "EPOCHS=50"
if not defined BATCH_SIZE set "BATCH_SIZE=%DEFAULT_BATCH_SIZE%"
if not defined SEED set "SEED=613"
if not defined POLICY_DATASET set "POLICY_DATASET=WTHOR_1977_2024"
if not defined POLICY_PYTHON set "POLICY_PYTHON=python"
set "EXIT_CODE=1"
pushd "%~dp0"
if errorlevel 1 exit /b 1

"%POLICY_PYTHON%" --version
if errorlevel 1 (
    echo [ERROR] Set POLICY_PYTHON to the Python executable with TensorFlow installed.
    goto finish
)
set "RUN_STAMP="
for /f %%T in ('powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "RUN_STAMP=%%T"
if not defined RUN_STAMP goto finish
set "RUN_NAME=prepared_%MODEL_NAME%_e%EPOCHS%_%RUN_STAMP%"
set "OUTPUT_DIR=trained\%RUN_NAME%"
set "LOG_FILE=train_log\%RUN_NAME%.log"
set "RESOURCE_FILE=train_log\%RUN_NAME%_resource.json"
if defined POLICY_TRAIN_OUTPUT set "OUTPUT_DIR=%POLICY_TRAIN_OUTPUT%"
if defined POLICY_TRAIN_LOG set "LOG_FILE=%POLICY_TRAIN_LOG%"
if defined POLICY_TRAIN_RESOURCE set "RESOURCE_FILE=%POLICY_TRAIN_RESOURCE%"
echo Training %MODEL_NAME% for %EPOCHS% epochs. Dataset: %POLICY_DATASET%
echo Progress: "%LOG_FILE%"
"%POLICY_PYTHON%" -u run_with_resource_log.py ^
    --log "%LOG_FILE%" --summary "%RESOURCE_FILE%" --tee ^
    -- "%POLICY_PYTHON%" -u train_policy_network.py ^
    --dataset "%POLICY_DATASET%" ^
    --configs "%MODEL_CONFIG%" ^
    --epochs %EPOCHS% --batch-size %BATCH_SIZE% --seed %SEED% ^
    --learning-rate 0.001 --l2 0.0 ^
    --predict-batch-size %BATCH_SIZE% ^
    --output-dir "%OUTPUT_DIR%" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Training or final test evaluation failed. Check the log above.
) else (
    echo Model: "%OUTPUT_DIR%\selected_model.h5"
    echo Final test metrics: "%OUTPUT_DIR%\test_metrics.json"
)
:finish
popd
if not defined POLICY_NO_PAUSE pause
exit /b %EXIT_CODE%
