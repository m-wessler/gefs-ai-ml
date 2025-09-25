@echo off
REM Simple batch script runner that avoids conda PowerShell issues
REM Usage: run_single_station.bat STATION FORECAST_HOUR TARGET_VARIABLE

if "%~1"=="" (
    echo Usage: %0 ^<STATION^> ^<FORECAST_HOUR^> ^<TARGET_VARIABLE^>
    echo Example: %0 KSEA f024 tmax
    exit /b 1
)

set STATION=%~1
set FORECAST_HOUR=%~2
set TARGET_VARIABLE=%~3

echo === Running ML Training for %STATION% %FORECAST_HOUR% ===
echo Station: %STATION%
echo Forecast Hour: %FORECAST_HOUR%
echo Target Variable: %TARGET_VARIABLE%
echo.

REM Set encoding for Unicode support
set PYTHONIOENCODING=utf-8

REM Activate conda environment and run Python script
call conda activate gefsai
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate gefsai conda environment
    echo Please make sure the environment exists: conda env list
    exit /b 1
)

echo Environment activated successfully
echo.

REM Run the Python script
python "%~dp0single_gefs_ml_trainer.py" --station_id %STATION% --forecast_hours %FORECAST_HOUR% --target_variable %TARGET_VARIABLE% --quick_mode

if %errorlevel% equ 0 (
    echo.
    echo === SUCCESS: %STATION% %FORECAST_HOUR% completed ===
) else (
    echo.
    echo === FAILED: %STATION% %FORECAST_HOUR% failed with exit code %errorlevel% ===
)

exit /b %errorlevel%