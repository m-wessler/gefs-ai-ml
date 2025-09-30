@echo off
REM GEFS ML Training Pipeline - Multiple Stations Runner (Windows)
REM This script runs the single_gefs_ml_trainer.py for multiple stations and forecast hours
REM Usage: run_multiple_stations.bat

setlocal enabledelayedexpansion

REM Set script directory
set "SCRIPT_DIR=%~dp0"
set "PYTHON_SCRIPT=%SCRIPT_DIR%single_gefs_ml_trainer.py"

REM Check if Python script exists
if not exist "%PYTHON_SCRIPT%" (
    echo Error: Python script not found at %PYTHON_SCRIPT%
    exit /b 1
)

REM Configuration
set "TARGET_VARIABLE=tmin"
set "USE_GPU=false"
set "QUICK_MODE=false"

REM Station limit for testing (set to 0 for unlimited)
set /a MAX_STATIONS=0

REM Data directory for automatic station discovery
set "DATA_DIR=N:\data\gefs-ml\forecast\f024"

REM Define forecast hours to process (space-separated)
set "FORECAST_HOURS=f024"

REM Auto-discover stations from forecast directory
echo Discovering stations from: %DATA_DIR%
set "STATIONS="
set /a STATION_COUNT=0

if not exist "%DATA_DIR%" (
    echo Error: Data directory not found: %DATA_DIR%
    echo Please check the path and try again.
    exit /b 1
)

echo Scanning for station files...
if %MAX_STATIONS% gtr 0 (
    echo Station limit: %MAX_STATIONS%
) else (
    echo Station limit: Unlimited
)

for %%f in ("%DATA_DIR%\*_2020_2025_f024.csv") do (
    REM Check if we've reached the station limit
    if %MAX_STATIONS% gtr 0 if !STATION_COUNT! geq %MAX_STATIONS% goto :station_limit_reached
    
    set "filename=%%~nf"
    REM Extract first 4 characters as station ID
    set "station_id=!filename:~0,4!"
    
    REM Add to stations list if not already present
    echo !STATIONS! | findstr /C:"!station_id!" >nul
    if !errorlevel! neq 0 (
        if "!STATIONS!"=="" (
            set "STATIONS=!station_id!"
        ) else (
            set "STATIONS=!STATIONS! !station_id!"
        )
        set /a STATION_COUNT+=1
        echo   Found station: !station_id! ^(from %%~nxf^)
    )
)

:station_limit_reached

if %STATION_COUNT% equ 0 (
    echo Error: No station files found in %DATA_DIR%
    echo Expected files in format: STID_2020_2025_f024.csv
    exit /b 1
)

if %MAX_STATIONS% gtr 0 if %STATION_COUNT% geq %MAX_STATIONS% (
    echo Station limit reached: %STATION_COUNT%/%MAX_STATIONS%
) else (
    echo Found %STATION_COUNT% stations: %STATIONS%
)

REM Create logs directory
if not exist "%SCRIPT_DIR%..\logs" mkdir "%SCRIPT_DIR%..\logs"

REM Log file with timestamp
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i
set "timestamp=%datetime:~0,8%_%datetime:~8,6%"
set "LOG_FILE=%SCRIPT_DIR%..\logs\multiple_stations_%timestamp%.log"

echo === GEFS ML Training - Multiple Stations Runner === > "%LOG_FILE%"
echo Started at: %date% %time% >> "%LOG_FILE%"
echo Target Variable: %TARGET_VARIABLE% >> "%LOG_FILE%"
echo GPU Acceleration: %USE_GPU% >> "%LOG_FILE%"
echo Quick Mode: %QUICK_MODE% >> "%LOG_FILE%"
echo Models: %MODELS% >> "%LOG_FILE%"
echo Stations: %STATIONS% >> "%LOG_FILE%"
echo Forecast Hours: %FORECAST_HOURS% >> "%LOG_FILE%"
echo Log file: %LOG_FILE% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo === GEFS ML Training - Multiple Stations Runner ===
echo Started at: %date% %time%
echo Target Variable: %TARGET_VARIABLE%
echo GPU Acceleration: %USE_GPU%
echo Quick Mode: %QUICK_MODE%
echo Models: %MODELS%
echo Stations: %STATIONS%
echo Forecast Hours: %FORECAST_HOURS%
echo Log file: %LOG_FILE%
echo.

REM Initialize counters
set /a TOTAL_RUNS=0
set /a SUCCESSFUL_RUNS=0
set /a FAILED_RUNS=0

REM Count total expected runs
REM STATION_COUNT already calculated during discovery

set /a FHR_COUNT=0
for %%f in (%FORECAST_HOURS%) do set /a FHR_COUNT+=1

set /a TOTAL_EXPECTED=STATION_COUNT*FHR_COUNT

echo Total expected runs: %TOTAL_EXPECTED%
echo Total expected runs: %TOTAL_EXPECTED% >> "%LOG_FILE%"
echo.
echo. >> "%LOG_FILE%"

REM Main execution loop
echo Starting training runs...
echo Starting training runs... >> "%LOG_FILE%"
echo.
echo. >> "%LOG_FILE%"

for %%s in (%STATIONS%) do (
    for %%f in (%FORECAST_HOURS%) do (
        set /a TOTAL_RUNS+=1
        
        echo === Run !TOTAL_RUNS!/%TOTAL_EXPECTED%: %%s %%f ===
        echo === Run !TOTAL_RUNS!/%TOTAL_EXPECTED%: %%s %%f ===
        echo === Run !TOTAL_RUNS!/%TOTAL_EXPECTED%: %%s %%f === >> "%LOG_FILE%"
        echo Started at: %date% %time%
        echo Started at: %date% %time% >> "%LOG_FILE%"
        
        REM Build command using the reliable single station runner
        set "CMD=call "%SCRIPT_DIR%run_single_station.bat" %%s %%f %TARGET_VARIABLE%"
        
        echo Command: !CMD!
        echo Command: !CMD! >> "%LOG_FILE%"
        echo.
        echo. >> "%LOG_FILE%"
        
        REM Run the command (let it handle its own output)
        !CMD!
        
        if !errorlevel! equ 0 (
            echo [SUCCESS] %%s %%f completed
            echo [SUCCESS] %%s %%f completed >> "%LOG_FILE%"
            set /a SUCCESSFUL_RUNS+=1
        ) else (
            echo [FAILED] %%s %%f failed
            echo [FAILED] %%s %%f failed >> "%LOG_FILE%"
            set /a FAILED_RUNS+=1
        )
        
        echo Completed at: %date% %time%
        echo Completed at: %date% %time% >> "%LOG_FILE%"
        echo.
        echo. >> "%LOG_FILE%"
        
        REM Small delay between runs
        timeout /t 2 /nobreak >nul
    )
)

REM Final summary
echo.
echo. >> "%LOG_FILE%"
echo === FINAL SUMMARY ===
echo === FINAL SUMMARY === >> "%LOG_FILE%"
echo Completed at: %date% %time%
echo Completed at: %date% %time% >> "%LOG_FILE%"
echo Total runs attempted: %TOTAL_RUNS%
echo Total runs attempted: %TOTAL_RUNS% >> "%LOG_FILE%"
echo Successful runs: %SUCCESSFUL_RUNS%
echo Successful runs: %SUCCESSFUL_RUNS% >> "%LOG_FILE%"
echo Failed runs: %FAILED_RUNS%
echo Failed runs: %FAILED_RUNS% >> "%LOG_FILE%"

if %FAILED_RUNS% equ 0 (
    echo All runs completed successfully!
    echo All runs completed successfully! >> "%LOG_FILE%"
) else (
    echo Some runs failed. Check the log file for details.
    echo Some runs failed. Check the log file for details. >> "%LOG_FILE%"
)

echo.
echo Script completed.
exit /b 0