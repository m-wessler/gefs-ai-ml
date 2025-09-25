@echo off
REM Quick test version - 3 stations, 1 model, quick mode
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "TARGET_VARIABLE=tmax"
set "MODELS=lightgbm"
set "MAX_STATIONS=3"
set "DATA_DIR=N:\data\gefs-ml\forecast\f024"
set "FORECAST_HOURS=f024"

echo === QUICK MULTI-STATION TEST ===
echo Testing %MAX_STATIONS% stations with %MODELS% model
echo.

REM Auto-discover stations
set "STATIONS="
set /a STATION_COUNT=0

for %%f in ("%DATA_DIR%\*_2020_2025_f024.csv") do (
    if !STATION_COUNT! geq %MAX_STATIONS% goto :station_limit_reached
    
    set "filename=%%~nf"
    set "station_id=!filename:~0,4!"
    
    echo !STATIONS! | findstr /C:"!station_id!" >nul
    if !errorlevel! neq 0 (
        if "!STATIONS!"=="" (
            set "STATIONS=!station_id!"
        ) else (
            set "STATIONS=!STATIONS! !station_id!"
        )
        set /a STATION_COUNT+=1
        echo   Found station: !station_id!
    )
)

:station_limit_reached
echo.
echo Running %STATION_COUNT% stations: %STATIONS%
echo.

REM Run each station
set /a CURRENT=0
for %%s in (%STATIONS%) do (
    set /a CURRENT+=1
    echo [!CURRENT!/%STATION_COUNT%] Running %%s...
    
    call "%SCRIPT_DIR%run_single_station.bat" %%s f024 tmax lightgbm
    
    if !errorlevel! equ 0 (
        echo   ✓ SUCCESS: %%s
    ) else (
        echo   ✗ FAILED: %%s
    )
    echo.
)

echo === TEST COMPLETE ===