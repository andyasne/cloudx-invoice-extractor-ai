@echo off
REM ==============================================================================
REM Copy Latest Invoices - Batch File Wrapper
REM ==============================================================================
REM This script calls the Python script to copy the latest M invoices from a
REM source folder to the data\raw\invoices folder.
REM
REM Usage:
REM     copy_latest_invoices.bat <source_folder> <number_of_invoices> [--clear]
REM
REM Example:
REM     copy_latest_invoices.bat "C:\invoices\2024" 1000
REM     copy_latest_invoices.bat "C:\invoices\2024" 1000 --clear
REM ==============================================================================

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check arguments
if "%~1"=="" (
    echo Usage: copy_latest_invoices.bat ^<source_folder^> ^<number_of_invoices^> [--clear]
    echo.
    echo Example:
    echo   copy_latest_invoices.bat "C:\invoices\2024" 1000
    echo   copy_latest_invoices.bat "C:\invoices\2024" 1000 --clear
    echo.
    echo Options:
    echo   --clear    Clear the destination folder before copying
    pause
    exit /b 1
)

if "%~2"=="" (
    echo Usage: copy_latest_invoices.bat ^<source_folder^> ^<number_of_invoices^> [--clear]
    echo.
    echo Example:
    echo   copy_latest_invoices.bat "C:\invoices\2024" 1000
    echo   copy_latest_invoices.bat "C:\invoices\2024" 1000 --clear
    echo.
    echo Options:
    echo   --clear    Clear the destination folder before copying
    pause
    exit /b 1
)

REM Call the Python script
python "%SCRIPT_DIR%copy_latest_invoices.py" %*

REM Pause if there was an error
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo Press any key to exit...
pause >nul
exit /b 0
