@echo off
echo =======================================================
echo Setting up InternIQ Daily Refresh (Windows Task Scheduler)
echo =======================================================

:: Get the current directory and the path to the daily_refresh.py script
set SCRIPT_DIR=%~dp0
set TARGET_SCRIPT=%SCRIPT_DIR%daily_refresh.py

:: Register the task to run daily at 18:00 (6:00 PM)
schtasks /create /tn "InternIQ_Daily_Refresh" /tr "python \"%TARGET_SCRIPT%\" --now" /sc daily /st 18:00 /f

echo.
echo Setup Complete! The InternIQ data will refresh automatically every day at 6:00 PM.
echo To run it immediately for testing, you can execute:
echo python "%TARGET_SCRIPT%" --now
pause
