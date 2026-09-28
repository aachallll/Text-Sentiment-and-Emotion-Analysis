@echo off
title Text Sentiment and Emotion Analysis Server
color 0B
echo ======================================================================
echo    Starting Text Sentiment and Emotion Analysis Web Server
echo ======================================================================
echo.

:: Navigate to the directory of this batch file
cd /d "%~dp0"

:: If manage.py is in sentiment_emotion_analysis subfolder, enter it
if exist "sentiment_emotion_analysis\manage.py" (
    cd sentiment_emotion_analysis
)

echo Working directory: %CD%
echo.

:: Detect Python executable
set PY_CMD=
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=python
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=py
    ) else if exist "C:\Python314\python.exe" (
        set PY_CMD=C:\Python314\python.exe
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set PY_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python was not found on your system!
    echo Please ensure Python is installed and added to Windows PATH.
    pause
    exit /b 1
)

echo Using Python: %PY_CMD%
echo.
echo Server will be available at:
echo    http://127.0.0.1:8000/
echo.
echo Press CTRL+C to stop the server.
echo.

%PY_CMD% manage.py runserver 127.0.0.1:8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Port 8000 may be in use. Trying port 8080...
    %PY_CMD% manage.py runserver 127.0.0.1:8080
)

pause
