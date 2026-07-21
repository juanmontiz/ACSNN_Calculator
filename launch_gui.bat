@echo off
setlocal EnableDelayedExpansion

REM Launcher for the Atomic Cross Section Calculator GUI
set PYTHON_CMD=
for %%C in (python python3) do (
    %%C --version >nul 2>&1
    if not errorlevel 1 if not defined PYTHON_CMD set PYTHON_CMD=%%C
)

if not defined PYTHON_CMD (
    echo Python is not installed. Please install Python 3.10 or higher.
    pause
    exit /b 1
)

REM --- ROBUST VERSION CHECK (Native Batch, no Python quoting issues) ---
set PY_MAJOR=0
set PY_MINOR=0
for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PY_VERSION=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

set VERSION_STATUS=new
if %PY_MAJOR% LSS 3 set VERSION_STATUS=old
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 set VERSION_STATUS=old
REM ---------------------------------------------------------------------

set USE_VENV=false
if "%VERSION_STATUS%"=="old" (
    REM ESCAPED PARENTHESES HERE to prevent the ". was unexpected" error!
    echo Detected Python %PY_VERSION% ^(below 3.10^).
    set /p REPLY=Python 3.10 virtual environment is required. Create it now? [Y/n]
    if "!REPLY!"=="" set USE_VENV=true
    if /I "!REPLY!"=="y" set USE_VENV=true
    if /I "!REPLY!"=="yes" set USE_VENV=true
    if "!USE_VENV!"=="false" (
        echo Cannot continue without creating a Python 3.10 virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Detected Python %PY_VERSION%.
    set /p REPLY=Create/use a Python 3.10 virtual environment? [y/N]
    if /I "!REPLY!"=="y" set USE_VENV=true
    if /I "!REPLY!"=="yes" set USE_VENV=true
)

if "!USE_VENV!"=="true" (
    py -3.10 --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.10 not found. Please install Python 3.10 to create the virtual environment.
        echo Tip: Download from https://www.python.org/downloads/release/python-31011/
        if "%VERSION_STATUS%"=="old" (
            pause
            exit /b 1
        )
        set /p REPLY=Continue without a virtual environment? [Y/n]
        if /I "!REPLY!"=="n" exit /b 1
        if /I "!REPLY!"=="no" exit /b 1
        set USE_VENV=false
    ) else (
        set VENV_PY=py -3.10
    )
)

if "!USE_VENV!"=="true" (
    if not exist "venv" (
        echo Creating virtual environment with py -3.10...
        %VENV_PY% -m venv venv
    ) else (
        echo Using existing virtual environment in venv
    )
    call venv\Scripts\activate.bat
    set RUN_PY=python
) else (
    set RUN_PY=%PYTHON_CMD%
)

echo Installing dependencies...
%RUN_PY% -m pip install -r requirements.txt
echo Starting Atomic Cross Section Calculator...
%RUN_PY% acsnn_gui.py
pause
