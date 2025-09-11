@echo off

echo Installing NaviGraph with UV...

REM Check if UV is installed
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo UV not found. Installing UV...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    REM Check if UV is now available
    where uv >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo UV installation failed. Please install UV manually:
        echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        exit /b 1
    )
    
    echo UV installed successfully!
) else (
    echo UV already installed
)

REM Install the project with uv sync
echo Installing NaviGraph with uv sync...
uv sync

echo.
echo Installation complete!
echo.
echo You can now use NaviGraph:
echo   uv run navigraph --help
echo   uv run navigraph setup graph config.yaml
echo   uv run navigraph run config.yaml
echo.
echo Or activate the environment and use directly:
echo   .venv\Scripts\activate
echo   navigraph --help