@echo off
setlocal

uv run pyinstaller -y --clean --noconfirm lumachords.spec
set "EXIT_CODE=%ERRORLEVEL%"

exit /b %EXIT_CODE%
