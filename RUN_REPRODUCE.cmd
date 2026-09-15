@echo off
setlocal
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" goto use_venv
where py >nul 2>nul
if not errorlevel 1 goto use_py
where python >nul 2>nul
if not errorlevel 1 goto use_python
echo Python not found. Read README.md and install Python 3.11-3.13 first.
goto failed
:use_venv
.venv\Scripts\python.exe scripts\reproduce.py
if errorlevel 1 goto failed
goto done
:use_py
py -3 scripts\reproduce.py
if errorlevel 1 goto failed
goto done
:use_python
python scripts\reproduce.py
if errorlevel 1 goto failed
goto done
:failed
echo Check the error above. For a custom environment, run python scripts\reproduce.py manually.
echo No frozen scientific results were overwritten.
popd
pause
exit /b 1
:done
echo Replay finished. Read the new REPORT.md under outputs.
popd
pause
exit /b 0
