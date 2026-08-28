@echo off
cd /d "%~dp0"
echo Creating Desktop Shortcut for TaskMatch AI...
powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([System.IO.Path]::Combine([System.Environment]::GetFolderPath('Desktop'), 'TaskMatch AI.lnk')); $s.TargetPath = [System.IO.Path]::Combine('%~dp0', 'start.bat'); $s.WorkingDirectory = '%~dp0'; $s.Description = 'TaskMatch AI Local LLM Benchmark Studio'; $s.Save()"
echo.
echo [✓] Shortcut 'TaskMatch AI' created on your Desktop!
echo.
pause
