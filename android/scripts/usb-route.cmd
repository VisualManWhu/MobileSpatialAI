@echo off
rem Recon3D: route the phone's 127.0.0.1:8765 to this laptop's 8765 over the USB cable (adb reverse).
rem Re-run after every unplug/replug: the route does not survive a USB reconnect.
set ADB=%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe
if not exist "%ADB%" (
  echo adb not found at %ADB% - install Android Studio / platform-tools first.
  pause
  exit /b 1
)
"%ADB%" reverse tcp:8765 tcp:8765
if errorlevel 1 (
  echo Failed. Is the phone plugged in with USB debugging allowed?  ("%ADB%" devices)
  pause
  exit /b 1
)
echo USB route ready: in the Recon3D app use 127.0.0.1:8765 as the compute address.
pause
