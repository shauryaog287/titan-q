# resume_download.ps1
# ===================
# Run by the scheduled task TitanQ_Angel_History (16:05 every day, no catch-up at boot, so it
# never starts during a trading session). It restarts whatever is not running:
#   - angel_history.py: the resumable 1-minute history download (it pauses itself 08:55-15:50
#     on trading days and exits if another copy runs);
#   - after_download.ps1: the research chain that waits for the research set, then runs the
#     M1 and VPIN development studies once each.
# Added 2026-10-01: the laptop is shut down at times, and these were one-off processes.

$proj = 'D:\Quant_Project'
Set-Location $proj
$procs = Get-CimInstance Win32_Process
if (-not ($procs | Where-Object { $_.Name -like 'python*.exe' -and $_.CommandLine -like '*angel_history.py*' })) {
    Start-Process -FilePath 'D:\trader_env\Scripts\python.exe' -ArgumentList 'angel_history.py' -WorkingDirectory $proj `
        -RedirectStandardOutput (Join-Path $proj 'logs\angel_history.out.log') `
        -RedirectStandardError (Join-Path $proj 'logs\angel_history.err.log') -WindowStyle Hidden
}
if (-not ($procs | Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*after_download.ps1*' })) {
    Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        (Join-Path $proj 'after_download.ps1')
}
