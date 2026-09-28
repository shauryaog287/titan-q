# after_download.ps1
# ==================
# Unattended follow-up to the Angel 1-minute history download. Start it detached:
#
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\Quant_Project\after_download.ps1'
#
# Steps, each logged to logs\after_download.log:
#   1. wait until no angel_history.py process is running;
#   2. run angel_history.py twice more to fetch any chunk that failed (existing files are
#      skipped; the downloader itself pauses 08:55-15:50 IST on weekdays);
#   3. run intraday_research.py (M1) on DEVELOPMENT data only, at below-normal priority,
#      and commit its report (reports/m1_intraday.*) so the decision is recorded before
#      any test run. It never passes --reveal-test;
#   4. wait until the market is closed, then back up to Google Drive (backup_to_drive.py),
#      so the multi-GB upload never competes with the live feed.
# Nothing is deleted.

$ErrorActionPreference = 'Continue'
$proj = 'D:\Quant_Project'
$py = 'D:\trader_env\Scripts\python.exe'
$git = 'D:\Git\cmd\git.exe'
$logFile = Join-Path $proj 'logs\after_download.log'
Set-Location $proj
$env:PYTHONIOENCODING = 'ascii:replace'
$env:GIT_TERMINAL_PROMPT = '0'

function Log($msg) {
    $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' - ' + $msg
    Add-Content -Path $logFile -Value $line -Encoding ascii
}

function DownloaderRunning {
    $p = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -like '*angel_history.py*' }
    return [bool]$p
}

function MarketHours {
    $n = Get-Date
    $weekday = $n.DayOfWeek -ne 'Saturday' -and $n.DayOfWeek -ne 'Sunday'
    $t = $n.TimeOfDay
    return $weekday -and $t -ge [TimeSpan]'08:45' -and $t -le [TimeSpan]'15:50'
}

Log 'started; waiting for the running download to finish'
while (DownloaderRunning) { Start-Sleep -Seconds 120 }
Log 'download finished'

foreach ($pass in 1, 2) {
    # the downloader logs in at start-up; never do that while the live recorder is in session
    while (MarketHours) { Start-Sleep -Seconds 300 }
    Log "retry pass $pass"
    & $py angel_history.py *>> (Join-Path $proj 'logs\angel_history_retry.log')
    $last = Get-Content (Join-Path $proj 'logs\angel_history.log') -Tail 1
    Log "retry pass $pass done: $last"
}

Log 'M1 development run (intraday_research.py, no test reveal)'
$m1 = Start-Process -FilePath $py -ArgumentList 'intraday_research.py' -WorkingDirectory $proj `
    -RedirectStandardOutput (Join-Path $proj 'logs\m1_dev.log') `
    -RedirectStandardError (Join-Path $proj 'logs\m1_dev.err.log') `
    -WindowStyle Hidden -PassThru
try { $m1.PriorityClass = 'BelowNormal' } catch {}
$m1.WaitForExit()
Log "M1 exit code $($m1.ExitCode)"
if ($m1.ExitCode -eq 0 -and (Test-Path (Join-Path $proj 'reports\m1_intraday.md'))) {
    & $git add reports/m1_intraday.md reports/m1_intraday.json
    & $git commit -q -m 'M1 development result, decided by the pre-registered rule (automatic run, test not revealed)' -- reports/m1_intraday.md reports/m1_intraday.json
    & $git push -q origin main
    Log 'M1 report committed and pushed'
}

while (MarketHours) { Start-Sleep -Seconds 300 }
Log 'Google Drive backup'
& $py backup_to_drive.py *>> (Join-Path $proj 'logs\backup_console.log')
Log 'all steps done'
