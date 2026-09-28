# after_download.ps1
# ==================
# Unattended research chain that follows the Angel 1-minute history download. Start it
# detached (the downloader is started separately):
#
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\Quant_Project\after_download.ps1'
#
# Steps, each logged to logs\after_download.log:
#   1. wait for data\intraday_hist_priority_done.json: angel_history.py writes it when the
#      research set (indices, the 50 live stocks, every month's top 100) is complete;
#   2. run intraday_research.py (M1) on DEVELOPMENT data only, at below-normal priority,
#      and commit its report, so the decision is recorded before any test run;
#   3. the same for vpin_research.py (VPIN buy gate);
#   4. wait until the whole download has finished, run one retry pass for failed chunks
#      (never during market hours: the downloader logs in at start-up);
#   5. wait until the market is closed, then back up to Google Drive (backup_to_drive.py).
# Neither research step is ever given --reveal-test. Nothing is deleted.

$ErrorActionPreference = 'Continue'
$proj = 'D:\Quant_Project'
$py = 'D:\trader_env\Scripts\python.exe'
$git = 'D:\Git\cmd\git.exe'
$logFile = Join-Path $proj 'logs\after_download.log'
$marker = Join-Path $proj 'data\intraday_hist_priority_done.json'
Set-Location $proj
$env:PYTHONIOENCODING = 'ascii:replace'
$env:GIT_TERMINAL_PROMPT = '0'

function Log($msg) {
    $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' - ' + $msg
    Add-Content -Path $logFile -Value $line -Encoding ascii
}

function DownloaderRunning {
    $p = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like 'python*.exe' -and $_.CommandLine -like '*angel_history.py*' }
    return [bool]$p
}

function MarketHours {
    $n = Get-Date
    $weekday = $n.DayOfWeek -ne 'Saturday' -and $n.DayOfWeek -ne 'Sunday'
    $t = $n.TimeOfDay
    return $weekday -and $t -ge [TimeSpan]'08:45' -and $t -le [TimeSpan]'15:50'
}

function RunResearch($script, $name, $report) {
    Log "$name development run ($script, no test reveal)"
    $p = Start-Process -FilePath $py -ArgumentList $script -WorkingDirectory $proj `
        -RedirectStandardOutput (Join-Path $proj "logs\$($name)_dev.log") `
        -RedirectStandardError (Join-Path $proj "logs\$($name)_dev.err.log") `
        -WindowStyle Hidden -PassThru
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $p.WaitForExit()
    Log "$name exit code $($p.ExitCode)"
    if ($p.ExitCode -eq 0 -and (Test-Path (Join-Path $proj "reports\$report.md"))) {
        & $git add "reports/$report.md" "reports/$report.json"
        & $git commit -q -m "$name development result, decided by the pre-registered rule (automatic run, test not revealed)" -- "reports/$report.md" "reports/$report.json"
        & $git push -q origin main
        Log "$name report committed and pushed"
    }
}

Log 'started; waiting for the research set of the minute download'
while (-not (Test-Path $marker)) { Start-Sleep -Seconds 120 }
Log ('research set complete: ' + ((Get-Content $marker -Raw) -replace '\s+', ' '))

RunResearch 'intraday_research.py' 'M1' 'm1_intraday'
RunResearch 'vpin_research.py' 'VPIN' 'vpin_gate'

Log 'waiting for the rest of the download to finish'
while (DownloaderRunning) { Start-Sleep -Seconds 300 }
while (MarketHours) { Start-Sleep -Seconds 300 }
Log 'retry pass for failed chunks'
& $py angel_history.py *>> (Join-Path $proj 'logs\angel_history_retry.log')
Log ('retry pass done: ' + (Get-Content (Join-Path $proj 'logs\angel_history.log') -Tail 1))

while (MarketHours) { Start-Sleep -Seconds 300 }
Log 'Google Drive backup'
& $py backup_to_drive.py *>> (Join-Path $proj 'logs\backup_console.log')
Log 'all steps done'
