# after_m1.ps1
# ============
# Second unattended step, started detached next to after_download.ps1. Waits until that
# chain has finished the download retries and the M1 run (it logs "M1 exit code"), then
# runs the VPIN gate test (vpin_research.py) on DEVELOPMENT data only, at below-normal
# priority, and commits its report so the decision is recorded before any test run.
# It never passes --reveal-test. Log: logs\after_download.log.

$ErrorActionPreference = 'Continue'
$proj = 'D:\Quant_Project'
$py = 'D:\trader_env\Scripts\python.exe'
$git = 'D:\Git\cmd\git.exe'
$logFile = Join-Path $proj 'logs\after_download.log'
Set-Location $proj
$env:PYTHONIOENCODING = 'ascii:replace'
$env:GIT_TERMINAL_PROMPT = '0'

function Log($msg) {
    $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' - [vpin] ' + $msg
    Add-Content -Path $logFile -Value $line -Encoding ascii
}

Log 'started; waiting for the M1 step of after_download.ps1'
while (-not (Select-String -Path $logFile -Pattern 'M1 exit code' -Quiet)) { Start-Sleep -Seconds 120 }

Log 'VPIN gate development run (vpin_research.py, no test reveal)'
$p = Start-Process -FilePath $py -ArgumentList 'vpin_research.py' -WorkingDirectory $proj `
    -RedirectStandardOutput (Join-Path $proj 'logs\vpin_dev.log') `
    -RedirectStandardError (Join-Path $proj 'logs\vpin_dev.err.log') `
    -WindowStyle Hidden -PassThru
try { $p.PriorityClass = 'BelowNormal' } catch {}
$p.WaitForExit()
Log "VPIN exit code $($p.ExitCode)"
if ($p.ExitCode -eq 0 -and (Test-Path (Join-Path $proj 'reports\vpin_gate.md'))) {
    & $git add reports/vpin_gate.md reports/vpin_gate.json
    & $git commit -q -m 'VPIN gate development result, decided by the pre-registered rule (automatic run, test not revealed)' -- reports/vpin_gate.md reports/vpin_gate.json
    & $git push -q origin main
    Log 'VPIN report committed and pushed'
}
