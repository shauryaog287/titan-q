# install_ollama.ps1 - one-off: wait for the Ollama installer download, check its signature,
# install silently to D:\Ollama with models in D:\OllamaModels, then pull qwen2.5:7b.
# Nothing large goes to C: (the user's C: drive is full). Log: logs\install_ollama.log
param([int]$WaitPid = 0, [switch]$PullOnly)

$ErrorActionPreference = 'Continue'
$log = 'D:\Quant_Project\logs\install_ollama.log'
function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -FilePath $log -Append -Encoding ascii }
$exe = 'D:\Ollama\ollama.exe'
$env:OLLAMA_MODELS = 'D:\OllamaModels'

$setup = 'D:\Installers\OllamaSetup.exe'
if (-not $PullOnly) {
$url = 'https://ollama.com/download/OllamaSetup.exe'
if ($WaitPid -gt 0) {
    Say "waiting for the download (pid $WaitPid)"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
}
# resume until the file is complete and validly signed; a stalled connection (laptop standby,
# VPN drop) is abandoned after 2 minutes below 5 KB/s and resumed
$fresh = $false
for ($i = 1; $i -le 60; $i++) {
    if ((Test-Path $setup) -and (Get-AuthenticodeSignature $setup).Status -eq 'Valid') { break }
    Say ("download attempt {0}, have {1:N0} bytes" -f $i, $(if (Test-Path $setup) { (Get-Item $setup).Length } else { 0 }))
    & curl.exe -s -L -C - --speed-limit 5000 --speed-time 120 --connect-timeout 30 -o $setup $url
    $code = $LASTEXITCODE
    if ($code -eq 0 -and (Get-AuthenticodeSignature $setup).Status -ne 'Valid' -and -not $fresh) {
        Say 'complete but not validly signed (mixed versions?); downloading from scratch once'
        Remove-Item $setup -Force
        $fresh = $true
        continue
    }
    if ($code -ne 0) { Say "curl exit code $code; retrying in 60 s"; Start-Sleep -Seconds 60 }
}
$size = (Get-Item $setup).Length
$sig = Get-AuthenticodeSignature $setup
Say ("installer {0:N0} bytes, signature {1}, signer {2}" -f $size, $sig.Status, $sig.SignerCertificate.Subject)
if ($sig.Status -ne 'Valid') { Say 'signature not valid; stopping'; exit 1 }

[Environment]::SetEnvironmentVariable('OLLAMA_MODELS', 'D:\OllamaModels', 'User')
$env:OLLAMA_MODELS = 'D:\OllamaModels'
New-Item -ItemType Directory -Force 'D:\OllamaModels' | Out-Null
$cFree0 = (Get-PSDrive C).Free
# WaitForExit, not Start-Process -Wait: -Wait also waits for the tray app the installer
# launches, which never exits (that hung this script on 2026-10-01)
$p = Start-Process -FilePath $setup -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/DIR=D:\Ollama' -PassThru
$p.WaitForExit()
Say "installer exit code $($p.ExitCode)"
if (-not (Test-Path $exe)) { Say 'ollama.exe not found in D:\Ollama; stopping'; exit 1 }
Say ("C: free changed by {0:N0} MB during install" -f (((Get-PSDrive C).Free - $cFree0) / 1MB))
}

# make sure a server is running with OLLAMA_MODELS on D:
Start-Sleep -Seconds 10
try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 5 | Out-Null; Say 'server up' }
catch { Say 'starting server'; Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 10 }

for ($i = 1; $i -le 20; $i++) {
    Say "pull attempt $i"
    & $exe pull qwen2.5:7b *> $null      # its progress bars are not ASCII; only the exit code is logged
    if ($LASTEXITCODE -eq 0) { Say 'pull complete'; break }
    Say "pull exit code $LASTEXITCODE"
    Start-Sleep -Seconds 120
}
& $exe list | Out-File -FilePath $log -Append -Encoding ascii
Say ("C: free now {0:N1} GB, D: free {1:N1} GB" -f ((Get-PSDrive C).Free / 1GB), ((Get-PSDrive D).Free / 1GB))
