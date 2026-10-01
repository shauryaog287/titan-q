# xvpn_connect.ps1
# ================
# Presses X-VPN's "Connect" button through Windows UI Automation (X-VPN has no command line).
# Used by vpn_switch.py. Prints one line:
#   CLICKED <status>    the Connect button was pressed (status = X-VPN's "VPN is ..." text)
#   ALREADY <status>    there was no Connect button (already connected or connecting)
#   NOWINDOW            the X-VPN window did not appear within 40 s
# The window title is "X-VPN - Fast & Stable VPN Proxy"; UI Automation drops the "&".

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$TS = [System.Windows.Automation.TreeScope]
$True_ = [System.Windows.Automation.Condition]::TrueCondition

function Find-Window {
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    foreach ($c in $root.FindAll($TS::Children, $True_)) {
        if ($c.Current.Name -like 'X-VPN*') { return $c }
    }
    return $null
}

$w = $null
for ($i = 0; $i -lt 20 -and -not $w; $i++) { $w = Find-Window; if (-not $w) { Start-Sleep 2 } }
if (-not $w) { 'NOWINDOW'; exit 2 }

$button = $null; $status = ''
for ($i = 0; $i -lt 10 -and -not $button; $i++) {
    foreach ($e in $w.FindAll($TS::Descendants, $True_)) {
        $n = $e.Current.Name
        if ($n -like 'VPN is*') { $status = $n }
        if ($n -eq 'Connect' -and $e.Current.ControlType.ProgrammaticName -eq 'ControlType.Button') { $button = $e }
    }
    if (-not $button) { Start-Sleep 2 }
}
if (-not $button) { 'ALREADY ' + $status; exit 0 }
$pattern = $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
$pattern.Invoke()
'CLICKED ' + $status
exit 0
