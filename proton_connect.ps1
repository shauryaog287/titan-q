# proton_connect.ps1
# ==================
# Presses Proton VPN's Connect or Disconnect button through Windows UI Automation (the Windows
# app has no command line). Used by vpn_switch.py. The app must already be running.
#   proton_connect.ps1 connect      -> CLICKED | ALREADY | DISABLED | FAILED <status>, NOWINDOW, NOBUTTON
#   proton_connect.ps1 disconnect   -> the same
#   proton_connect.ps1 status       -> STATUS <status>  | NOWINDOW
# <status> is the app's protection title ("Unprotected", "Connecting", "Protected" ...).
# Element ids (Proton VPN 5.1.5): ConnectionCardConnectButton ("Connect") when disconnected,
# ConnectionCardDisconnectButton ("Disconnect") when connected, a "Cancel" button while
# connecting, and ProtectionTitle for the status.
param([string]$Action = 'connect')

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
$TS = [System.Windows.Automation.TreeScope]
$T = [System.Windows.Automation.Condition]::TrueCondition
$AE = [System.Windows.Automation.AutomationElement]

function Find-Window {
    foreach ($c in $AE::RootElement.FindAll($TS::Children, $T)) {
        if ($c.Current.Name -eq 'Proton VPN') { return $c }
    }
    return $null
}

function Find-ById($root, $id) {
    $cond = New-Object System.Windows.Automation.PropertyCondition($AE::AutomationIdProperty, $id)
    return $root.FindFirst($TS::Descendants, $cond)
}

$w = $null
for ($i = 0; $i -lt 20 -and -not $w; $i++) { $w = Find-Window; if (-not $w) { Start-Sleep 2 } }
if (-not $w) { 'NOWINDOW'; exit 2 }

function Card-Buttons($root) {
    # the connection card's buttons by name: Connect, Disconnect, Cancel
    $out = @{}
    $btn = New-Object System.Windows.Automation.PropertyCondition($AE::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::Button)
    foreach ($e in $root.FindAll($TS::Descendants, $btn)) {
        if ($e.Current.AutomationId -like 'ConnectionCard*' -and $e.Current.Name -in @('Connect', 'Disconnect', 'Cancel')) {
            $out[$e.Current.Name] = $e
        }
    }
    return $out
}

# an open settings panel (kill switch, NetShield ...) disables the connection card: close it
$close = Find-ById $w 'CloseSettingsButton'
if ($close -and $close.Current.IsEnabled) {
    try { $close.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke(); Start-Sleep 2 } catch {}
}

$buttons = @{}; $title = $null
for ($i = 0; $i -lt 15 -and $buttons.Count -eq 0; $i++) {
    $buttons = Card-Buttons $w
    $title = Find-ById $w 'ProtectionTitle'
    if ($buttons.Count -eq 0) { Start-Sleep 2 }
}
$status = if ($title) { $title.Current.Name } else { '' }
if ($Action -eq 'status') { 'STATUS ' + $status; exit 0 }
if ($buttons.Count -eq 0) { 'NOBUTTON ' + $status; exit 3 }

$target = $null
if ($Action -eq 'disconnect') {
    if ($buttons.ContainsKey('Disconnect')) { $target = $buttons['Disconnect'] }
    elseif ($buttons.ContainsKey('Cancel')) { $target = $buttons['Cancel'] }
} elseif ($buttons.ContainsKey('Connect')) { $target = $buttons['Connect'] }
if (-not $target) { 'ALREADY ' + $status; exit 0 }
if (-not $target.Current.IsEnabled) { 'DISABLED ' + $target.Current.Name + ' ' + $status; exit 4 }
try {
    $target.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
} catch {
    'FAILED ' + $target.Current.Name + ' ' + $status; exit 5
}
'CLICKED ' + $status
exit 0
