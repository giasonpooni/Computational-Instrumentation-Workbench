<#
.SYNOPSIS
Set up and control a persistent, loopback-only local CIW deployment.
.DESCRIPTION
Setup creates a repository-local .venv and installs this package. Without
-PythonExecutable it tries py -3.12, then python (Python 3.11 or later required).
Start runs a hidden service, restoring the saved workspace when present.
Stop verifies ownership and saves the workspace before stopping that process.
No Windows service, scheduled task, registry entry, or global package is created.
.EXAMPLE
.\scripts\workbench.ps1 Setup -PythonExecutable C:\Python312\python.exe
.EXAMPLE
.\scripts\workbench.ps1 Start
.EXAMPLE
.\scripts\workbench.ps1 Status
.EXAMPLE
.\scripts\workbench.ps1 Stop
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('Setup', 'Start', 'Status', 'Stop')]
    [string]$Action,
    [string]$PythonExecutable = '',
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [string]$DataDirectory = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not $DataDirectory) { $DataDirectory = Join-Path $RepoRoot '.ciw' }
$DataDirectory = [IO.Path]::GetFullPath($DataDirectory)
$StatePath = Join-Path $DataDirectory 'service.json'
$WorkspacePath = Join-Path $DataDirectory 'workspace.json'
$StdoutPath = Join-Path $DataDirectory 'service.stdout.log'
$StderrPath = Join-Path $DataDirectory 'service.stderr.log'
$Utf8NoBom = New-Object Text.UTF8Encoding($false)

function Quote-Argument([string]$Value) {
    # Windows paths cannot contain quotes. Doubling final backslashes follows
    # the Windows argv quoting rule for a quoted directory argument.
    if ($Value.Contains('"')) { throw 'Arguments containing double quotes are not supported.' }
    return '"' + ($Value -replace '(\\+)$', '$1$1') + '"'
}

function Save-State($State) {
    $temporary = Join-Path $DataDirectory ('service-' + [guid]::NewGuid().ToString('N') + '.tmp')
    try {
        [IO.File]::WriteAllText($temporary, ($State | ConvertTo-Json -Depth 10), $Utf8NoBom)
        Move-Item -LiteralPath $temporary -Destination $StatePath -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}

function Read-State {
    if (-not (Test-Path -LiteralPath $StatePath)) { return $null }
    $state = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($name in @('controller_version', 'process_id', 'process_started_at', 'executable_path', 'session_id', 'port', 'data_directory', 'stdout_log', 'stderr_log')) {
        if (-not $state.PSObject.Properties[$name]) { throw "Invalid ownership file: missing $name." }
    }
    if ($state.controller_version -ne 1 -or [int]$state.process_id -lt 1 -or [int]$state.port -lt 1 -or [int]$state.port -gt 65535) {
        throw 'Invalid deployment ownership file.'
    }
    if ([IO.Path]::GetFullPath([string]$state.data_directory) -ine $DataDirectory) {
        throw 'Ownership file belongs to a different data directory.'
    }
    return $state
}

function Get-OwnedProcess($State) {
    if ($State.PSObject.Properties['stopped_at']) { return $null }
    $process = Get-Process -Id ([int]$State.process_id) -ErrorAction SilentlyContinue
    if (-not $process) { return $null }
    # PowerShell 7 may deserialize ISO JSON timestamps directly into DateTime;
    # converting that value to string loses fractional seconds and uses locale.
    # Compare UTC ticks, accepting both that representation and PS 5.1 strings.
    try {
        if ($State.process_started_at -is [DateTime]) {
            $recordedStart = $State.process_started_at.ToUniversalTime()
        } elseif ($State.process_started_at -is [DateTimeOffset]) {
            $recordedStart = $State.process_started_at.UtcDateTime
        } else {
            $recordedStart = [DateTime]::ParseExact(
                [string]$State.process_started_at, 'o', [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::RoundtripKind
            ).ToUniversalTime()
        }
    } catch {
        throw 'Ownership file contains an invalid process start timestamp. No process will be stopped.'
    }
    if ($process.StartTime.ToUniversalTime().Ticks -ne $recordedStart.Ticks -or
        [IO.Path]::GetFullPath($process.Path) -ine [IO.Path]::GetFullPath([string]$State.executable_path)) {
        throw "PID $($State.process_id) no longer matches the stored start time and executable. No process will be stopped."
    }
    return $process
}

function Invoke-Ciw([string]$Kind, [int]$ServicePort) {
    if (-not (Test-Path -LiteralPath $VenvPython)) { throw 'Run Setup before using the local service.' }
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = $VenvPython
    $startInfo.Arguments = '-m ciw send ' + (Quote-Argument $Kind) + ' --url ' + (Quote-Argument "ws://127.0.0.1:$ServicePort")
    $startInfo.WorkingDirectory = $RepoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $probe = New-Object Diagnostics.Process
    $probe.StartInfo = $startInfo
    try {
        [void]$probe.Start()
        $output = $probe.StandardOutput.ReadToEndAsync()
        $errors = $probe.StandardError.ReadToEndAsync()
        if (-not $probe.WaitForExit(25000)) {
            $probe.Kill()
            [void]$probe.WaitForExit(5000)
            throw "Timed out requesting $Kind; the service was not stopped."
        }
        if ($probe.ExitCode -ne 0) { throw "Request $Kind failed: $($errors.Result.Trim()) $($output.Result.Trim())" }
        $response = $output.Result | ConvertFrom-Json
        if ($response.protocol_version -ne 1 -or $response.type -ne 'response') { throw "Request $Kind returned an invalid response." }
        return $response.payload
    } finally {
        $probe.Dispose()
    }
}

function Get-ListenerOwners([int]$ServicePort) {
    try {
        $connections = @(Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $ServicePort -State Listen -ErrorAction Stop)
    } catch {
        if ($_.CategoryInfo.Category -eq [Management.Automation.ErrorCategory]::ObjectNotFound) { return }
        throw "Cannot inspect listener ownership: $($_.Exception.Message) Run this controller in a shell permitted to inspect local TCP connections. No process will be stopped."
    }
    $connections | Select-Object -ExpandProperty OwningProcess -Unique
}

function Get-OwnedSnapshot($State) {
    if (-not (Get-OwnedProcess $State)) { throw 'The owned service is not running.' }
    if (-not $State.session_id) { throw 'Ownership is incomplete; inspect the deployment logs before taking further action.' }
    $owners = @(Get-ListenerOwners ([int]$State.port))
    if ($owners.Count -ne 1 -or [int]$owners[0] -ne [int]$State.process_id) {
        throw 'The owned process does not own the configured listener. No process will be stopped.'
    }
    $snapshot = Invoke-Ciw 'session.get' ([int]$State.port)
    if ($snapshot.session_id -cne $State.session_id) {
        throw 'The endpoint session does not match the owned service. No process will be stopped.'
    }
    return $snapshot
}

function Write-Status($State) {
    if (-not $State) {
        [ordered]@{ status = 'not_started'; data_directory = $DataDirectory; workspace = $WorkspacePath } | ConvertTo-Json
        return
    }
    $status = [ordered]@{
        status = 'stopped'; process_id = $State.process_id; session_id = $State.session_id
        endpoint = "ws://127.0.0.1:$($State.port)"; data_directory = $DataDirectory
        workspace = $WorkspacePath; workspace_exists = (Test-Path -LiteralPath $WorkspacePath)
        stdout_log = $State.stdout_log; stderr_log = $State.stderr_log
    }
    if (Get-OwnedProcess $State) {
        $snapshot = Get-OwnedSnapshot $State
        $status.status = 'running'
        $status.run_id = $snapshot.run.run_id
        $status.evidence_id = $snapshot.run.evidence_id
        $status.selection_revision = $snapshot.selection.revision
        $status.result_count = @($snapshot.results).Count
    }
    $status | ConvertTo-Json -Depth 10
}

function Assert-PortAvailable([int]$ServicePort) {
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $ServicePort)
    $listener.Server.ExclusiveAddressUse = $true
    try {
        $listener.Start()
    } catch {
        throw "Loopback port $ServicePort is already occupied or unavailable. No existing service was contacted or stopped."
    } finally {
        $listener.Stop()
    }
}

function New-Ownership($Process, [int]$ServicePort, [string]$SessionId) {
    return [ordered]@{
        controller_version = 1; process_id = $Process.Id
        process_started_at = $Process.StartTime.ToUniversalTime().ToString('o')
        executable_path = $Process.Path; session_id = $SessionId; port = $ServicePort
        data_directory = $DataDirectory; stdout_log = $StdoutPath; stderr_log = $StderrPath
    }
}

function Resolve-ServiceProcess($Launched, [int]$ServicePort) {
    $owners = @(Get-ListenerOwners $ServicePort)
    if ($owners.Count -ne 1) { throw 'Could not identify the temporary launcher''s service listener.' }
    $serviceProcess = Get-Process -Id ([int]$owners[0])
    $ancestor = $serviceProcess.Id
    for ($depth = 0; $depth -lt 8; $depth++) {
        if ($ancestor -eq $Launched.Id) {
            if ($Launched.HasExited -or $serviceProcess.StartTime -lt $Launched.StartTime) { break }
            return $serviceProcess
        }
        $record = Get-CimInstance Win32_Process -Filter "ProcessId=$ancestor"
        if (-not $record -or $record.ParentProcessId -le 0 -or $record.ParentProcessId -eq $ancestor) { break }
        $ancestor = [int]$record.ParentProcessId
    }
    throw 'Listener is not the process launched by this controller or its child. No process will be stopped.'
}

try {
    if ($env:OS -ne 'Windows_NT') { throw 'This local deployment controller requires Windows PowerShell. Use the Python CLI on other platforms.' }
    if ($Action -eq 'Setup') {
        $existing = Read-State
        if ($existing -and (Get-OwnedProcess $existing)) { throw 'Stop the owned service before updating its environment.' }
        if (-not (Test-Path -LiteralPath $VenvPython)) {
            $pythonArgs = @()
            if ($PythonExecutable) {
                $selectedPython = (Get-Command $PythonExecutable -ErrorAction Stop).Source
            } else {
                $selectedPython = $null
                $launcher = Get-Command py -ErrorAction SilentlyContinue
                if ($launcher) {
                    try {
                        & $launcher.Source -3.12 --version 2>$null | Out-Null
                        if ($LASTEXITCODE -eq 0) { $selectedPython = $launcher.Source; $pythonArgs = @('-3.12') }
                    } catch { $selectedPython = $null }
                }
                if (-not $selectedPython) { $selectedPython = (Get-Command python -ErrorAction Stop).Source }
            }
            & $selectedPython @pythonArgs -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'
            if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 or later is required.' }
            & $selectedPython @pythonArgs -m venv (Join-Path $RepoRoot '.venv')
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated environment.' }
        }
        & $VenvPython -m pip install --no-cache-dir --disable-pip-version-check --editable $RepoRoot
        if ($LASTEXITCODE -ne 0) { throw 'Package installation failed.' }
        [ordered]@{ status = 'ready'; python = $VenvPython; data_directory = $DataDirectory } | ConvertTo-Json
        exit 0
    }

    $state = Read-State
    if ($Action -eq 'Status') { Write-Status $state; exit 0 }

    if ($Action -eq 'Start') {
        if (-not (Test-Path -LiteralPath $VenvPython)) { throw 'Run Setup before Start.' }
        if ($state -and (Get-OwnedProcess $state)) {
            if ($PSBoundParameters.ContainsKey('Port') -and $Port -ne [int]$state.port) { throw 'The owned service is already running on another port.' }
            Write-Status $state
            exit 0
        }
        if ($state -and -not $PSBoundParameters.ContainsKey('Port')) { $Port = [int]$state.port }
        Assert-PortAvailable $Port
        New-Item -ItemType Directory -Path $DataDirectory -Force | Out-Null
        $arguments = @('-m', 'ciw', 'serve', '--port', [string]$Port, '--output-dir', (Quote-Argument $DataDirectory))
        if (Test-Path -LiteralPath $WorkspacePath) { $arguments += @('--workspace', (Quote-Argument $WorkspacePath)) }
        $launched = Start-Process -FilePath $VenvPython -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath -PassThru
        $state = New-Ownership $launched $Port ''
        Save-State $state
        $deadline = [DateTime]::UtcNow.AddSeconds(20)
        $sessionId = ''
        while ([DateTime]::UtcNow -lt $deadline) {
            $launched.Refresh()
            if ($launched.HasExited) { throw "Service startup failed. Inspect $StderrPath." }
            $log = Get-Content -LiteralPath $StdoutPath -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
            if ($log -and $log.Contains("Computational Instrumentation Workbench: ws://127.0.0.1:$Port") -and $log -match '(?m)^Session\s+(session-[a-zA-Z0-9]+)\s+\|') {
                $sessionId = $Matches[1]
                break
            }
            Start-Sleep -Milliseconds 200
        }
        if (-not $sessionId) { throw "Service did not become ready. Inspect $StdoutPath and $StderrPath; no process was force-stopped." }
        $service = Resolve-ServiceProcess $launched $Port
        $state = New-Ownership $service $Port $sessionId
        Save-State $state
        [void](Get-OwnedSnapshot $state)
        Write-Status $state
        exit 0
    }

    if ($Action -eq 'Stop') {
        if (-not $state -or -not (Get-OwnedProcess $state)) { Write-Status $state; exit 0 }
        $snapshot = Get-OwnedSnapshot $state
        $saved = Invoke-Ciw 'workspace.save' ([int]$state.port)
        if ([IO.Path]::GetFullPath([string]$saved.workspace_file) -ine $WorkspacePath -or -not (Test-Path -LiteralPath $WorkspacePath)) {
            throw 'Workspace save did not confirm the expected file. The service remains running.'
        }
        $workspace = Get-Content -LiteralPath $WorkspacePath -Raw -Encoding UTF8 | ConvertFrom-Json
        $current = Get-OwnedSnapshot $state
        $savedIds = @($workspace.results | ForEach-Object { $_.result_id } | Sort-Object)
        $currentIds = @($current.results | ForEach-Object { $_.result_id } | Sort-Object)
        if ($workspace.workspace_version -ne 1 -or $workspace.run.run_id -cne $current.run.run_id -or
            $workspace.run.evidence_id -cne $current.run.evidence_id -or $workspace.selection.revision -ne $current.selection.revision -or
            [string]::Join(',', $savedIds) -cne [string]::Join(',', $currentIds)) {
            throw 'The saved workspace does not match the current session; it may still be changing. The service remains running. Retry Stop after changes finish.'
        }
        $owned = Get-OwnedProcess $state
        if (-not $owned) { throw 'The service exited before the verified stop.' }
        Stop-Process -InputObject $owned -Confirm:$false
        if (-not $owned.WaitForExit(5000)) { throw 'The saved service has not yet exited. Inspect Status before further action.' }
        $state | Add-Member -NotePropertyName stopped_at -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
        Save-State $state
        Write-Status $state
        exit 0
    }
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
