<#
.SYNOPSIS
Verify the real Windows local service lifecycle and persistent replay.
.DESCRIPTION
Requires the repository .venv to be installed already. Uses a new directory
under work and an available ephemeral loopback port. It never controls .ciw
or port 8765. Test records and diagnostics remain in work for inspection.
.EXAMPLE
.\scripts\check_local.ps1
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Controller = Join-Path $PSScriptRoot 'workbench.ps1'
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Shell = Join-Path $PSHOME 'pwsh.exe'
if (-not (Test-Path -LiteralPath $Shell)) { $Shell = Join-Path $PSHOME 'powershell.exe' }
$Utf8NoBom = New-Object Text.UTF8Encoding($false)

function Quote-Argument([string]$Value) {
    if ($Value.Contains('"')) { throw 'Arguments containing double quotes are not supported; use a payload file.' }
    return '"' + ($Value -replace '(\\+)$', '$1$1') + '"'
}

function Read-SharedLog([string]$Path) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $reader = New-Object IO.StreamReader($stream)
    try { return $reader.ReadToEnd() }
    finally { $reader.Dispose() }
}

function Invoke-JsonProcess([string]$Executable, [string[]]$Arguments, [string]$Label) {
    # Persistent descendants can retain a controller's pipe handles after it
    # exits. Capture to files and wait for this exact command process, not EOF
    # on inherited pipes or the lifetime of its background service.
    $capture = Join-Path $TestDirectory ('command-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    $stdoutPath = $capture + '.stdout.log'
    $stderrPath = $capture + '.stderr.log'
    $argumentLine = ($Arguments | ForEach-Object { Quote-Argument $_ }) -join ' '
    $child = Start-Process -FilePath $Executable -ArgumentList $argumentLine -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    try {
        if (-not $child.WaitForExit(60000)) {
            $child.Kill()
            [void]$child.WaitForExit(5000)
            throw "$Label timed out after 60 seconds. Only the test command was stopped; service cleanup follows."
        }
        $output = Read-SharedLog $stdoutPath
        $errors = Read-SharedLog $stderrPath
        if ($child.ExitCode -ne 0) {
            throw "$Label exited with code $($child.ExitCode). $($errors.Trim()) $($output.Trim())"
        }
        try { return ($output | ConvertFrom-Json) }
        catch { throw "$Label did not return valid JSON. $($output.Trim()) $($errors.Trim())" }
    } finally {
        $child.Dispose()
    }
}

function Invoke-Controller([string]$Action) {
    Invoke-JsonProcess $Shell @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $Controller,
        $Action, '-DataDirectory', $TestDirectory, '-Port', [string]$ServicePort
    ) "Controller $Action"
}

function Invoke-Request([string]$Kind, $Payload = $null) {
    $arguments = @('-m', 'ciw', 'send', $Kind, '--url', $Endpoint)
    if ($null -ne $Payload) {
        $payloadPath = Join-Path $TestDirectory 'request.json'
        [IO.File]::WriteAllText($payloadPath, ($Payload | ConvertTo-Json -Depth 30), $Utf8NoBom)
        $arguments += @('--payload-file', $payloadPath)
    }
    $response = Invoke-JsonProcess $Python $arguments "Request $Kind"
    if ($response.protocol_version -ne 1 -or $response.type -ne 'response') {
        throw "Request $Kind did not return a protocol-v1 success response."
    }
    return $response.payload
}

function Assert-SameJson($Expected, $Actual, [string]$Label) {
    $expectedJson = $Expected | ConvertTo-Json -Depth 100 -Compress
    $actualJson = $Actual | ConvertTo-Json -Depth 100 -Compress
    if ($expectedJson -cne $actualJson) { throw "$Label changed across Stop/Start." }
}

try {
    if ($env:OS -ne 'Windows_NT') { throw 'This integration check requires Windows.' }
    if (-not (Test-Path -LiteralPath $Python)) { throw 'Install the repository .venv using workbench.ps1 Setup before this check.' }

    $TestDirectory = Join-Path $RepoRoot ('work\localcheck-' + [guid]::NewGuid().ToString('N').Substring(0, 10))
    if (Test-Path -LiteralPath $TestDirectory) { throw 'The generated test directory already exists; rerun to obtain a new directory.' }
    [void][IO.Directory]::CreateDirectory($TestDirectory)
    $probe = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
    $probe.Server.ExclusiveAddressUse = $true
    try {
        $probe.Start()
        $ServicePort = ([Net.IPEndPoint]$probe.LocalEndpoint).Port
    } finally {
        $probe.Stop()
    }
    if ($ServicePort -eq 8765) { throw 'The test refuses to use the interactive service port.' }
    $Endpoint = "ws://127.0.0.1:$ServicePort"
    Write-Host "Checking local deployment on $Endpoint"
    Write-Host "Test data and logs: $TestDirectory"

    $failure = $null
    $cleanupFailure = $null
    try {
        Write-Host 'Starting the isolated service...'
        $started = Invoke-Controller 'Start'
        if ($started.status -ne 'running') { throw 'Start did not report a running service.' }
        $initial = Invoke-Request 'session.get'
        if ($initial.session_id -cne $started.session_id) { throw 'Controller health and protocol session identities disagree.' }

        # A changed selection proves restoration of shared state, not defaults.
        [void](Invoke-Request 'selection.update' @{
            expected_revision = $initial.selection.revision
            channel = 'v'; interval_s = @(0.5, 2.5); cursor_s = 0.75
        })
        $result = Invoke-Request 'analysis.stats'
        if (-not $result.result_id -or -not $result.execution_id -or $result.data.sample_count -lt 1) {
            throw 'Statistics did not produce a nonempty identified result.'
        }
        $before = Invoke-Request 'session.get'
        if (@($before.results).Count -ne 1) { throw 'The fresh test session must contain exactly one result.' }

        Write-Host 'Saving and stopping the isolated service...'
        $stopped = Invoke-Controller 'Stop'
        if ($stopped.status -ne 'stopped' -or -not $stopped.workspace_exists) { throw 'Stop did not confirm a saved, stopped service.' }

        Write-Host 'Restarting and verifying persisted identities, numbers, and selection...'
        $restarted = Invoke-Controller 'Start'
        if ($restarted.status -ne 'running') { throw 'The restored service did not start.' }
        $after = Invoke-Request 'session.get'
        if ($after.session_id -ceq $before.session_id) { throw 'Restart did not create a new service session.' }
        if ($after.session_id -cne $restarted.session_id) { throw 'Restored controller and protocol session identities disagree.' }
        Assert-SameJson $before.selection $after.selection 'Shared selection'
        Assert-SameJson $before.results $after.results 'Result inventory'
        $restored = Invoke-Request 'result.get' @{ result_id = $result.result_id }
        if ($restored.result_id -cne $result.result_id -or $restored.execution_id -cne $result.execution_id) {
            throw 'Restoring the workspace replaced a result or execution identity.'
        }
        Assert-SameJson $result.data $restored.data 'Numerical result'
        Assert-SameJson $result $restored 'Complete scientific result'
    } catch {
        $failure = $_.Exception.Message
    } finally {
        # The controller checks PID, start time, executable, listener and session
        # ownership, then saves before stopping. Never fall back to a blind kill.
        if (Test-Path -LiteralPath (Join-Path $TestDirectory 'service.json')) {
            try {
                $finalState = Invoke-Controller 'Stop'
                if ($finalState.status -ne 'stopped') { throw 'Final cleanup did not confirm a stopped service.' }
            } catch {
                $cleanupFailure = $_.Exception.Message
            }
        }
    }
    if ($failure -or $cleanupFailure) {
        throw (@($failure, $cleanupFailure) | Where-Object { $_ }) -join [Environment]::NewLine
    }
    Write-Host 'PASS: local Start/Stop/Resume preserves result and execution IDs, exact numbers, and selection; test service stopped.'
    exit 0
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
