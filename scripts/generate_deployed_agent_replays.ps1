[CmdletBinding()]
param(
    [string]$CurrentAgent = "outputs\current_agent",
    [string]$BestAgent = "outputs\submission_packages\best_agent",
    [string]$FirstAgent = "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research",
    [string]$Stage400Agent = "outputs\auto_league_dagger_v7_16x16\best_agent",
    [string]$Stage350Agent = "outputs\auto_league_dagger_v4_16x16\learner_agent",
    [string]$NodeExe = "",
    [int[]]$Seeds = @(20260821),
    [int[]]$MapSizes = @(12, 24),
    [string[]]$OpponentNames = @("best_agent", "first", "stage350", "stage400"),
    [int[]]$Sides = @(0, 1),
    [string]$OutputDir = "outputs\spatial_risk_deployed_replays",
    [int]$AgentTurnTimeoutMs = 30000,
    [int]$TimeoutSeconds = 240,
    [int]$HeartbeatSeconds = 60,
    [int]$MaxAttempts = 1,
    [int]$RetryDelaySeconds = 5,
    [switch]$DisableEnginePipeFix,
    [switch]$DisableRoleTrace,
    [switch]$ContinueOnFailure
)

$ErrorActionPreference = "Stop"
$LuxRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $LuxRoot ".venv\Scripts\python.exe"
$PythonDir = Split-Path -Parent $Python
if (-not $NodeExe) {
    $bundledNode = Join-Path $LuxRoot ".tools\node16\node_modules\node\bin\node.exe"
    if (Test-Path -LiteralPath $bundledNode) {
        $NodeExe = $bundledNode
    } else {
        $nodeCommand = Get-Command node.exe -ErrorAction Stop
        $NodeExe = $nodeCommand.Source
    }
}
$Node = (Resolve-Path -LiteralPath $NodeExe).Path
$NodeRoot = Split-Path -Parent $Node
$LuxCli = Join-Path $LuxRoot "node_modules\@lux-ai\2021-challenge\lib\es5\bin\index.js"
$Converter = Join-Path $LuxRoot "scripts\convert_replay_stateful.js"
$EnginePatch = Join-Path $LuxRoot "scripts\lux_sequential_state_send_patch.js"

function Resolve-ProjectPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return $Path }
    return Join-Path $LuxRoot $Path
}

$CurrentAgent = Resolve-ProjectPath $CurrentAgent
$BestAgent = Resolve-ProjectPath $BestAgent
$FirstAgent = Resolve-ProjectPath $FirstAgent
$Stage400Agent = Resolve-ProjectPath $Stage400Agent
$Stage350Agent = Resolve-ProjectPath $Stage350Agent

if (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $PSScriptRoot) $OutputDir
}
$ReplayDir = Join-Path $OutputDir "replays"
$LogDir = Join-Path $OutputDir "logs"
$RoleDir = Join-Path $OutputDir "roles"
New-Item -ItemType Directory -Path $ReplayDir, $LogDir, $RoleDir -Force | Out-Null
$AttemptDiagnostics = Join-Path $OutputDir "attempt_diagnostics.jsonl"
$EngineLogRoot = Join-Path $LuxRoot "errorlogs"

$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "Process")
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")
[Environment]::SetEnvironmentVariable("Path", "$NodeRoot;$PythonDir;$CurrentPath", "Process")
[Environment]::SetEnvironmentVariable("VIRTUAL_ENV", (Join-Path $LuxRoot ".venv"), "Process")
$NodeModules = @(
    (Join-Path $LuxRoot "node_modules\@lux-ai\2021-challenge\node_modules"),
    (Join-Path $LuxRoot "node_modules"),
    (Join-Path $LuxRoot "node_modules\.pnpm\node_modules")
)
[Environment]::SetEnvironmentVariable("NODE_PATH", ($NodeModules -join ";"), "Process")

$Opponents = @(
    [pscustomobject]@{ Name = "best_agent"; Path = $BestAgent },
    [pscustomobject]@{ Name = "first"; Path = $FirstAgent },
    [pscustomobject]@{ Name = "stage400"; Path = $Stage400Agent }
)
if ($Stage350Agent) {
    $Opponents += [pscustomobject]@{ Name = "stage350"; Path = $Stage350Agent }
}
$selectedOpponentNames = [System.Collections.Generic.HashSet[string]]::new(
    [string[]]$OpponentNames,
    [System.StringComparer]::OrdinalIgnoreCase
)
$Opponents = @($Opponents | Where-Object { $selectedOpponentNames.Contains($_.Name) })
if ($Opponents.Count -eq 0) { throw "No opponents selected: $($OpponentNames -join ', ')" }
foreach ($side in $Sides) {
    if ($side -notin @(0, 1)) { throw "Invalid side $side; expected 0 or 1." }
}

$requiredPaths = @($Python, $Node, $LuxCli, $Converter, (Join-Path $CurrentAgent "main.py"))
if (-not $DisableEnginePipeFix) { $requiredPaths += $EnginePatch }
foreach ($required in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required path not found: $required" }
}
foreach ($opponent in $Opponents) {
    if (-not (Test-Path -LiteralPath (Join-Path $opponent.Path "main.py"))) {
        throw "Opponent agent not found: $($opponent.Name) at $($opponent.Path)"
    }
}

function Stop-Tree([int]$Id) {
    try { & taskkill.exe /PID $Id /T /F 2>$null | Out-Null } catch { }
}

function Get-DescendantProcessIds([int]$RootId) {
    try {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
        return @()
    }
    $descendants = [System.Collections.Generic.List[int]]::new()
    $frontier = [System.Collections.Generic.Queue[int]]::new()
    $frontier.Enqueue($RootId)
    while ($frontier.Count -gt 0) {
        $parentId = $frontier.Dequeue()
        foreach ($child in $processes | Where-Object { $_.ParentProcessId -eq $parentId }) {
            $childId = [int]$child.ProcessId
            $descendants.Add($childId)
            $frontier.Enqueue($childId)
        }
    }
    return @($descendants)
}

function Stop-MatchProcesses([int]$NodeId) {
    $descendants = @(Get-DescendantProcessIds $NodeId)
    [array]::Reverse($descendants)
    foreach ($processId in $descendants) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $NodeId -Force -ErrorAction SilentlyContinue
    Stop-Tree $NodeId
}

function Test-JsonReplay([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -eq 0) { return $false }
    try {
        $replay = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return $null -ne $replay -and $null -ne $replay.results
    } catch {
        return $false
    }
}

function Test-AgentTimeout([string]$StdoutPath, [string]$StderrPath) {
    foreach ($path in @($StdoutPath, $StderrPath)) {
        if ((Test-Path -LiteralPath $path) -and
            (Select-String -LiteralPath $path -Pattern "timed out after" -Quiet)) {
            return $true
        }
    }
    return $false
}

function Get-LastAgentTurn([datetime]$Started) {
    $lastTurn = -1
    $latestMatch = Get-ChildItem (Join-Path $LuxRoot "errorlogs") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.CreationTime -ge $Started.AddSeconds(-2) } |
        Sort-Object CreationTime -Descending |
        Select-Object -First 1
    if ($latestMatch) {
        foreach ($log in Get-ChildItem $latestMatch.FullName -Filter "agent_*.log" -ErrorAction SilentlyContinue) {
            $matches = Select-String -LiteralPath $log.FullName -Pattern "Turn:\s*(\d+)" -AllMatches -ErrorAction SilentlyContinue
            foreach ($line in $matches) {
                foreach ($match in $line.Matches) {
                    $lastTurn = [Math]::Max($lastTurn, [int]$match.Groups[1].Value)
                }
            }
        }
    }
    return $lastTurn
}

function Write-AttemptDiagnostic(
    [string]$Name, [int]$Attempt, [datetime]$Started, [string]$Status,
    [int]$ProcessId, [Nullable[int]]$ExitCode, [string]$CommandReplay,
    [string]$StdoutPath, [string]$StderrPath, [string]$ErrorText
) {
    $record = [ordered]@{
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        name = $Name
        attempt = $Attempt
        status = $Status
        elapsed_seconds = [Math]::Round(([DateTime]::UtcNow - $Started).TotalSeconds, 3)
        process_id = $ProcessId
        exit_code = $ExitCode
        command_bytes = if (Test-Path -LiteralPath $CommandReplay) { (Get-Item -LiteralPath $CommandReplay).Length } else { 0 }
        stdout_bytes = if (Test-Path -LiteralPath $StdoutPath) { (Get-Item -LiteralPath $StdoutPath).Length } else { 0 }
        stderr_bytes = if (Test-Path -LiteralPath $StderrPath) { (Get-Item -LiteralPath $StderrPath).Length } else { 0 }
        last_agent_turn = Get-LastAgentTurn $Started
        error = $ErrorText
    }
    Add-Content -LiteralPath $AttemptDiagnostics -Value ($record | ConvertTo-Json -Compress)
}

function Invoke-Match(
    [string]$Player0,
    [string]$Player1,
    [int]$Seed,
    [int]$MapSize,
    [string]$Name,
    [int]$Attempt = 1
) {
    $commandReplay = Join-Path $ReplayDir "$Name.commands.json"
    $statefulReplay = Join-Path $ReplayDir "$Name.json"
    $stdout = Join-Path $LogDir "$Name.attempt$Attempt.stdout.log"
    $stderr = Join-Path $LogDir "$Name.attempt$Attempt.stderr.log"
    $traceBase = Join-Path $RoleDir "$Name.attempt$Attempt"
    $currentSide = if ($Player0 -eq (Join-Path $CurrentAgent "main.py")) { 0 } else { 1 }
    $engineLogsBefore = @{}
    if (Test-Path -LiteralPath $EngineLogRoot) {
        Get-ChildItem -LiteralPath $EngineLogRoot -Directory | ForEach-Object {
            $engineLogsBefore[$_.FullName] = $true
        }
    }
    function Copy-MatchAgentLogs([int]$CurrentSide) {
        if (-not (Test-Path -LiteralPath $EngineLogRoot)) { return }
        $newLogDir = Get-ChildItem -LiteralPath $EngineLogRoot -Directory |
            Where-Object { -not $engineLogsBefore.ContainsKey($_.FullName) } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -eq $newLogDir) { return }
        $currentLog = Join-Path $newLogDir.FullName "agent_$CurrentSide.log"
        $opponentLog = Join-Path $newLogDir.FullName "agent_$((1 - $CurrentSide)).log"
        if (Test-Path -LiteralPath $currentLog) {
            Copy-Item -LiteralPath $currentLog -Destination (Join-Path $LogDir "$Name.attempt$Attempt.agent_current.log") -Force
        }
        if (Test-Path -LiteralPath $opponentLog) {
            Copy-Item -LiteralPath $opponentLog -Destination (Join-Path $LogDir "$Name.attempt$Attempt.agent_opponent.log") -Force
        }
    }
    $arguments = @()
    if (-not $DisableEnginePipeFix) {
        $arguments += @("-r", "`"$EnginePatch`"")
    }
    $arguments += @(
        "`"$LuxCli`"", "`"$Player0`"", "`"$Player1`"", "--python", "`"$Python`"",
        "--seed", $Seed, "--loglevel", "1", "--memory", "8000",
        "--maxtime", $AgentTurnTimeoutMs, "--width", $MapSize, "--height", $MapSize,
        "--storeLogs=true", "--statefulReplay=false", "--out", "`"$commandReplay`""
    )
    if (-not (Test-JsonReplay $commandReplay)) {
        Remove-Item -LiteralPath $commandReplay -Force -ErrorAction SilentlyContinue
        Write-Host "Running $Name attempt=$Attempt/$MaxAttempts"
        if (-not $DisableRoleTrace) {
            Remove-Item -LiteralPath "$traceBase.p0.jsonl", "$traceBase.p1.jsonl" -Force -ErrorAction SilentlyContinue
            [Environment]::SetEnvironmentVariable("LUX_ROLE_TRACE_PATH", $traceBase, "Process")
        }
        $process = Start-Process -FilePath $Node -ArgumentList $arguments -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        [Environment]::SetEnvironmentVariable("LUX_ROLE_TRACE_PATH", $null, "Process")
        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        $started = [DateTime]::UtcNow
        $nextHeartbeat = $started.AddSeconds($HeartbeatSeconds)
        while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Seconds 2
            if (Test-AgentTimeout $stdout $stderr) {
                $agentExitCode = if ($process.HasExited) { [Nullable[int]]$process.ExitCode } else { $null }
                Write-AttemptDiagnostic $Name $Attempt $started "agent_turn_timeout" $process.Id $agentExitCode $commandReplay $stdout $stderr "Lux engine reported an agent turn timeout"
                Stop-MatchProcesses $process.Id
                Copy-MatchAgentLogs $currentSide
                throw "Agent turn timeout was reported by the Lux engine: $Name"
            }
            if (Test-JsonReplay $commandReplay) { break }
            if ($HeartbeatSeconds -gt 0 -and [DateTime]::UtcNow -ge $nextHeartbeat) {
                $elapsed = [int]([DateTime]::UtcNow - $started).TotalSeconds
                $commandBytes = if (Test-Path -LiteralPath $commandReplay) { (Get-Item -LiteralPath $commandReplay).Length } else { 0 }
                $stdoutBytes = if (Test-Path -LiteralPath $stdout) { (Get-Item -LiteralPath $stdout).Length } else { 0 }
                $stderrBytes = if (Test-Path -LiteralPath $stderr) { (Get-Item -LiteralPath $stderr).Length } else { 0 }
                Write-Host "Replay active name=$Name pid=$($process.Id) elapsed=${elapsed}s command_bytes=$commandBytes stdout_bytes=$stdoutBytes stderr_bytes=$stderrBytes"
                $nextHeartbeat = [DateTime]::UtcNow.AddSeconds($HeartbeatSeconds)
            }
        }
        $exitCode = if ($process.HasExited) { [Nullable[int]]$process.ExitCode } else { $null }
        if (Test-JsonReplay $commandReplay) {
            Write-AttemptDiagnostic $Name $Attempt $started "completed" $process.Id $exitCode $commandReplay $stdout $stderr ""
            Stop-MatchProcesses $process.Id
            Copy-MatchAgentLogs $currentSide
        } else {
            $failureStatus = if ($process.HasExited) { "process_exit_no_replay" } else { "outer_timeout" }
            Stop-MatchProcesses $process.Id
            Copy-MatchAgentLogs $currentSide
            Write-AttemptDiagnostic $Name $Attempt $started $failureStatus $process.Id $exitCode $commandReplay $stdout $stderr "Replay did not produce valid JSON"
            throw "Replay did not produce valid JSON within $TimeoutSeconds seconds: $Name"
        }
    } else {
        Write-Host "Reusing completed command replay $Name"
    }
    if (Test-AgentTimeout $stdout $stderr) {
        throw "Agent turn timeout was reported by the Lux engine: $Name"
    }
    if (Test-JsonReplay $statefulReplay) {
        return [pscustomobject]@{
            name = $Name; map_size = $MapSize; seed = $Seed; replay = $statefulReplay;
            command_replay = $commandReplay; bytes = (Get-Item -LiteralPath $statefulReplay).Length
        }
    }
    $convertOut = Join-Path $LogDir "$Name.convert.stdout.log"
    $convertErr = Join-Path $LogDir "$Name.convert.stderr.log"
    $convert = Start-Process -FilePath $Node -ArgumentList @(
        "`"$Converter`"", "`"$commandReplay`"", "`"$statefulReplay`""
    ) `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $convertOut -RedirectStandardError $convertErr
    if (-not $convert.WaitForExit(300000)) {
        Stop-Tree $convert.Id
        throw "Stateful conversion timed out: $Name"
    }
    $conversionDeadline = [DateTime]::UtcNow.AddSeconds(5)
    while (-not (Test-JsonReplay $statefulReplay) -and [DateTime]::UtcNow -lt $conversionDeadline) {
        Start-Sleep -Milliseconds 200
    }
    if (-not (Test-JsonReplay $statefulReplay)) {
        throw "Stateful conversion did not produce valid JSON: $Name"
    }
    return [pscustomobject]@{
        name = $Name; map_size = $MapSize; seed = $Seed; replay = $statefulReplay;
        command_replay = $commandReplay; bytes = (Get-Item -LiteralPath $statefulReplay).Length
    }
}

$completed = @()
$failures = @()
foreach ($mapSize in $MapSizes) {
    foreach ($seed in $Seeds) {
        foreach ($opponent in $Opponents) {
            foreach ($side in $Sides) {
                $name = "map_${mapSize}x${mapSize}_vs_$($opponent.Name)_${seed}_p${side}"
                $p0 = if ($side -eq 0) { Join-Path $CurrentAgent "main.py" } else { Join-Path $opponent.Path "main.py" }
                $p1 = if ($side -eq 0) { Join-Path $opponent.Path "main.py" } else { Join-Path $CurrentAgent "main.py" }
                try {
                    $result = $null
                    $attemptErrors = @()
                    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
                        try {
                            $result = Invoke-Match $p0 $p1 $seed $mapSize $name $attempt
                            break
                        } catch {
                            $attemptErrors += $_.Exception.Message
                            if ($attempt -lt $MaxAttempts) {
                                Write-Warning "Attempt $attempt/$MaxAttempts failed for $name; retrying after $RetryDelaySeconds second(s)."
                                Start-Sleep -Seconds $RetryDelaySeconds
                            }
                        }
                    }
                    if ($null -eq $result) {
                        throw ($attemptErrors -join " | ")
                    }
                    if (-not $DisableRoleTrace) {
                        $trace = Join-Path $RoleDir "$name.attempt$attempt.p$side.jsonl"
                        $roleSidecar = Join-Path $ReplayDir "$name.roles.json"
                        if (Test-Path -LiteralPath $trace) {
                            & $Python (Join-Path $PSScriptRoot "finalize_role_trace.py") `
                                --trace $trace --replay $result.replay --output $roleSidecar
                            if ($LASTEXITCODE -ne 0) { throw "Role trace finalization failed: $name" }
                            $result | Add-Member -NotePropertyName role_sidecar -NotePropertyValue $roleSidecar
                        } else {
                            Write-Warning "Role trace was not produced: $name"
                        }
                    }
                    $completed += $result
                } catch {
                    $failures += [pscustomobject]@{ name = $name; error = $_.Exception.Message }
                    Write-Warning $_.Exception.Message
                    if (-not $ContinueOnFailure) { throw }
                }
            }
        }
    }
}

$manifest = [pscustomobject]@{
    current_agent = $CurrentAgent
    opponents = $Opponents
    seeds = $Seeds
    map_sizes = $MapSizes
    sides = $Sides
    completed = $completed
    failures = $failures
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputDir "manifest.json") -Encoding utf8
Write-Host "Completed $($completed.Count) replay(s); failures=$($failures.Count); output=$OutputDir"
