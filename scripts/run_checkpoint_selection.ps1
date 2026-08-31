[CmdletBinding()]
param(
    [ValidateSet("phase1", "phase2", "rescue", "distill", "repro", "rolelocal", "stageA", "stage4", "rot180", "outcome", "outcome24", "outcome32", "teacher_eval", "teacher_eval16", "teacher_eval24", "teacher_eval32")]
    [string]$Phase = "phase1",
    [string[]]$Checkpoints = @("bc", "10816", "20128", "30272", "40288", "50112", "60288", "70560"),
    [int[]]$Seeds = @(20260824),
    [int[]]$MapSizes = @(),
    [int]$AgentTurnTimeoutMs = 30000,
    [int]$TimeoutSeconds = 240,
    [int]$MaxAttempts = 1,
    [int]$RetryDelaySeconds = 5,
    [switch]$DisableRoleTrace,
    [switch]$SkipPackaging
)

$ErrorActionPreference = "Stop"
$Checkpoints = @(
    $Checkpoints |
        ForEach-Object { $_ -split ',' } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Agents = Join-Path $Root "outputs\checkpoint_selection\agents"
$PhaseRoot = Join-Path $Root "outputs\checkpoint_selection\$Phase"
$EvalOpponents = Join-Path $PhaseRoot "_eval_opponents"
if ($MapSizes.Count -eq 0) {
    $MapSizes = if ($Phase -eq "phase1") { @(12, 24) } else { @(12, 16, 24, 32) }
}
$Opponents = if ($Phase -in @("phase1", "rescue", "distill", "repro", "rolelocal", "stageA", "stage4", "rot180", "outcome", "outcome24", "outcome32", "teacher_eval", "teacher_eval16", "teacher_eval24", "teacher_eval32")) { @("best_agent") } else { @("best_agent", "first", "stage350", "stage400") }

if (-not $SkipPackaging) {
    & $Python (Join-Path $PSScriptRoot "prepare_checkpoint_agents.py")
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint packaging failed." }
}

& $Python (Join-Path $PSScriptRoot "prepare_checkpoint_eval_runtime.py") `
    --candidate-root $Agents `
    --candidate-names $Checkpoints `
    --opponent-root $EvalOpponents `
    --best-agent (Join-Path $Root "outputs\submission_packages\best_agent") `
    --first-agent (Join-Path $Root "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research") `
    --stage350-agent (Join-Path $Root "outputs\auto_league_dagger_v4_16x16\learner_agent") `
    --stage400-agent (Join-Path $Root "outputs\auto_league_dagger_v7_16x16\best_agent") `
    --opponent-names $Opponents
if ($LASTEXITCODE -ne 0) { throw "Evaluation runtime preparation failed." }

foreach ($checkpoint in $Checkpoints) {
    $agent = Join-Path $Agents $checkpoint
    if (-not (Test-Path -LiteralPath (Join-Path $agent "main.py"))) {
        throw "Checkpoint agent is missing: $agent"
    }
    $output = Join-Path $PhaseRoot $checkpoint
    & (Join-Path $PSScriptRoot "generate_deployed_agent_replays.ps1") `
        -CurrentAgent $agent `
        -BestAgent (Join-Path $EvalOpponents "best_agent") `
        -FirstAgent (Join-Path $EvalOpponents "first") `
        -Stage350Agent (Join-Path $EvalOpponents "stage350") `
        -Stage400Agent (Join-Path $EvalOpponents "stage400") `
        -Seeds $Seeds `
        -MapSizes $MapSizes `
        -OpponentNames $Opponents `
        -Sides 0,1 `
        -OutputDir $output `
        -AgentTurnTimeoutMs $AgentTurnTimeoutMs `
        -TimeoutSeconds $TimeoutSeconds `
        -MaxAttempts $MaxAttempts `
        -RetryDelaySeconds $RetryDelaySeconds `
        -DisableRoleTrace:$DisableRoleTrace `
        -ContinueOnFailure
    $manifestPath = Join-Path $output "manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        Write-Warning "Replay generation did not produce a manifest for $checkpoint"
        continue
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $failureCount = @($manifest.failures).Count
    if ($failureCount -gt 0) {
        Write-Warning "Replay generation recorded $failureCount failure(s) for $checkpoint"
    }
}

& $Python (Join-Path $PSScriptRoot "summarize_checkpoint_selection.py") `
    --root $PhaseRoot `
    --output-dir $PhaseRoot
if ($LASTEXITCODE -ne 0) { throw "Checkpoint summary failed." }
