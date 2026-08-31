[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$CheckpointFiles,
    [int[]]$Seeds = @(20260925, 20260926, 20260927),
    [int[]]$MapSizes = @(12, 16, 24, 32),
    [ValidateSet("best_agent", "first", "stage400")]
    [string[]]$OpponentNames = @("best_agent", "first", "stage400"),
    [string]$OutputDir = "outputs\checkpoint_selection\teacher_global_promotion",
    [int]$AgentTurnTimeoutMs = 30000,
    [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$AgentRoot = Join-Path $Root "outputs\checkpoint_selection\agents"
$OutputRoot = if ([IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir
} else {
    Join-Path $Root $OutputDir
}
$EvalOpponents = Join-Path $OutputRoot "_eval_opponents"

$checkpointArgs = @()
$labels = @()
foreach ($rawFile in $CheckpointFiles) {
    $file = if ([IO.Path]::IsPathRooted($rawFile)) {
        [IO.Path]::GetFullPath($rawFile)
    } else {
        [IO.Path]::GetFullPath((Join-Path $Root $rawFile))
    }
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Checkpoint file is missing: $file"
    }
    $label = "global_" + ([IO.Path]::GetFileNameWithoutExtension($file) -replace "_weights$", "")
    $labels += $label
    $checkpointArgs += @("--checkpoint", "$label=$file")
}

& $Python (Join-Path $PSScriptRoot "prepare_checkpoint_agents.py") `
    --output-dir $AgentRoot `
    --enable-role-adapter `
    @checkpointArgs
if ($LASTEXITCODE -ne 0) { throw "Checkpoint packaging failed." }

& $Python (Join-Path $PSScriptRoot "prepare_checkpoint_eval_runtime.py") `
    --candidate-root $AgentRoot `
    --candidate-names $labels `
    --opponent-root $EvalOpponents `
    --best-agent (Join-Path $Root "outputs\submission_packages\best_agent") `
    --first-agent (Join-Path $Root "internal_testing\hall_of_fame\11-24_12-56-23_062179520_must_research") `
    --stage350-agent (Join-Path $Root "outputs\submission_packages\best_agent") `
    --stage400-agent (Join-Path $Root "outputs\submission_packages\F_2") `
    --opponent-names @OpponentNames
if ($LASTEXITCODE -ne 0) { throw "Evaluation runtime preparation failed." }

foreach ($label in $labels) {
    & (Join-Path $PSScriptRoot "generate_deployed_agent_replays.ps1") `
        -CurrentAgent (Join-Path $AgentRoot $label) `
        -BestAgent (Join-Path $EvalOpponents "best_agent") `
        -FirstAgent (Join-Path $EvalOpponents "first") `
        -Stage400Agent (Join-Path $EvalOpponents "stage400") `
        -OpponentNames $OpponentNames `
        -Seeds $Seeds `
        -MapSizes $MapSizes `
        -Sides @(0, 1) `
        -OutputDir (Join-Path $OutputRoot $label) `
        -AgentTurnTimeoutMs $AgentTurnTimeoutMs `
        -TimeoutSeconds $TimeoutSeconds `
        -MaxAttempts 1 `
        -DisableRoleTrace `
        -ContinueOnFailure
}

& $Python (Join-Path $PSScriptRoot "summarize_checkpoint_selection.py") `
    --root $OutputRoot `
    --output-dir $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Checkpoint summary failed." }

Write-Host "Promotion results: $OutputRoot"
