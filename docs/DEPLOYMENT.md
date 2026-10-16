# Deployment and evaluation

[Project README](../README.md) · [Documentation index](README.md) ·
[Training guide](../TRAINING.md)

The published agent is
[`deployments/routed_teacher_final/`](../deployments/routed_teacher_final/).
It includes its runtime source, model configs, role coefficients, and three
Git LFS checkpoints. Datasets and training opponents are not required to package
it; evaluation requires an opponent package.

All PowerShell examples below start at the **repository root**.

## Environment setup

Prepare Git LFS, Python, and Node.js. Use pnpm for the committed
[`pnpm-lock.yaml`](../pnpm-lock.yaml).

After following the clone steps in the [README](../README.md#get-started),
create the Python environment:

```powershell
python -m venv .venv
$Python = Join-Path (Get-Location).Path ".venv/Scripts/python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r ./requirements.txt
pnpm install --frozen-lockfile
pnpm run lux:help
```

The recorded [requirements](../requirements.txt) pin
`torch==2.12.1+cu132` and a CUDA wheel index. Use a Python/platform combination
that supports that build. If the pinned wheel is unavailable for your system,
consult the [PyTorch installation guidance](https://pytorch.org/get-started/locally/)
and record any environment changes before comparing results.

The PowerShell replay helper expects `.venv/Scripts/python.exe`. On other
platforms the Python environment normally uses `.venv/bin/python`; the helper
also uses Windows process-management commands, so the paired-evaluation example
below requires Windows.

A dependency import check does not start training or play a match:

```powershell
& $Python -c "import torch, gym, kaggle_environments; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The agent selects CPU when CUDA is unavailable. Its packaged runtime otherwise
requests `cuda:0`; timing and timeout behavior depend on the actual environment.

## Verify downloaded weights

Run `git lfs pull` first. Then compare each downloaded checkpoint with the
[package manifest](../deployments/routed_teacher_final/PACKAGE_MANIFEST.json):

```powershell
$Package = Join-Path (Get-Location).Path "deployments/routed_teacher_final"
$ManifestPath = Join-Path $Package "PACKAGE_MANIFEST.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

foreach ($Route in $Manifest.routes.PSObject.Properties) {
    $Checkpoint = Join-Path $Package $Route.Value.checkpoint
    $ActualHash = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $Route.Value.sha256) {
        throw "Checkpoint hash mismatch for $($Route.Name). Check git lfs pull."
    }
    Write-Host "$($Route.Name): checkpoint verified"
}
```

A small text file beginning with `version https://git-lfs.github.com/spec/v1`
is an LFS pointer, not a downloaded model.

The model router resolves checkpoint filenames inside the package. Some model
YAML files and `BASELINE_LOCK.json` retain historical training-source paths.
Those are experiment/provenance records; they are not directories you need to
recreate for final inference. Runtime route paths are configured in
[`rl_agent_config.yaml`](../deployments/routed_teacher_final/lux_ai/rl_agent/rl_agent_config.yaml).

## Package the published agent

To make an archive with `main.py`, `__init__.py`, and `lux_ai/` at its root:

```powershell
$RepoRoot = (Get-Location).Path
$Package = Join-Path $RepoRoot "deployments/routed_teacher_final"
$OutputDir = Join-Path $RepoRoot "outputs"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Archive = Join-Path $OutputDir "routed_teacher_final.tar.gz"
tar -czf $Archive -C $Package main.py __init__.py lux_ai
```

Use an archive filename of your choice if a local archive already exists.
The repository-root `main.py` is a Lux stdin/stdout entry point, not a match
runner. The final deployment has its own `main.py` and runtime files.

## Paired evaluation

The recorded `first` suite schedules 10 seeds × 4 maps × 2 sides = 80 games.
Supply your own deployed `first` package; it is not bundled with this repository.

Run the following in one PowerShell session. The prompt accepts the opponent
package directory containing `main.py`:

```powershell
$FirstAgent = (Resolve-Path -LiteralPath (Read-Host "Path to the first opponent package")).Path
$EvalOptions = @{
    CurrentAgent = "./deployments/routed_teacher_final"
    FirstAgent = $FirstAgent
    OpponentNames = @("first")
    Seeds = @(20260920..20260929)
    MapSizes = @(12, 16, 24, 32)
    Sides = @(0, 1)
    OutputDir = "./outputs/checkpoint_selection/routed_teacher_final_vs_first_10seeds_local"
    AgentTurnTimeoutMs = 30000
    TimeoutSeconds = 180
    MaxAttempts = 1
    DisableRoleTrace = $true
    ContinueOnFailure = $true
}
& ./scripts/generate_deployed_agent_replays.ps1 @EvalOptions
```

Invoke the helper in the same PowerShell process to preserve typed array
arguments. If local policy blocks the unsigned script, open a session with
`powershell -NoProfile -ExecutionPolicy Bypass`, return to the repository root,
and rerun the example there.

The helper reuses completed replay files. Choose a fresh output directory when
evaluating a different agent or runtime setting.

Inspect `manifest.json`, `attempt_diagnostics.jsonl`, and the `logs/` directory
under your output directory. The summary tool accepts that directory:

```powershell
$Python = Join-Path (Get-Location).Path ".venv/Scripts/python.exe"
& $Python ./scripts/summarize_checkpoint_selection.py --root ./outputs/checkpoint_selection/routed_teacher_final_vs_first_10seeds_local
```

Report completed-game win rate, failures/timeouts, city and unit margins,
BUILD_CITY behavior, worst-night loss, and per-map/per-side results. Define
`Score = final city tiles + final units` when reporting Score. The
[recorded evaluation report](../deployments/routed_teacher_final/EVALUATION.md)
is the source for published results; locally generated reports are separate.

The [development/promotion/holdout seed suites](../conf/evaluation/paired_seed_suites.yaml)
cover 16/24/32 maps and are a separate protocol from the four-map `first` suite
above.

## Role replays

For visualization, omit `DisableRoleTrace` or set it to `$false`, and use a
separate output directory. The replay helper may then emit:

| File | Purpose |
| --- | --- |
| `replays/<match>.commands.json` | Official command replay |
| `replays/<match>.json` | Converted stateful replay |
| `replays/<match>.roles.json` | Role sidecar, when a trace is produced |
| `roles/<match>.attempt<N>.p<P>.jsonl` | Per-player raw role trace |

Open the [bundled viewer](../tools/role_replay_viewer/README.md) for the stateful
replay and sidecar. For the separate
[Role-Colored Lux Viewer](https://github.com/Gassyuku-groupC/Role-Colored-Lux-Viewer),
merge the command replay and sidecar using that viewer repository's documented
merge script, then upload the resulting `*.role-colored.json`.
