# Training, packaging, and evaluation

[Project README](README.md) · [Documentation index](docs/README.md) ·
[Deployment and evaluation](docs/DEPLOYMENT.md)

Commands use **Windows PowerShell from this repository root**. Resolve local
paths from your checkout or enter them when prompted; no fixed drive or user
directory is required. Set up the Python environment as described in the
[deployment guide](docs/DEPLOYMENT.md#environment-setup).

## Repository and local artifacts

Git contains source, training configurations, tests, packaging scripts,
documentation, and the final deployment under
[`deployments/routed_teacher_final/`](deployments/routed_teacher_final/).
Checkpoint files use Git LFS.

| Artifact | Available in a fresh clone? | Use |
| --- | --- | --- |
| Published routed agent and its three checkpoints | Yes, after `git lfs pull` | Final inference, packaging, and evaluation |
| `best_agent` teacher package | No | KL/teacher anchoring |
| `first` and other opponent packages | No | Opponent training/evaluation |
| Intermediate Role/Local/Teacher checkpoints | No | Resume a particular training stage |
| Replay corpora and processed datasets | No | Offline diagnosis, BC, and preference research |
| Generated logs, packages, and evaluation outputs | No | Local experiment products |

Historical defaults refer to local trees such as
`outputs/submission_packages/best_agent/`, `internal_testing/hall_of_fame/`,
`dataset/raw/data/`, and `replays/first battle/`. These are artifact locations,
not missing source directories that Git should create.

The [published routes](deployments/routed_teacher_final/PACKAGE_MANIFEST.json)
remain `er100_35072` on 12x12, `role_05376_nofs` on 16x16/32x32, and
`log_03584` on 24x24. See [ROLE.md](ROLE.md) for active inference settings.

## Configure local paths

Files in [`conf/`](conf/) preserve experiment settings, including original
machine-specific paths. Override those values for your local checkout rather
than assuming that a configuration can run unchanged.

| Configuration field | Local value required |
| --- | --- |
| `load_dir`, `checkpoint_file` | Directory and filename of the intended resume checkpoint |
| `teacher_load_dir`, `teacher_checkpoint_file` | Teacher model directory and checkpoint filename |
| `checkpoint_dir`, `hydra.run.dir` | A new local output directory |
| `fixed_opponent_config`, `fixed_opponent_checkpoint` | Model YAML and checkpoint for a targeted opponent |
| `fixed_opponent_configs`, `fixed_opponent_checkpoints` | Corresponding ordered lists for mixed opponents |
| Other pretraining/sidecar paths in the selected config | Existing artifacts for that research stage |

A checkpoint directory is the directory containing the `.pt` file, not
necessarily the top-level deployed agent directory. Use the intended stage's
checkpoint identity; the final routed package is not a substitute for every
historical training input.

### Example: Role/Local outcome APPO

This configuration trains Role bias and the Local Adapter while keeping the
Actor/backbone and policy/value heads frozen. Enter your matching resume and
teacher artifacts. Forward slashes and quoted Hydra values support checkout
paths containing spaces:

```powershell
$RepoRoot = (Get-Location).Path.Replace('\', '/')
$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
$ResumeDir = (Resolve-Path -LiteralPath (Read-Host "Directory containing the resume checkpoint")).Path.Replace('\', '/')
$ResumeFile = Read-Host "Resume checkpoint filename, for example 05376.pt"
$TeacherDir = (Resolve-Path -LiteralPath (Read-Host "Directory containing the teacher checkpoint")).Path.Replace('\', '/')
$TeacherFile = Read-Host "Teacher checkpoint filename, for example candidate_weights.pt"
$OutputDir = "$RepoRoot/outputs/role_local_outcome_appo_local"

$TrainingArgs = @(
    "--config-name", "conv_role_local_outcome_appo"
    "load_dir='$ResumeDir'"
    "checkpoint_file='$ResumeFile'"
    "teacher_load_dir='$TeacherDir'"
    "teacher_checkpoint_file='$TeacherFile'"
    "checkpoint_dir='$OutputDir'"
    "hydra.run.dir='$OutputDir'"
    "disable_wandb=true"
)
& $Python ./run_monobeast.py @TrainingArgs --cfg job --resolve
```

The last command **prints the resolved configuration**. Inspect its paths,
trainable modules, devices, map sampling, and opponent settings before starting.
Use a fresh `$OutputDir` for a different run.

To start that inspected configuration, run:

```powershell
& $Python ./run_monobeast.py @TrainingArgs
```

### Other retained stages

| Config name | Purpose | Additional path setup |
| --- | --- | --- |
| [`conv_role_vs_first_targeted`](conf/conv_role_vs_first_targeted.yaml) | Role/Local adaptation against `first` | Override both singular `fixed_opponent_*` fields |
| [`conv_teacher_joint_vs_first`](conf/conv_teacher_joint_vs_first.yaml) | Policy/Value Head and Role/Local training with frozen backbone | Override both singular `fixed_opponent_*` fields |
| [`conv_teacher_mixed_logscale`](conf/conv_teacher_mixed_logscale.yaml) | Mixed-opponent log-scale outcome continuation | Override both ordered `fixed_opponent_*` lists |

These stages have different resume checkpoints and trainable surfaces. Inspect
their inherited configuration with `--cfg job --resolve`, set every local
artifact path, and retain the stage-specific learning rates and freeze settings.
Changing only `--config-name` in the example is not enough to prepare them.

## Training checks and selection

Monitor finite total/policy/baseline/KL/BC losses, `APPO invalid == 0`, finite
gradients, Teacher BC accuracy, KL divergence, Role/Local/Head parameter deltas,
BUILD_CITY frequency, map/opponent counts, and actor/learner throughput.

`Learning finished` plus a final checkpoint makes the terminal CUDA IPC warning
non-fatal. Training loss never promotes a checkpoint. Use completed paired
matches, city/unit margins, worst-night loss, and timeout rate.

## Rebuild a routed package

The published package can be used directly. Rebuilding instead requires three
locally prepared agent packages. The builder's default inputs are:

```text
outputs/checkpoint_selection/agents/role_05376_nofs/
outputs/checkpoint_selection/agents/er100_35072/
outputs/checkpoint_selection/agents/log_03584/
```

To use your own locations, enter the directories containing each `main.py`:

```powershell
$Python = Join-Path (Get-Location).Path ".venv/Scripts/python.exe"
$BaseAgent = (Resolve-Path -LiteralPath (Read-Host "Path to role_05376_nofs package")).Path
$Map12Agent = (Resolve-Path -LiteralPath (Read-Host "Path to er100_35072 package")).Path
$Map24Agent = (Resolve-Path -LiteralPath (Read-Host "Path to log_03584 package")).Path

& $Python ./scripts/build_second_match_agent.py --base-agent $BaseAgent --map12-agent $Map12Agent --map24-agent $Map24Agent --output ./outputs/submission_packages/routed_teacher_final_local
```

The [builder](scripts/build_second_match_agent.py) assembles route-specific
configs/checkpoints, enables Rot180, disables the runtime scorer gate, bypasses
Role bias on 12x12, checks for active FuelStation code, and writes
`SECOND_MATCH_AGENT.json` with artifact hashes. It refuses an existing output
directory unless `--force` is supplied; that option replaces the directory.

The rebuilt package is a local candidate. Validate it before associating it
with the [recorded final evaluation](deployments/routed_teacher_final/EVALUATION.md).
Follow the [deployment guide](docs/DEPLOYMENT.md) to archive the published
package or run the paired evaluation helper.

## Research and retired directions

The [documentation index](docs/README.md) separates current usage from offline
research and historical experiments.

Retired directions include global full-frame BC after convergence, global
negative BUILD_CITY bias, Stage4 Role curriculum checkpoints, Sidecar/Gate
deployment without paired evidence, and promotion from unequal completed-game
counts or training loss. Their generated checkpoints and replay products remain
local research artifacts.
