# Training and Evaluation

Commands assume PowerShell and `D:\Luxai\Kaggle_Lux_AI_2021`.

## Local Artifacts

Weights, replay corpora, generated agents, and outputs are intentionally not stored in Git. Expected local inputs include:

```text
outputs/submission_packages/best_agent/
outputs/checkpoint_selection/agents/role_05376/
internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/
dataset/raw/data/
replays/first battle/
```

Use `.venv\Scripts\python.exe`; global Python may not contain the pinned Lux dependencies.

## Lock The Promoted Baseline

After checking out the intended code revision:

```powershell
.\.venv\Scripts\python.exe .\scripts\lock_role_baseline.py
```

This copies `role_05376 + Rot180` to `outputs/submission_packages/role_05376_rot180_locked` and writes `BASELINE_LOCK.json` containing the Git revision and SHA256 hashes of weights, model config, runtime config, and Role YAML. Refuse overwrite unless `--force` is explicitly supplied.

## Replay Catalog And Critical States

Build a deduplicated best/C catalog before extracting samples:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_best_c_replay_catalog.py --help
.\.venv\Scripts\python.exe .\scripts\build_weighted_bc_index_from_catalog.py --help
.\.venv\Scripts\python.exe .\scripts\audit_role_assignments.py --help
```

All frames from a replay/seed group stay in one split. B/G winner replays are offline expert evidence, not executable PFSP opponents. Failed windows are excluded from positive global BC and retained for critical-state analysis.

## Current Role Training Surface

The promoted Actor and Rot180 runtime are the baseline. New experiments begin with Role bias and Role-conditioned Local Adapter parameters only. They may progress to the final policy head and ResNet blocks 23-24 only after critical-state margin and action-flip gates pass.

Hard constraints:

- no fixed BUILD_CITY penalty;
- five-turn cooldown;
- Firefighter override only for a critical non-Sacrificial city;
- adjacent allied-unit transfer relay only;
- Attacker has lowest priority;
- at most one strictly qualified SacrificialDecay city;
- Sidecar and Risk Gate disabled unless an experiment explicitly evaluates them.

Stage4 is a retired negative experiment and must not be used as a continuation checkpoint.

## KL-APPO

The learner supports APPO clipped policy loss with V-trace targets, frozen reference-policy KL/BC, distinct optimizer groups, role codes in rollout buffers, and checkpointed Role/Local parameters. State-adaptive KL must be applied per state before reduction. Normal and small-map states retain the best-agent anchor; validated critical states may reduce the KL coefficient.

Monitor:

- finite total, policy, baseline, KL, and BC losses;
- `APPO invalid` and non-finite gradient count;
- Teacher BC accuracy and KL by critical/non-critical state;
- Role parameter and Local Delta changes;
- BUILD_CITY frequency;
- actor/learner throughput and recoverable zero-SPS intervals.

A successful training log does not promote a checkpoint.

## Paired Evaluation Suites

Seeds are preregistered in `conf/evaluation/paired_seed_suites.yaml`:

- Development: 6 seeds per map for frequent screening.
- Promotion: 20 seeds per map after Development passes.
- Holdout: 20 untouched seeds per map for final confirmation.

Maps are 16, 24, and 32; every seed runs both player positions. Do not replace a difficult seed after seeing its result. Timeouts and invalid replay generation remain failures. Use a bounded outer timeout and one attempt.

Example Development run:

```powershell
& .\scripts\run_checkpoint_selection.ps1 `
  -Phase repro `
  -Checkpoints @("role_05376", "candidate") `
  -Seeds @(175307220,391590161,784648117,971699130,296648577,971878553) `
  -MapSizes @(16) `
  -AgentTurnTimeoutMs 30000 `
  -TimeoutSeconds 240 `
  -MaxAttempts 1 `
  -SkipPackaging
```

Run each map with its own registered seed list. Compare only matched conditions and report per map, side, opponent, timeout rate, win rate, city/unit margin, worst-night city loss, BUILD_CITY count, and first persistent divergence turn.

## Sidecar/Gate Research

The repository retains Sidecar BC, calibration, Step-0 equivalence, Gate-only APPO, progressive backbone migration, and PFSP code. These are reproducible research paths, not the current promoted deployment. Their checkpoints and replay products remain local and are never committed.
