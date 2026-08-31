# Training, Packaging, And Evaluation

Commands assume PowerShell and `D:\Luxai\Kaggle_Lux_AI_2021`.

## Repository And Local Artifacts

Git contains all source, successful training configurations, tests, packaging scripts, documentation, and the final self-contained deployment under `deployments/routed_teacher_final`. Final `.pt` files use Git LFS.

Large datasets, replay corpora, generated outputs, intermediate checkpoints, logs, and opponent packages remain local and ignored. Expected local training inputs include:

```text
outputs/submission_packages/best_agent/
internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research/
dataset/raw/data/
replays/first battle/
```

Use `.venv\Scripts\python.exe`; global Python may not contain the pinned Lux dependencies.

## Final Route

```text
12x12 -> er100_35072 + Rot180, Role bias OFF
16x16 -> role_05376_nofs + Rot180, Role bias ON
24x24 -> log_03584 + Rot180, Role bias ON
32x32 -> role_05376_nofs + Rot180, Role bias ON
```

The successful configuration chain retained in `conf/` covers Role repair, outcome APPO, targeted `first` training, mixed executable opponents, log-scale outcome training, and edge repair. Sidecar/Gate and Stage4 configurations are research references, not final continuation points.

## Training Commands

Inspect a configuration before starting:

```powershell
.\.venv\Scripts\python.exe .\run_monobeast.py `
  --config-name conv_role_local_outcome_appo --cfg job
```

Run Role/Local outcome APPO:

```powershell
.\.venv\Scripts\python.exe .\run_monobeast.py `
  --config-name conv_role_local_outcome_appo
```

Run targeted executable-opponent adaptation:

```powershell
.\.venv\Scripts\python.exe .\run_monobeast.py `
  --config-name conv_role_vs_first_targeted
```

Run limited Policy Head plus Role/Local Teacher training:

```powershell
.\.venv\Scripts\python.exe .\run_monobeast.py `
  --config-name conv_teacher_joint_vs_first
```

Run mixed-opponent log-scale continuation:

```powershell
.\.venv\Scripts\python.exe .\run_monobeast.py `
  --config-name conv_teacher_mixed_logscale
```

Every continuation path must explicitly set `load_dir` and `checkpoint_file` to an available local checkpoint. Config paths document the original experiment layout and may need local path overrides.

## Required Training Checks

Monitor:

- finite total, policy, baseline, KL, and BC losses;
- `APPO invalid == 0` and finite gradients;
- Teacher BC accuracy and KL divergence;
- Role/Local/Head parameter deltas;
- BUILD_CITY frequency;
- map/opponent sampling counts;
- actor/learner throughput and recoverable zero-SPS periods.

`Learning finished` plus a final checkpoint makes the terminal CUDA IPC warning non-fatal. A successful training log never promotes a checkpoint.

## Build The Final Routed Package

After locally packaging `role_05376_nofs`, `er100_35072`, and `log_03584` under `outputs/checkpoint_selection/agents/`:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_second_match_agent.py --force
```

The builder enforces:

- Rot180 enabled;
- Risk Gate disabled;
- FuelStation absent;
- Role bias disabled on 12x12;
- route-specific model configs and checkpoints;
- SHA256 manifest generation.

The committed deployment can be used directly from:

```text
deployments/routed_teacher_final/
```

## Paired Evaluation

Use both player positions, fixed seeds, one outer attempt, and explicit timeout reporting. Do not replace a difficult seed after observing its result.

Final 10-seed validation against `first` used:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& {
  & '.\scripts\generate_deployed_agent_replays.ps1' `
    -CurrentAgent '.\deployments\routed_teacher_final' `
    -OpponentNames @('first') `
    -Seeds @(20260920,20260921,20260922,20260923,20260924,20260925,20260926,20260927,20260928,20260929) `
    -MapSizes @(12,16,24,32) `
    -Sides @(0,1) `
    -OutputDir '.\outputs\checkpoint_selection\routed_teacher_final_vs_first_10seeds' `
    -AgentTurnTimeoutMs 30000 `
    -TimeoutSeconds 180 `
    -MaxAttempts 1 `
    -DisableRoleTrace `
    -ContinueOnFailure
}"
```

Report completed-game win rate, timeout rate, mean Score, city margin, unit margin, worst-night loss, and per-map/per-side results. Final Score is:

```text
Score = final city tiles + final units
```

The final test completed 73/80 games, won 38/73, and produced mean Score `150.49`, city margin `+7.86`, and unit margin `+8.26`. All seven failures occurred on 32x32.

## Retired Directions

- global full-frame BC after convergence;
- global negative BUILD_CITY bias;
- Stage4 Role curriculum checkpoints;
- Sidecar/Gate enabled deployment without paired evidence;
- promotion from unequal completed-game counts or training loss.

These implementations may remain for research reproducibility, but their generated checkpoints and replay products are not repository assets.
