# Critical-State Preference Training

> Status: research-only offline preference path. It was not used to select the final `routed_teacher_final` weights because the strict same-state DPO set was too small. The final deployment instead uses map-routed checkpoints selected by paired matches.

This historical stage starts from the locked `role_05376 + Rot180` package. The Actor,
Policy Head, Sidecar, and Gate remain frozen. Only `RoleCityAdapter` biases and
the Role-Conditioned Local Adapter are optimized.

## Data classes

- `strict_same_state_dpo.csv`: identical full game state, different preferred
  and rejected actions, with outcome evidence.
- `near_weak_preferences.csv`: similar but non-identical states. These samples
  use 0.25 preference weight and are never reported as strict DPO.
- `critical_focal_bc.csv`: winning actions in the window preceding
  `t_action`, `t_economy`, or `t_star`.

Any preference whose rejected action list contains `BUILD_CITY` is excluded.
External B/G/D winner trajectories are used only as Focal BC on their own
states; they are not converted into artificial same-state DPO pairs.

The current v3 catalog contains 40 paired divergences, 238 near-state weak
preferences, and 1,441 Critical Focal BC samples. It contains zero strict
same-state DPO samples. Strict samples require controlled branch rollouts from
the same serialized state.

## Build shards

Run each split independently so replay-level train/validation/calibration
boundaries remain intact.

```powershell
$Python = ".\.venv\Scripts\python.exe"
$Index = ".\dataset\processed\critical_role_preferences_v3\critical_preferences.csv"
$RoleConfig = ".\outputs\submission_packages\role_05376_rot180_locked\lux_ai\rl_agent\rl_agent_config.yaml"

foreach ($Split in @("train", "validation", "calibration")) {
  & $Python .\scripts\extract_imitation_shards.py `
    --index $Index `
    --agent-dir .\outputs\submission_packages\role_05376_rot180_locked `
    --output-dir ".\dataset\processed\critical_role_shards_v3\$Split" `
    --split $Split `
    --role-assignment-config $RoleConfig
  if ($LASTEXITCODE -ne 0) { throw "Shard extraction failed: $Split" }
}
```

`rl_agent_config.yaml` supplies role classification and map-routing rules.
`role_city_bias_params.yaml` is loaded separately by the training script for
the 14 initial Role bias values.

## Train Role and Local Adapter

```powershell
& ".\.venv\Scripts\python.exe" .\scripts\train_critical_role_preferences.py `
  --train-shards .\dataset\processed\critical_role_shards_v3\train `
  --validation-shards .\dataset\processed\critical_role_shards_v3\validation `
  --agent-dir .\outputs\submission_packages\role_05376_rot180_locked `
  --output-dir .\outputs\critical_role_preferences_v3 `
  --epochs 3 `
  --batch-size 64 `
  --local-lr 1e-5 `
  --role-lr 2e-6 `
  --teacher-kl-cost 0.01
```

Monitor `relative_margin`, `preferred_rank_rate`, `target_flip_rate`, and the
strict/weak/focal effective counts. Do not use global validation loss alone for
promotion.

## Unlock gate

The Policy Head remains frozen unless all of the following are true:

1. Offline target margin or preferred-rank rate improves.
2. At least 32 effective strict same-state pairs exist in validation.
3. Fixed extreme-seed replay evaluation moves `t_action`, `t_economy`, or
   `t_star` later without reducing BUILD_CITY throughput.
4. Paired Development and Promotion evaluation does not regress win rate,
   worst-night city loss, or timeout rate.

The current dataset cannot satisfy condition 2, so
`policy_head_unlock_eligible_offline` must remain false. State-adaptive
KL-APPO is the next stage only after this offline and replay gate passes.
