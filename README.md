# Lux AI 2021 Role-Conditioned Routed Agent

This repository extends the Lux AI 2021 first-place training stack while retaining its 24-block ResNet Actor. The final evaluated deployment is `routed_teacher_final`, a map-routed policy with Rot180 inference and a plug-and-play Role-City adapter.

## Final Deployment

| Map | Policy checkpoint | Rot180 | Role bias |
|---|---|:---:|:---:|
| 12x12 | `er100_35072` | ON | OFF |
| 16x16 | `role_05376_nofs` | ON | ON |
| 24x24 | `log_03584` | ON | ON |
| 32x32 | `role_05376_nofs` | ON | ON |

The self-contained package is stored in [`deployments/routed_teacher_final`](deployments/routed_teacher_final). Its three `.pt` files are tracked by Git LFS. Risk Gate is disabled, the obsolete FuelStation role is removed, and BUILD_CITY remains protected from negative Role bias.

```text
Game observation
  -> map-size checkpoint router
  -> 24-block ResNet Actor
  -> Rot180 policy ensemble
  -> RoleCityAdapter positive soft guidance (16/24/32)
  -> legal-action mask and collision handling
  -> Lux commands
```

## Role-City Contribution

- Worker roles: Harvester, Builder, Firefighter, and lowest-priority Attacker.
- City roles: FuelDepot, ResearchStation, ManufacturingPoint, and strictly bounded SacrificialDecay.
- Five-turn role cooldown with critical Firefighter override.
- Positive additive logit guidance; no hard action forcing.
- Transfer guidance only between adjacent allied units.
- Compact NumPy state, vectorized distance computation, grid-neighbor lookup, and bounded runtime fallback.
- `preserve_build_city_logit` prevents Role logic from suppressing expansion.

## Final Evaluation

Against `first`, seeds `20260920` through `20260929`, all four map sizes and both sides:

- 73 completed games and 7 timeouts, all on 32x32;
- 38 wins, completed-game win rate `52.1%`;
- mean Score `150.49`, where `Score = final city tiles + final units`;
- mean city margin `+7.86` and unit margin `+8.26`.

See [`deployments/routed_teacher_final/EVALUATION.md`](deployments/routed_teacher_final/EVALUATION.md) for the map-level breakdown. Timeouts are always reported separately and are never silently counted as completed losses.

## Training And Reproduction

- [METHODOLOGY.md](METHODOLOGY.md): architecture, training decisions, and evidence boundaries.
- [TRAINING.md](TRAINING.md): required local artifacts, training stages, packaging, and evaluation commands.
- [ROLE.md](ROLE.md): exact Role-City semantics and runtime constraints.
- [solution.md](solution.md): concise final-system overview.
- [conf/evaluation/paired_seed_suites.yaml](conf/evaluation/paired_seed_suites.yaml): preregistered paired evaluation seeds.

Spatial Risk Sidecar and Intervention Gate code remains available for reproducible research, but those modules are disabled in the final deployment because evaluated candidates did not improve competitive results.

## Attribution

Based on [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021). Original attribution and license terms remain applicable.
