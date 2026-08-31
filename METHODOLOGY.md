# Methodology

## Evidence Status

The final evaluated Agent is `routed_teacher_final`, not a single checkpoint. It routes map sizes to three validated policy checkpoints while preserving one shared runtime implementation and Role configuration.

| Map | Checkpoint | Runtime behavior |
|---|---|---|
| 12x12 | `er100_35072` | Rot180, Role bias bypassed |
| 16x16 | `role_05376_nofs` | Rot180 plus positive Role guidance |
| 24x24 | `log_03584` | Rot180 plus positive Role guidance |
| 32x32 | `role_05376_nofs` | Rot180 plus positive Role guidance |

The deployment hashes and route definitions are recorded in `deployments/routed_teacher_final/PACKAGE_MANIFEST.json` and `SECOND_MATCH_AGENT.json`.

## Base Actor And Routing

All routed policies use the legacy convolutional Actor-Critic interface. `RLAgent` selects the model config and checkpoint from `model_route_by_map_size` before inference. Rot180 aggregates native and 180-degree policy logits. Role guidance is applied after aggregation and before legal action selection.

```text
final_logits = LegalMask(Rot180Ensemble(Actor(state)) + PositiveRoleDelta)
```

The 12x12 route bypasses Role bias because dense early expansion was consistently more reliable without role intervention. Routing changes policy weights, not the observation/action contract, so the package remains a normal Lux submission Agent.

## Role-City Core Contribution

`RoleCityAdapter.update(...)` classifies units and cities once per turn. `apply(...)` adds bounded legal-action deltas. The adapter is external to the original Actor checkpoint and can be disabled without changing the Actor state dictionary.

Roles use five-turn cooldown state. Firefighter may override cooldown only for a critical non-abandonment city. Transfer guidance targets adjacent allied units; resources deposited into city fuel are never assumed recoverable. Attacker is assigned last. SacrificialDecay is limited to one strictly qualified single-tile city.

Only positive guidance is deployed. `preserve_build_city_logit` prevents Attacker or Firefighter logic from lowering BUILD_CITY. The removed FuelStation concept is absent from runtime code and configuration.

## Training Evolution

The successful route was built conservatively from `role_05376_nofs + Rot180`:

1. Role-only and Role/Local APPO established a stable plug-in policy adapter.
2. Outcome-dominant APPO retained terminal win/loss as the primary objective.
3. Targeted training against `first` improved executable-opponent adaptation.
4. Teacher/KL stages allowed limited Policy Head movement while freezing the backbone.
5. Fixed paired evaluation selected separate 12x12 and 24x24 specialists.
6. The final router combined specialists only after map-level replay evidence.

Training loss never promoted a checkpoint. Promotion used completed-game win rate, city margin, unit margin, BUILD_CITY behavior, worst-night loss, and timeout rate.

## APPO, V-trace, And Teacher Anchoring

The learner supports PPO clipped policy loss with V-trace critic targets, immutable reference-policy KL/BC, separate optimizer groups, mixed precision checks, and Role/Local checkpoint persistence. Fixed opponent pools rotate per completed environment episode; opponent choice does not depend on actor-environment count.

The best Actor is a drift anchor, not an oracle. KL is intentionally weak enough to permit improvement against executable opponents, but candidates that regress paired best-agent evidence are rejected.

## Sidecar And Gate Research

`SpatialRiskAttentionSidecar` uses detached Actor features, adaptive 8x8 pooled key/value attention, and tile-level risk/safe-expansion maps. `InterventionGate` uses a zero-initialized additive logit delta and masks legality after intervention. Replay calibration is grouped by replay/seed and map size.

These modules are implemented and reproducible, but disabled in `routed_teacher_final`. They are not credited for the final competitive result.

## Evaluation Protocol

Every scheduled seed is run from both positions. Score is defined as:

```text
Score = final city tiles + final units
```

Timeouts are reported separately from completed-game win rate. The final 10-seed test against `first` produced 73 valid games, 38 wins, mean Score `150.49`, city margin `+7.86`, unit margin `+8.26`, and seven 32x32 timeouts. The evidence supports a small overall advantage against `first`, while identifying 12x12 strength and 32x32 runtime stability as remaining limitations.
