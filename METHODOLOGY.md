# Methodology

## Evidence Status

The current promoted local baseline is `role_05376 + Rot180`. It preserves the legacy Actor and applies the Role-City adapter after Actor inference. Rot180 is retained because matched ablation showed a large 16x16 benefit and a 24x24 benefit; normal model inference remained far below the Lux turn limit. Stage4 checkpoints were rejected after lower win rate, lower city margin, and lower BUILD_CITY frequency.

Spatial Sidecar and Risk Gate are implemented research contributions, but remain disabled in the promoted agent. Documentation distinguishes architecture implemented in the repository from architecture enabled in evaluated deployment.

## Role-City Core Contribution

`RoleCityAdapter.update(...)` classifies units and cities once per turn. `apply(...)` adds small role-conditioned deltas after Actor logits and before legal action selection:

```text
final_logits = LegalMask(ActorLogits + RoleDelta)
```

The adapter is outside the legacy Actor checkpoint and can be disabled for exact rollback. Roles use five-turn cooldown state. Firefighter may override cooldown only for a critical city that is not an abandonment target. Transfer guidance only targets adjacent allied units. Attacker is lowest priority.

Expansion is a protected capability. `preserve_build_city_logit` disables fixed Attacker and Firefighter BUILD_CITY penalties. SacrificialDecay is limited to at most one single-tile city with zero fuel turns, sustained confirmation, no timely rescue, and sufficient distance from fuel resources.

## Spatial Risk Sidecar Contribution

`SpatialRiskAttentionSidecar` remains available for diagnostic research. It consumes `actor_features.detach()`, projects to 64 channels, uses full-resolution queries and adaptive 8x8 pooled key/value tokens, and emits tile-level risk and safe-expansion maps. Pooled-KV attention avoids quadratic 32x32 full self-attention while preserving global spatial context.

Replay groups are split by seed/replay, never by frame. Calibration is independent for 12, 16, 24, and 32 maps and reports sample count, PR curve, average precision, and the first threshold satisfying precision at least 0.85.

## Intervention Gate Contribution

The optional Gate produces an additive policy-logit delta. Its final projection is zero initialized, so Step 0 is mathematically equivalent to the base Actor. Safe expansion has whitelist priority, map/phase rules limit activation, and legal masking occurs after delta addition. Current competitive evidence does not justify enabling it in `role_05376`.

## Critical-State Learning

Lux has snowballing divergence: an early city, resource, or positioning action can create a large late-game city margin. Future training therefore avoids global full-frame BC and focuses on pivotal states.

For paired trajectories record:

- `t_action`: first meaningful legal-action divergence;
- `t_economy`: first persistent city/unit/economic divergence;
- `t_city`: first city-margin threshold crossing.

Training windows precede the earliest reliable divergence. Strict DPO pairs require the same observation, two legal actions, and outcome or branch-rollout evidence. B/G replay-only actions from different states are weak preference or focal-BC evidence, not strict DPO pairs.

No global BUILD_CITY penalty is permitted. Initial critical-state DPO excludes BUILD_CITY as a rejected action. Promotion monitors BUILD_CITY count explicitly.

## Layered Optimization

Trainable scope expands only when the previous scope cannot change critical decisions:

1. Role biases and Role-conditioned Local Adapter.
2. Final policy head.
3. ResNet blocks 23-24.

The remaining Actor stays frozen by default. State-adaptive Teacher KL is computed per state before reduction: normal and small-map states retain stronger `best_agent` anchoring; validated critical branching and fuel-emergency states receive weaker KL. Teacher constraints are drift protection, not an oracle assumption.

Online PFSP may use only executable policies: best, first, stage350, stage400, and historical checkpoints. B/G replay-only policies remain offline evidence.

## Reward And Promotion Boundaries

Terminal outcome dominates learning. The absolute cumulative contribution of all shaping rewards is bounded relative to terminal win/loss. City and unit shaping use bounded potential differences; city-loss shaping cannot be farmed by repeatedly building and losing cities.

Every candidate is evaluated on both sides with preregistered seeds. Development is used for screening, Promotion for full comparison, and Holdout only for final confirmation. Timeouts remain failures. Promotion requires matched completed-game evidence, preserved small-map strength, no material BUILD_CITY decline, controlled worst-night loss, and no timeout regression.
