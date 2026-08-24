# Lux AI 2021 Role-Conditioned Agent

This repository extends the Lux AI 2021 first-place training stack while preserving its legacy 24-block ResNet Actor. The current promoted local baseline is `role_05376 + Rot180`. Its central addition is a plug-and-play Role-City policy adapter; spatial risk Sidecar and intervention Gate research remain implemented but are disabled in the promoted inference path because their evaluated checkpoints did not improve competitive results.

The repository contains source code and configuration. Model weights, replays, datasets, experiment outputs, and local opponent packages are excluded from Git.

## Current Promoted Architecture

```text
Game observation
  -> legacy 24-block ResNet Actor
  -> Rot180 test-time policy ensemble
  -> worker and city action logits
  -> RoleCityAdapter soft positive guidance
  -> legal-action mask and collision resolution
  -> Lux commands
```

Role-City is the core project contribution:

- Unit roles: Harvester, Builder, Firefighter, and lowest-priority Attacker.
- City roles: FuelDepot, FuelStation, ResearchStation, ManufacturingPoint, and strictly bounded SacrificialDecay.
- Five-turn role cooldown with a critical Firefighter override.
- Additive legal-action logit bias; no hard action forcing.
- Adjacent allied-unit relay semantics for transfer; workers never transfer directly to cities.
- Compact NumPy state, vectorized distances, grid-neighbor lookup, and bounded runtime fallback.
- `preserve_build_city_logit` prevents Attacker/Firefighter rules from suppressing the Actor's expansion decision.

## Research Extensions

The codebase also retains two independently switchable research components:

- `SpatialRiskAttentionSidecar`: detached Actor features, pooled-KV MHA, and tile-level risk/safe-expansion maps.
- `InterventionGate`: zero-initialized additive logit delta with legal masking and calibrated map/phase rules.

These modules establish checkpoint-compatible risk diagnosis and Step-0-equivalent intervention. They are not credited for the current `role_05376` performance and are OFF in the promoted package. Failed Sidecar/Gate and Stage4 checkpoints are not repository assets.

## Training Direction

```text
deduplicated best/C and external replays
  -> replay/seed-grouped splits
  -> critical-state catalog
  -> same-state Margin/DPO or critical focal BC
  -> Role/Local Adapter training
  -> policy-head and final-block unfreezing only after explicit gates
  -> state-adaptive KL-APPO and executable-opponent PFSP
  -> paired Development, Promotion, and Holdout evaluation
```

Full-frame behavior cloning and Stage4 role-local curriculum are retired because they reduced BUILD_CITY frequency and competitive strength. Current training protects expansion and promotes checkpoints only through matched replay evidence.

## Reproducibility

- [METHODOLOGY.md](METHODOLOGY.md): architecture, evidence status, and learning method.
- [TRAINING.md](TRAINING.md): local artifacts, baseline locking, training and evaluation commands.
- [ROLE.md](ROLE.md): Role-City semantics and runtime constraints.
- [conf/evaluation/paired_seed_suites.yaml](conf/evaluation/paired_seed_suites.yaml): preregistered non-overlapping paired seeds.

## Attribution

Based on [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021). Original attribution and license terms remain applicable.
