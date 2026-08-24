# Role-City Adapter

## Status

Role-City is the core innovation enabled in the current `role_05376 + Rot180` baseline. It is a plug-and-play policy adapter outside the legacy Actor checkpoint. Spatial Sidecar and Risk Gate are separate optional research modules and are OFF in the promoted runtime.

## Data Flow

```text
Actor logits
  -> Rot180 aggregation
  -> RoleCityAdapter.update(game state)
  -> additive role/city soft bias
  -> legal-action mask
  -> action selection
```

`update(...)` refreshes assignments and cooldown state once per turn. `apply(...)` changes legal logits only. Disabling the adapter restores the Actor path without changing the observation space or Actor state dictionary.

## Worker Roles

- `Harvester`: positive guidance toward mining and resource access.
- `Builder`: positive guidance for expansion opportunities.
- `Firefighter`: movement and adjacent-unit relay guidance toward a critical city.
- `Attacker`: lowest-priority positional pressure after economic and survival roles are assigned.

Role changes normally wait five turns. Firefighter may override cooldown only for a critical city that is not `SacrificialDecay`.

## City Roles

- `FuelDepot`: combined transport hub using normalized fuel access and city centrality.
- `FuelStation`: fuel-oriented support city.
- `ResearchStation`: research priority.
- `ManufacturingPoint`: worker production priority.
- `SacrificialDecay`: strictly bounded abandonment candidate.

SacrificialDecay is limited to at most one city and requires a single city tile, zero fuel turns, repeated confirmation, no timely rescue, and sufficient distance from fuel resources. It does not hard-delete legal actions or force the Actor to abandon a city.

## Lux Constraints

Lux workers transfer only to adjacent allied units. Firefighter transfer bias therefore targets an adjacent relay that is at least as close to the critical city. The adapter never assumes direct worker-to-city transfer.

BUILD_CITY is a protected Actor capability. `preserve_build_city_logit: true` prevents fixed Attacker and Firefighter BUILD_CITY penalties. Future critical-state learning must not introduce a global BUILD_CITY suppression rule.

## Runtime Engineering

- compact NumPy role codes and cooldown-expiry arrays;
- broadcast Manhattan distance and direction calculation;
- board-grid neighbor lookup instead of all-pairs transfer search;
- bounded per-map worker budgets and role scales;
- previous-assignment reuse when the update budget is exceeded;
- role trace fields for update time, degradation, cooldown, role change, and reason.

Attacker is processed last. Completed research short-circuits research-oriented city work. Runtime failures are evaluated separately from normal per-turn model inference.

## Learning Interface

The 14 role coefficients can be fixed YAML values or `nn.ParameterDict` parameters. `RoleConditionedLocalAdapter` can add a bounded learned local delta while the Actor remains frozen. Learner rollout buffers transport compact role codes; checkpoints include role/local tensors and optimizer state.

The allowed progression is:

1. Role bias plus Local Adapter on critical states.
2. Final policy head if the small adapter cannot alter target decisions.
3. ResNet blocks 23-24 only after explicit evaluation gates.

Global full-frame BC and the Stage4 continuation are retired. Strict DPO requires preferred and rejected legal actions on the same observation. Replay-only B/G examples from different states are weak preference or focal-BC samples.

## Evaluation

Use the preregistered Development, Promotion, and Holdout suites in `conf/evaluation/paired_seed_suites.yaml`. Every seed is run from both positions. Promotion requires preserved 16-map strength, improved or maintained 24/32 performance, no material BUILD_CITY decline, controlled worst-night loss, and no timeout regression.

## Replay Visualization

Replay generation can emit a matching `*.roles.json` sidecar containing per-turn unit/city roles, cooldowns, changes, reasons, and bias status. Viewer code is maintained separately from the training repository; role trace generation remains here because it is part of evaluation and diagnosis.
