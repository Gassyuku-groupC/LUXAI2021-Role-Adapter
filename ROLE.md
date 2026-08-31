# Role-City Adapter

## Deployment Status

Role-City is the central project contribution. It is enabled in the final `routed_teacher_final` package on 16x16, 24x24, and 32x32. The 12x12 route intentionally bypasses Role bias while retaining the same compatible runtime.

Spatial Sidecar and Risk Gate are separate optional research modules and are OFF in final inference.

## Data Flow

```text
Actor logits
  -> Rot180 aggregation
  -> RoleCityAdapter.update(game state)
  -> positive role/city logit delta
  -> legal-action mask
  -> collision-aware action selection
```

`update(...)` refreshes assignments and cooldown state once per turn. `apply(...)` modifies legal logits only. No Role rule directly emits a Lux command.

## Worker Roles

- `Harvester`: positive guidance toward mining and resource access.
- `Builder`: positive guidance for expansion opportunities.
- `Firefighter`: movement and adjacent-unit relay guidance toward a critical city.
- `Attacker`: lowest-priority positional pressure after economy and survival assignments.

Role changes normally wait five turns. Firefighter may override cooldown only for a critical city that is not a SacrificialDecay target.

## City Roles

- `FuelDepot`: a delivery destination selected from fuel need, access, and city centrality.
- `ResearchStation`: research priority.
- `ManufacturingPoint`: worker-production priority.
- `SacrificialDecay`: one strictly bounded abandonment candidate.

FuelStation was removed because Lux city fuel is a one-way reserve and cannot be transferred back to units. SacrificialDecay requires a single city tile, zero fuel turns, repeated confirmation, no timely rescue, and sufficient distance from fuel resources.

## Lux Rules Preserved

- Unit `transfer` targets only an adjacent allied unit.
- Cargo entering a city becomes city fuel and cannot be withdrawn.
- Role logic never assumes city-to-unit or direct unit-to-city transfer actions.
- BUILD_CITY remains an Actor decision. `preserve_build_city_logit: true` blocks negative Role suppression.
- Legal masking is applied after all soft deltas.

## Runtime Engineering

- compact NumPy role codes and cooldown-expiry arrays;
- vectorized Manhattan distances and direction masks;
- grid-neighbor lookup instead of all-pairs transfer search;
- per-map worker budgets and Role scales;
- previous-assignment reuse when the update time budget is exceeded;
- Role trace fields for cooldown, changes, reasons, and timing.

Attacker is processed last. Completed research short-circuits unnecessary research-role work.

## Learning Interface

Role coefficients can be fixed YAML values or checkpointed trainable parameters. `RoleConditionedLocalAdapter` adds a bounded learned local delta while the Actor remains frozen. Rollout buffers carry compact Role codes, and optimizer groups can separately control Role, Local Adapter, Policy Head, and backbone-tail learning rates.

The accepted progression is:

1. Role bias and Local Adapter.
2. Final Policy/Value Heads.
3. Final ResNet blocks only after explicit paired-evaluation gates.

Global BUILD_CITY penalties and the Stage4 continuation are retired negative directions.

## Visualization

Replay generation may emit `*.roles.json` sidecars containing per-turn unit/city roles, cooldowns, changes, reasons, and bias status. The standalone Role-Colored Lux Viewer consumes those sidecars without changing official replay state.
