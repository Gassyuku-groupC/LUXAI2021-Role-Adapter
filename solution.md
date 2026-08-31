# Final Solution Overview

## Summary

The final Agent is `routed_teacher_final`: a self-contained Lux AI 2021 submission that selects a validated policy checkpoint by map size, applies Rot180 inference, and adds positive Role-City guidance on 16x16, 24x24, and 32x32.

```text
12x12 -> er100_35072
16x16 -> role_05376_nofs
24x24 -> log_03584
32x32 -> role_05376_nofs
```

All routes use Rot180. Role bias is bypassed on 12x12. Risk Gate is disabled and FuelStation is removed.

## Why Routing

Lux policies are highly map-size sensitive. A single small logit change can alter early expansion and create a large late-game snowball. Paired evaluation showed that no one continuation checkpoint dominated every map. Routing preserves each specialist only on the map where it was selected by replay evidence while keeping one shared observation/action interface.

## Role-City Adapter

The Actor first produces normal policy logits. `RoleCityAdapter` classifies workers and cities, then applies bounded positive deltas before legal masking. It never directly forces an action.

- Harvester supports resource collection.
- Builder supports expansion.
- Firefighter supports movement and adjacent-unit fuel relay toward critical cities.
- Attacker is lowest priority.
- ResearchStation and ManufacturingPoint guide city actions.
- SacrificialDecay is restricted to one strictly qualified city.

BUILD_CITY is protected. City fuel is treated as a one-way reserve; transfer exists only between adjacent allied units.

## Training

The system progressed from a frozen Role baseline through Role/Local APPO, targeted executable-opponent adaptation, weak best-policy KL anchoring, and limited Policy Head training. Terminal outcome remained the dominant reward. Checkpoints were selected only by paired matches, not by learner loss.

## Evaluation

Against `first`, using ten fixed seeds, all map sizes, and both sides:

- 73 valid games and seven 32x32 timeouts;
- 38 wins, completed-game win rate `52.1%`;
- mean Score `150.49`, with `Score = city tiles + units`;
- city margin `+7.86` and unit margin `+8.26`.

12x12 remains the weakest map. 32x32 is strong in completed games but has unresolved local replay-generation/runtime timeout behavior.

## Research Extensions

The repository retains tile-level Spatial Risk Sidecar and zero-init Intervention Gate implementations. They are disabled in the final package because enabled candidates did not pass paired promotion tests.
