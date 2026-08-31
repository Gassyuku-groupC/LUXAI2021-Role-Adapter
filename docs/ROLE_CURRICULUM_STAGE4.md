# Role Curriculum Stage 4

> Status: retired negative experiment. Stage4 checkpoints reduced competitive strength and are not valid continuation points. The final deployment is `routed_teacher_final` with `er100_35072` on 12x12, `role_05376_nofs` on 16x16/32x32, and `log_03584` on 24x24.

## Objective

Continue from `role_05376` without weakening its BUILD_CITY behavior:

- improve 16x16 operation against `first`;
- improve 24x24 pre-night fueling and city survival;
- project historical Group C/best 32x32 weaknesses into a low-weight auxiliary curriculum.

The immutable `best_agent` remains a weak KL and teacher-BC drift anchor rather
than a behavior ceiling. Stage 4A uses KL `0.005` and anneals teacher BC from
`0.03` to `0.01` over 100 games. Spatial Sidecar and Risk Gate remain disabled.

## Historical 32x32 diagnosis

Group C used the same policy identity as `best_agent`. Its two 32x32 losses were:

| Opponent | Side | City margin t120 | t240 | Final | Total night loss | Worst single loss |
|---|---:|---:|---:|---:|---:|---:|
| B | P0 | -2 | -14 | -57 | 48 | 12 at turn 73 |
| G | P0 | -4 | -18 | -18 | 62 | 16 at turn 194 |

The 32x32 win controls averaged `+10.3` city tiles at turn 120 and `+47.0` at
turn 240. Early BUILD_CITY count was not lower in the losses (`84.5`) than in
the wins (`80.0`). The projected weakness is therefore expansion quality,
positioning, and preservation of the expansion, not insufficient BUILD_CITY
frequency. No global BUILD_CITY penalty is allowed.

## Trainable surface

- Frozen: best Actor/backbone, policy/value heads, Sidecar, Risk Gate.
- Trainable: Role-conditioned Local Adapter and 14 compact Role Bias values.
- BUILD_CITY protection: Attacker and Firefighter negative BUILD_CITY codes are
  omitted in both training and deployment paths.
- Map sampling: 16x16 `3/8`, 24x24 `4/8`, 32x32 `1/8`.

## First run

Run 100 games from `role_05376`. Stop immediately on non-finite loss or
gradient. Training loss alone cannot promote a checkpoint.

## Evaluation gates

Use fixed paired seeds and both sides.

- 16x16 vs `first`: win rate and city/unit margins must improve over
  `role_05376`.
- 24x24 vs `best_agent`: reduce `night loss > 40` cases without lowering the
  paired BUILD_CITY rate materially.
- 32x32 vs `best_agent`: evaluate only complete seed pairs. Require no timeout
  regression and no systematic negative city margin at turns 120 and 240.
- Reject any checkpoint whose BUILD_CITY count falls by more than 5% on matched
  completed games.
- Keep Teacher KL controlled; two consecutive 100-game stages without paired
  improvement stop this route.
