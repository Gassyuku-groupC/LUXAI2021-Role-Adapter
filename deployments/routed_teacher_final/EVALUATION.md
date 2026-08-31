# routed_teacher_final Evaluation

## Score

```text
Score = final city tiles + final units
```

Timeouts are reported separately and are not counted as completed losses.

## Versus first

Seeds `20260920` through `20260929`, all four maps, both sides, 180-second outer timeout, one attempt.

| Scope | Completed | Failed | Win rate | Timeout | Mean Score | Score margin | City margin | Unit margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 73 | 7 | **52.1%** | 8.8% | **150.49** | **+16.12** | **+7.86** | **+8.26** |
| 12x12 | 20 | 0 | 35.0% | 0.0% | 54.50 | -23.10 | -12.10 | -11.00 |
| 16x16 | 20 | 0 | 55.0% | 0.0% | 95.95 | -12.90 | -7.85 | -5.05 |
| 24x24 | 20 | 0 | 60.0% | 0.0% | 204.80 | +0.40 | +0.80 | -0.40 |
| 32x32 | 13 | 7 | 61.5% | 35.0% | 298.54 | +145.31 | +73.62 | +71.69 |
| P0 | 37 | 3 | 54.1% | 7.5% | 149.49 | +11.84 | +5.41 | +6.43 |
| P1 | 36 | 4 | 50.0% | 10.0% | 151.53 | +20.53 | +10.39 | +10.14 |

Result: 38 wins from 73 completed games. All seven failures occurred on 32x32.

## Versus GroupH

Seeds `20260925` and `20260926`, both sides.

| Map | Completed | Wins | Win rate | Timeout | City margin | Unit margin |
|---|---:|---:|---:|---:|---:|---:|
| 12x12 | 4 | 2 | 50.0% | 0.0% | +11.75 | +8.00 |
| 16x16 | 4 | 2 | 50.0% | 0.0% | -3.00 | -1.50 |
| 24x24 | 4 | 3 | 75.0% | 0.0% | -12.00 | -10.25 |
| 32x32 | 0 | 0 | N/A | 100.0% | N/A | N/A |

Completed-game win rate on 12/16/24 was 58.3% (7/12). Four scheduled 32x32 games timed out.

## Limitations

- 12x12 remains weaker than `first` despite the dedicated specialist.
- 32x32 completed games are strong, but timeout stability is unresolved.
- GroupH evaluation uses only two seeds and is a screen rather than a promotion-grade suite.
