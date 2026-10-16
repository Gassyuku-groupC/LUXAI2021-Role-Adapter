# Lux AI 2021 Role Adapter

A Lux AI 2021 agent with a **plug-and-play Role-City adapter**, built on the
first-place training stack and its 24-block ResNet Actor. The final evaluated
deployment, `routed_teacher_final`, selects a checkpoint by map size and uses
Rot180 inference.

[Get started](#get-started) · [Deployment guide](docs/DEPLOYMENT.md) ·
[Training](TRAINING.md) · [Role-City design](ROLE.md) ·
[Documentation index](docs/README.md)

## Get started

Install [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
before cloning so the checkpoint files are downloaded:

```sh
git lfs install
git clone https://github.com/Gassyuku-groupC/LUXAI2021-Role-Adapter.git
cd LUXAI2021-Role-Adapter
git lfs pull
git lfs ls-files
```

Run subsequent commands from this repository root. No particular drive letter,
Windows username, or parent directory is required.

| Goal | Start here |
| --- | --- |
| Use the published agent | [Deployment and environment setup](docs/DEPLOYMENT.md) |
| Reproduce paired evaluation | [Evaluation commands](docs/DEPLOYMENT.md#paired-evaluation) |
| Continue training with local artifacts | [Training and local path configuration](TRAINING.md) |
| Understand the architecture | [Methodology](METHODOLOGY.md) and [solution overview](solution.md) |
| Inspect role-colored replays | [Replay viewers](#replay-viewers) |

The published agent lives in
[`deployments/routed_teacher_final/`](deployments/routed_teacher_final/).
Use that package for the reported final policy. Repository-root `main.py`
loads the working configs in `lux_ai/rl_agent/`, which differ from the
published deployment.

## Final deployment

| Map | Policy checkpoint | Rot180 | Role bias |
| --- | --- | :---: | :---: |
| 12x12 | `er100_35072` | ON | OFF |
| 16x16 | `role_05376_nofs` | ON | ON |
| 24x24 | `log_03584` | ON | ON |
| 32x32 | `role_05376_nofs` | ON | ON |

The three weight files use Git LFS. Route identities and SHA256 values are
recorded in the
[package manifest](deployments/routed_teacher_final/PACKAGE_MANIFEST.json);
the [deployment guide](docs/DEPLOYMENT.md#verify-downloaded-weights) includes a
local hash check.

```text
Game observation
  -> map-size checkpoint router
  -> 24-block ResNet Actor
  -> Rot180 policy ensemble
  -> RoleCityAdapter positive soft guidance (16/24/32)
  -> legal-action mask and collision handling
  -> Lux commands
```

Risk Gate is disabled in the final deployment. The obsolete FuelStation role is
removed, and `preserve_build_city_logit` protects BUILD_CITY from negative Role
bias.

## Role-City contribution

- Worker roles: Harvester, Builder, Firefighter, and lowest-priority Attacker.
- City roles: FuelDepot, ResearchStation, ManufacturingPoint, and strictly
  bounded SacrificialDecay.
- Five-turn role cooldown with a critical Firefighter override.
- Positive additive logit guidance; actions remain selected by the policy.
- Transfer guidance only between adjacent allied units.
- Compact NumPy state, vectorized distances, grid-neighbor lookup, and a bounded
  runtime fallback.

See [ROLE.md](ROLE.md) for exact role semantics, runtime constraints, and the
learning interface.

## Reported evaluation

Against `first`, seeds `20260920` through `20260929`, four map sizes and both
player positions:

| Metric | Recorded result |
| --- | ---: |
| Scheduled games | 80 |
| Completed games | 73 |
| Timeouts | 7, all on 32x32 |
| Wins / completed-game win rate | 38 / 52.1% |
| Mean Score | 150.49 |
| Mean city margin | +7.86 |
| Mean unit margin | +8.26 |

`Score = final city tiles + final units`. Timeouts are reported separately from
completed-game win rate. These are the recorded evaluation results; cloning the
repository does not rerun the suite.

The [evaluation report](deployments/routed_teacher_final/EVALUATION.md) contains
map/side breakdowns, the GroupH screen, and limitations. In particular, 12x12
strength and 32x32 timeout stability remain unresolved.

## Replay viewers

| Viewer | Input |
| --- | --- |
| [Bundled role replay viewer](tools/role_replay_viewer/README.md) | Stateful replay JSON plus matching `*.roles.json`, loaded separately |
| [Role-Colored Lux Viewer](https://github.com/Gassyuku-groupC/Role-Colored-Lux-Viewer) | A command replay with embedded role frames, produced by its merge script |

The [replay helper](scripts/generate_deployed_agent_replays.ps1) can produce
command replays, stateful copies, and role sidecars. See the
[replay workflow](docs/DEPLOYMENT.md#role-replays) for the correct viewer input.

## Repository guide

| Location | Purpose |
| --- | --- |
| [`deployments/routed_teacher_final/`](deployments/routed_teacher_final/) | Published agent package and evaluation evidence |
| [`lux_ai/`](lux_ai/) | Agent, Role-City, environment, model, and learner source |
| [`conf/`](conf/) | Training configurations and evaluation seed suites |
| [`scripts/`](scripts/) | Packaging, replay generation, analysis, and training tools |
| [`tests/`](tests/) | Agent and training-component tests |
| [`tools/role_replay_viewer/`](tools/role_replay_viewer/) | Browser-based stateful replay viewer |
| [`docs/`](docs/README.md) | Deployment guide, research notes, and presentation material |
| [`TRAINING.md`](TRAINING.md) | Local artifacts, portable overrides, and training checks |

Datasets, opponent packages, intermediate checkpoints, replay corpora, and
generated outputs are local artifacts; see [TRAINING.md](TRAINING.md).
Spatial Risk Sidecar and Intervention Gate implementations remain available as
research extensions and are disabled in the final deployment.

## Attribution

Based on [IsaiahPressman/Kaggle_Lux_AI_2021](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021).
Original attribution and license terms remain applicable.
