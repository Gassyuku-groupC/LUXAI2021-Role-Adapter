# Documentation

[Project README](../README.md)

## Start here

| Document | Contents |
| --- | --- |
| [Deployment and evaluation](DEPLOYMENT.md) | Environment setup, Git LFS verification, packaging, paired evaluation, and replay viewers |
| [Training guide](../TRAINING.md) | Local artifacts, path overrides, training stages, and package rebuilding |
| [Role-City adapter](../ROLE.md) | Worker/city roles, runtime constraints, and learning interface |
| [Methodology](../METHODOLOGY.md) | Architecture, training evolution, and evidence boundaries |
| [Solution overview](../solution.md) | Concise explanation of the final system |
| [Published evaluation](../deployments/routed_teacher_final/EVALUATION.md) | Recorded results, map/side breakdowns, and limitations |

The current deployment is
[`routed_teacher_final`](../deployments/routed_teacher_final/), with route
identities and weight hashes in its
[package manifest](../deployments/routed_teacher_final/PACKAGE_MANIFEST.json).

## Research and historical notes

| Document | Status |
| --- | --- |
| [Best/C replay catalog](BEST_C_REPLAY_CATALOG.md) | Offline-data pipeline; generated datasets are local artifacts |
| [Critical-state preference training](CRITICAL_STATE_PREFERENCE_TRAINING.md) | Research-only path; not used to select final weights |
| [Role curriculum Stage 4](ROLE_CURRICULUM_STAGE4.md) | Retired negative experiment |
| [Japanese presentation notes](PRESENTATION_RISK_ROLE_ARCHITECTURE_JA.md) | Explanation of the final routed agent and Role-City contribution |
| [Paired seed suites](../conf/evaluation/paired_seed_suites.yaml) | Development/promotion/holdout protocol for 16/24/32 maps |

Research commands assume the repository root and locally prepared artifacts.
They do not imply that datasets, opponents, or intermediate checkpoints are
included in a fresh clone.

## Visualization

The [bundled viewer](../tools/role_replay_viewer/README.md) opens a stateful
replay and a role sidecar separately. The separate
[Role-Colored Lux Viewer](https://github.com/Gassyuku-groupC/Role-Colored-Lux-Viewer)
uses a command replay with embedded role frames. See the
[replay workflow](DEPLOYMENT.md#role-replays) before choosing files.
