# Lux Role Replay Viewer

[Project README](../../README.md) ·
[Replay generation guide](../../docs/DEPLOYMENT.md#role-replays)

This bundled browser viewer displays a **stateful replay** and a matching role
sidecar. It draws unit role rings and city role overlays without modifying the
replay.

## Open the viewer

Open [`index.html`](index.html) from your local clone in a browser, then load
both files through the viewer's file controls:

| Input | Example |
| --- | --- |
| Converted stateful replay | `<match>.json` |
| Matching role sidecar | `<match>.roles.json` |

The replay must contain a nonempty `stateful` array. A command-only
`<match>.commands.json` file has to be converted first. Match the seed and the
intended traced player. Unit and city role overlays can be toggled independently.

## Generate the inputs

The repository's
[`generate_deployed_agent_replays.ps1`](../../scripts/generate_deployed_agent_replays.ps1)
generates command replays, converted stateful copies, and, when tracing is
enabled, role sidecars. Follow the
[deployment guide](../../docs/DEPLOYMENT.md#role-replays) from the repository root.

The separate
[Role-Colored Lux Viewer](https://github.com/Gassyuku-groupC/Role-Colored-Lux-Viewer)
preserves the original Lux artwork and uses a command replay with embedded
`roleFrames`. Follow its merge instructions for that viewer.
