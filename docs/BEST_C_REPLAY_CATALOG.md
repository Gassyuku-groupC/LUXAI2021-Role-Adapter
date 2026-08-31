# Best/C Replay Catalog

> Status: reproducible offline-data pipeline. The catalog supported diagnosis and training design, but generated datasets are local artifacts and are not required to run `deployments/routed_teacher_final`.

`build_best_c_replay_catalog.py` treats Group C in `first battle` and deployed
`best_agent` opponents as the same teacher identity. Replays are deduplicated by
map, seed, and the complete action sequence. Stateful and command-only copies of
the same match therefore remain in one replay group and one dataset split.

The catalog records the canonical action replay, the preferred stateful metric
copy, all duplicate paths, outcome, teacher side, opponent, failure turn, and a
deterministic train/validation/calibration split. All replays sharing the same
map size and seed are assigned to the same split.

`build_weighted_bc_index_from_catalog.py` applies these policies:

- Best/C wins are positive expert demonstrations.
- Small-map Best/C wins receive extra anchor weight.
- Best/C losses contribute only the safe prefix before the failure window.
- The final 20 turns before a detected failure are excluded from positive BC.
- When C loses to B, G, or D in first battle, the winner is added as an external
  expert. B receives the largest weight on 24x24 and 32x32.
- Actions immediately followed by city loss are strongly downweighted.

The output index remains frame-level, but splitting and deduplication are always
performed at the replay fingerprint level.
