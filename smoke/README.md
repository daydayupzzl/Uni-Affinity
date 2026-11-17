# Smoke test

This directory serves two purposes: confirming that the released checkpoint still reproduces its
reference predictions, and providing the graphs that `tests/` runs on. The complexes come from the
`S3_cold_target_seq30cc` test split, seed 0 — the cold-target setting, where no test protein shares a
30%-identity cluster with any training protein.

| | |
|---|---|
| `cache/<id>/*.pt` | the featurized graphs, with the ESM-2 and Uni-Mol node features already in them |
| `checkpoint/model.pt` | the released model, config inside, no optimizer state |
| `index.csv` | the eight columns the code reads; structure paths are placeholders |
| `split.csv` | every row marked test |
| `structure/` | one pocket and one ligand, the featurization test's input |
| `expected.json` | reference predictions and the full-split metrics |

```bash
python smoke/run_smoke.py
python smoke/run_smoke.py --device cuda
```

It scores them with the released checkpoint and compares each prediction against `expected.json`. Any
residual difference is CPU-versus-GPU float arithmetic. The full-split metrics it prints are read from
`expected.json`; those are this one seed, not an average.

`<id>` is derived from the featurization settings; the dataset code recomputes it and would refuse to
reuse this cache under a different graph config.
