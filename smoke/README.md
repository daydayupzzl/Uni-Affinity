# Smoke test

64 complexes from the `S3_cold_target_seq30cc` test split, seed 0 — the cold-target setting, where no test
protein shares a 30%-identity cluster with any training protein.

```bash
python smoke/run_smoke.py
python smoke/run_smoke.py --device cuda
```

It scores the 64 complexes with the released checkpoint and compares every prediction against what the
paper's run produced. Tolerance 1e-4, actual agreement around 8e-6, which is CPU-versus-GPU float
arithmetic.

```
cache/<hash>/*.pt    64 featurized graphs, 28 MiB
checkpoint/best.pt   the seed-0 model, 10.5 MiB, config inside, no optimizer state
index.csv            the eight columns the code reads; structure paths are placeholders
split.csv            all 64 marked test
expected.json        reference predictions + the full-split metrics
```

The graphs already carry the ESM-2 and Uni-Mol node features, so no structure file or pretrained model is
touched. `<hash>` is over the featurization settings; the dataset code recomputes it and would refuse to
reuse this cache under a different graph config.

`run_smoke.py` reports the metrics of the full split (r = 0.5481, RMSE = 1.7123, n = 1609) from
`expected.json`; the 64 bundled complexes are a reproduction check, not a measurement.
