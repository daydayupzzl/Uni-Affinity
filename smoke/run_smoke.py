#!/usr/bin/env python
"""python smoke/run_smoke.py

Scores the bundled complexes with the released checkpoint and checks them against expected.json.
Needs no download: the cached graphs already carry the ESM-2 / Uni-Mol features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-4, help="max allowed |prediction - reference|")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    from uniaffinity.trainer import evaluate, get_device, load_model, make_loaders

    reference = json.loads((HERE / "expected.json").read_text())
    checkpoint = HERE / "checkpoint" / "best.pt"

    device = get_device(args.device)
    print(f"[smoke] device: {device}")
    model, config = load_model(checkpoint, device)
    loaders, _, _ = make_loaders(config, HERE.parent)
    _, records = evaluate(model, loaders["test"], device, config["loss"])

    got = {r["pdb_id"]: float(r["prediction"]) for r in records}
    want = reference["predictions"]
    missing = sorted(set(want) - set(got))
    if missing:
        print(f"[smoke] FAIL: {len(missing)} complexes were not scored, e.g. {missing[:5]}")
        return 1

    worst_id, worst = max(((k, abs(got[k] - v)) for k, v in want.items()), key=lambda t: t[1])
    print(f"[smoke] n={len(want)}  max |delta| = {worst:.2e} on {worst_id}  (tolerance {args.tol:g})")

    print(f"[smoke] {reference['unit']} test split (n={reference['full_test_n']}): "
          f"Pearson r = {reference['full_test_pearson']:.4f}, RMSE = {reference['full_test_rmse']:.4f}")

    if worst > args.tol:
        print("[smoke] FAIL: predictions drifted beyond the tolerance.")
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
