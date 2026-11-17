"""python cofold_filter/filter.py --csv predictions.csv --policy strict

J_mm picks the novelty tier, the tier sets the ligand pLDDT the prediction has to reach. The csv needs
lig_plddt and J_mm; other columns are passed through.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
POLICIES = ("strict", "balanced", "lenient")


def thresholds(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or HERE / "thresholds.csv")


def tier_of(j_mm: float, table: pd.DataFrame | None = None) -> str:
    """Novelty tier of one complex. Bins are left-closed, right-open: J = 20 is in 'J 20-40'."""
    table = thresholds() if table is None else table
    for row in table.itertuples():
        if row.J_min <= j_mm < row.J_max:
            return row.J_tier
    raise ValueError(f"J_mm={j_mm} is outside [0, 100]")


def classify(lig_plddt: float, j_mm: float, table: pd.DataFrame | None = None) -> dict:
    """accepted / uncertain / rejected."""
    table = thresholds() if table is None else table
    tier = tier_of(j_mm, table)
    row = table[table.J_tier == tier].iloc[0]
    if lig_plddt >= row.lig_plddt_strict:
        state = "accepted"
    elif lig_plddt >= row.lig_plddt_lenient:
        state = "uncertain"
    else:
        state = "rejected"
    return {"J_tier": tier, "state": state,
            **{f"threshold_{p}": float(row[f"lig_plddt_{p}"]) for p in POLICIES}}


def apply_to_frame(frame: pd.DataFrame, policy: str = "strict") -> pd.DataFrame:
    """Add J_tier / state / passes_<policy> columns. `policy` only affects the pass/fail column."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy}")
    table = thresholds()
    verdicts = pd.DataFrame([classify(r.lig_plddt, r.J_mm, table) for r in frame.itertuples()],
                            index=frame.index)
    out = frame.join(verdicts)
    out[f"passes_{policy}"] = out.lig_plddt >= out[f"threshold_{policy}"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="CSV with at least lig_plddt and J_mm columns")
    ap.add_argument("--policy", default="strict", choices=POLICIES)
    ap.add_argument("--output", default=None, help="where to write the annotated CSV (default: stdout)")
    args = ap.parse_args()

    frame = pd.read_csv(args.csv)
    for column in ("lig_plddt", "J_mm"):
        if column not in frame:
            raise SystemExit(f"{args.csv} has no {column} column")
    out = apply_to_frame(frame, args.policy)
    if args.output:
        out.to_csv(args.output, index=False)
        counts = out.state.value_counts().reindex(["accepted", "uncertain", "rejected"], fill_value=0)
        print(f"wrote {args.output}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
