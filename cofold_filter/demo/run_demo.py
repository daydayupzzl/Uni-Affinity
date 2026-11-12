#!/usr/bin/env python
"""python cofold_filter/demo/run_demo.py

Recomputes the ligand pLDDT and the ligand novelty from the files in structures/ and applies the filter.
J_mm comes from cases.csv (it needs references too large to ship), the RMSD only checks the verdicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from filter import apply_to_frame, thresholds  # noqa: E402
from ligand_plddt import ligand_plddt  # noqa: E402
from novelty import ligand_novelty  # noqa: E402

TOL = 1e-4


def main() -> int:
    cases = pd.read_csv(HERE / "cases.csv")

    print("quality: mean pLDDT over the target ligand's atoms, recomputed from each prediction's plddt_*.npz")
    print(f"   {'case':<32} {'ligand':>8} {'complex':>9} {'non-protein':>12}")
    recomputed = []
    for row in cases.itertuples():
        base = HERE / "structures" / row.case / "boltz_output"
        q = ligand_plddt(base / "predictions" / row.system_id,
                         base / "processed" / "structures" / f"{row.system_id}.npz",
                         row.model_ligand_chain)
        recomputed.append(q)
        print(f"   {row.case:<32} {q['lig_plddt']:>8.3f} {q['complex_plddt']:>9.3f} {q['nonprotein_plddt']:>12.3f}")
        assert abs(q["lig_plddt"] - row.lig_plddt) < TOL, f"{row.case}: lig_plddt drifted"
    print()

    print("novelty vs. the PDB up to 2021-09-30. L_tc_ccd is recomputed against ../reference/; P_mm and J_mm")
    print("come from cases.csv, they need a sequence search and BioLiP's ligand table (see ../novelty.py)")
    print(f"   {'case':<32} {'P_mm':>7} {'L_tc_ccd':>9} {'closest':>8} {'J_mm':>7}")
    print(f"   {'':<32} {'(table)':>7} {'(recomp)':>9} {'':>8} {'(table)':>7}")
    for row in cases.itertuples():
        got = ligand_novelty(row.ligand_smiles)
        assert abs(got["L_tc_ccd"] - row.L_tc_ccd) < TOL, f"{row.case}: L_tc_ccd drifted"
        print(f"   {row.case:<32} {row.P_mm:>7.1f} {got['L_tc_ccd']:>9.3f} {got['closest_ccd']:>8} {row.J_mm:>7.2f}")
    print()

    print("verdict: J_mm picks the tier, the tier sets the ligand pLDDT the prediction has to reach")
    print(thresholds().to_string(index=False))
    print()
    out = apply_to_frame(cases, policy="strict")
    width = max(len(c) for c in out.case)
    failures = 0
    for row in out.itertuples():
        outcome = "correct pose" if row.true_success else "WRONG pose"
        print(f"   {row.case:<{width}}  pLDDT {row.lig_plddt:.3f} vs {row.threshold_strict:.2f} needed"
              f"  -> {row.state:<9}  [true RMSD {row.true_ligand_rmsd:6.2f} A -> {outcome}]")
        if row.state == "accepted" and not row.true_success:
            failures += 1
            print(f"   {'':<{width}}  ^ accepted a wrong pose")
    print()
    print("A single global threshold on the same axis is 0.84, which accepts the middle case; its tier needs")
    print("0.91. 'accepted' is ~90% precision, not a guarantee -- see README.md.")

    if failures:
        print(f"\nFAIL: {failures} case(s) accepted a wrong pose")
        return 1
    print("\nOK: recomputed values match cases.csv and no wrong pose was accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
