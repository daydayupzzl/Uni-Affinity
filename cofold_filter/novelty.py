"""Novelty of a complex relative to the co-folding model's training set (PDB up to CUTOFF).

L_tc_ccd runs as-is, the component list ships in reference/. P_mm and J_mm need a sequence search over
pre-cutoff chains and BioLiP's ligand table, see DATA.md.
"""
from __future__ import annotations

import argparse
import gzip
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
CUTOFF = "2021-09-30"
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else _MORGAN.GetFingerprint(mol)


@lru_cache(maxsize=1)
def ccd_reference(path: str | None = None) -> pd.DataFrame:
    """PDB chemical components released on or before CUTOFF: ccd, date, smiles."""
    source = Path(path) if path else HERE / "reference" / "ccd_components.csv.gz"
    with (gzip.open(source, "rt") if str(source).endswith(".gz") else open(source)) as handle:
        table = pd.read_csv(handle, keep_default_na=False)   # component 'NA' is sodium, not a missing value
    table = table[(table.date <= CUTOFF) & (table.smiles != "")]
    return table[["ccd", "date", "smiles"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def _ccd_fingerprints() -> tuple[list[str], list]:
    table = ccd_reference()
    codes, fps = [], []
    for row in table.itertuples():
        fp = fingerprint(row.smiles)
        if fp is not None:
            codes.append(row.ccd); fps.append(fp)
    return codes, fps


def ligand_novelty(smiles: str) -> dict:
    """L_tc_ccd for one ligand, plus which component it matched."""
    from rdkit import DataStructs

    query = fingerprint(smiles)
    if query is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    codes, fps = _ccd_fingerprints()
    similarity = np.asarray(DataStructs.BulkTanimotoSimilarity(query, fps))
    best = int(similarity.argmax())
    return {"L_tc_ccd": float(similarity[best]), "closest_ccd": codes[best]}


def protein_novelty(hits: pd.DataFrame) -> float:
    """P_mm from a sequence search: one row per pre-cutoff chain hit, columns fident (0-1) and
    query_coverage (0-1). We used mmseqs easy-search -s 7.5 against pre-cutoff pdb_seqres."""
    kept = hits[hits.get("query_coverage", 1.0) >= 0.5]
    return float(kept.fident.max() * 100) if len(kept) else 0.0


def complex_novelty(smiles: str, homologs: pd.DataFrame) -> dict:
    """J_mm: max over (homologous pre-cutoff chain c, ligand bound to c) of min(identity, Tanimoto)."""
    from rdkit import DataStructs

    query = fingerprint(smiles)
    if query is None:
        raise ValueError(f"RDKit could not parse {smiles!r}")
    smiles_of = dict(zip(ccd_reference().ccd, ccd_reference().smiles))

    best = {"J_mm": 0.0, "closest_chain_fident": None, "closest_ccd": None}
    l_tc_homolog, p_best = 0.0, 0.0
    for row in homologs.itertuples():
        reference_smiles = smiles_of.get(row.ccd)
        if reference_smiles is None:
            continue
        fp = fingerprint(reference_smiles)
        if fp is None:
            continue
        identity = float(row.fident) * 100
        tanimoto = DataStructs.TanimotoSimilarity(query, fp) * 100
        p_best = max(p_best, identity)
        if identity >= 30:
            l_tc_homolog = max(l_tc_homolog, tanimoto / 100)
        joint = min(identity, tanimoto)
        if joint > best["J_mm"]:
            best = {"J_mm": joint, "closest_chain_fident": identity, "closest_ccd": row.ccd}
    return {**best, "L_tc_homolog": l_tc_homolog, "P_best_with_ligand": p_best}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ligand novelty (L_tc_ccd) for one SMILES or a CSV of them.")
    ap.add_argument("--smiles", help="a single SMILES")
    ap.add_argument("--csv", help="CSV with a smiles column")
    ap.add_argument("--column", default="smiles")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    reference = ccd_reference()
    print(f"[novelty] {len(reference)} pre-cutoff PDB components (release date <= {CUTOFF})")
    if args.smiles:
        print(ligand_novelty(args.smiles))
    elif args.csv:
        frame = pd.read_csv(args.csv)
        values = [ligand_novelty(s) for s in frame[args.column]]
        out = frame.join(pd.DataFrame(values, index=frame.index))
        out.to_csv(args.output, index=False) if args.output else print(out.to_string(index=False))
    else:
        ap.error("pass --smiles or --csv")


if __name__ == "__main__":
    main()
