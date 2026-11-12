"""Mean pLDDT over the target ligand's atoms, from a Boltz-1 prediction directory.

Boltz writes one pLDDT per token: one per standard residue, one per atom on a non-polymer chain, so the
ligand's tokens are the ones on the ligand chain. demo/ has worked examples.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def tokens(structure_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    """(chain name, mol_type) per token. mol_type follows Boltz: 0 protein, 1 DNA, 2 RNA, 3 non-polymer."""
    blob = np.load(structure_npz)
    chains, residues = blob["chains"][blob["mask"]], blob["residues"]
    chain_of, type_of = [], []
    for chain in chains:
        span = residues[chain["res_idx"]: chain["res_idx"] + chain["res_num"]]
        for residue in span:
            count = 1 if residue["is_standard"] else int(residue["atom_num"])
            chain_of += [str(chain["name"])] * count
            type_of += [int(chain["mol_type"])] * count
    return np.asarray(chain_of), np.asarray(type_of)


def ligand_plddt(prediction_dir: Path, structure_npz: Path, ligand_chain: str, model: int = 0) -> dict:
    """Mean pLDDT over the tokens of `ligand_chain`, with the two weaker scopes alongside it."""
    prediction_dir, structure_npz = Path(prediction_dir), Path(structure_npz)
    system = structure_npz.stem
    plddt = np.load(prediction_dir / f"plddt_{system}_model_{model}.npz")["plddt"]
    chain_of, type_of = tokens(structure_npz)
    if len(plddt) != len(chain_of):
        raise ValueError(f"{system}: {len(plddt)} pLDDT values but {len(chain_of)} tokens -- "
                         f"the prediction and the structure file do not belong together")
    selected = chain_of == ligand_chain
    if not selected.any():
        raise ValueError(f"{system}: no token on chain {ligand_chain!r}; chains are {sorted(set(chain_of))}")
    if not (type_of[selected] == 3).all():
        raise ValueError(f"{system}: chain {ligand_chain!r} is not a non-polymer chain")
    interface = type_of != 0
    # NOTE complex_plddt / nonprotein_plddt are only here for comparison, nothing downstream reads them
    return {"system": system, "ligand_chain": ligand_chain,
            "lig_plddt": float(plddt[selected].mean()),
            "lig_plddt_min": float(plddt[selected].min()),
            "lig_n_tokens": int(selected.sum()),
            "complex_plddt": float(plddt.mean()),
            "nonprotein_plddt": float(plddt[interface].mean()) if interface.any() else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prediction", required=True, help="Boltz predictions/<id> directory")
    ap.add_argument("--structure", required=True, help="Boltz processed/structures/<id>.npz")
    ap.add_argument("--ligand-chain", required=True, help="chain id of the target ligand in the model")
    ap.add_argument("--model", type=int, default=0)
    args = ap.parse_args()
    for key, value in ligand_plddt(Path(args.prediction), Path(args.structure), args.ligand_chain, args.model).items():
        print(f"{key:>18}: {value}")


if __name__ == "__main__":
    main()
