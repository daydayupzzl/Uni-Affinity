# Data

## 1. PDBbind v2020

From [pdbbind.org.cn](http://www.pdbbind.org.cn/), registration required. We use `refined` and
`general_minus_refined`, which do not overlap. Labels are `pK = -log10(K / M)`; the loader keeps
`relation == '='` only, so censored rows are not scored.

The index CSV the code expects: `pdb_id`, `pK`, `relation`, `affinity_type`, `uniprot_id`, `pocket_pdb`,
`ligand_sdf`, `ligand_mol2`. The last three are paths, absolute or relative to `data.root`.
`smoke/index.csv` is a worked example.

## 2. Splits (`data/splits/`)

Two-column `pdb_id,split` CSVs. A complex absent from a file is unused, which is how the CASF-2016 core is
held out. Censored labels are dropped by the loader instead, so the complexes actually scored are slightly fewer
than `n_test`.

| setting | held out | seeds |
|---|---|---|
| `S1_random` | nothing, random assignment | 5 |
| `S2_cold_ligand_scaffold` | Bemis–Murcko scaffolds | 5 |
| `S2b_cold_ligand_ecfp40` | ligands with ECFP Tanimoto > 0.4 to any training ligand | 5 |
| `S3_cold_target_seq30cc` | proteins in the same single-linkage 30%-identity component | 5 |
| `S3b_cold_target_mmseqs30` | proteins in the same MMseqs2 30% cluster | 5 |
| `S3c_cold_target_seq90` | proteins in the same 90% cluster (a leaky control) | 5 |
| `S3d_cold_target_strict` | 30% sequence identity and 0.60 pocket TM-score | 5 |
| `S4_cold_both` | cold ligand and cold target at once | 5 |
| `S5_time` | everything deposited after a cut-off year | 1 |
| `S7_within_target` | ligands of targets that are also in training | 5 |

`seq30cc` and `mmseqs30` are different objects, not two names for one thing. `S3b` uses MMseqs2
set-cover clusters, where membership is defined by a representative, so homologues can still end up on
both sides of a split. `S3` instead takes connected components of all-against-all hits at 30% identity
over at least 50 aligned residues: single linkage, and strictly more conservative.

`data/splits/summary.csv` has the sizes per split and seed. `n_drop` counts rows marked `drop` in the split file
itself, which is a fourth value alongside `train`, `valid` and `test`: `S4` is the only setting that
produces them, because a complex whose target is too close to the test targets and whose ligand is too
close to the training ligands can go on neither side.

Rebuilding the splits needs MMseqs2 for the sequence clusters and the all-against-all identities
(set-cover clustering at coverage 0.8 for the 90% and 30% settings), Foldseek in exhaustive mode for
pocket similarity (two pockets are linked when the smaller of the two directional TM-scores is at least
0.60 over an alignment of at least 20 residues), and ECFP4 Taylor-Butina clustering at Tanimoto 0.40 for
the chemotype setting.

## 3. ESM-2 and Uni-Mol caches

Node inputs are concatenations: 38 geometric features + 1280 from ESM-2 650M
(`facebook/esm2_t33_650M_UR50D`) per residue, 43 + 512 from Uni-Mol (`unimolv1`) per ligand heavy atom, both
built offline into `<id>.pt` files.

The caches are not shipped; the bundled graphs already carry the features. Featurizing new complexes does
need them, so this is the layout the featurizer checks and will refuse to proceed without:

* ESM: a dict with `embedding` ([n_residues, 1280], every residue of the structure, not just the pocket) and
  `residue_keys`, `chain:resnum[icode]` in exactly the order `data.protein_atom_table` produces: waters
  dropped, metals kept, file order. Pocket rows are selected by index; a `residue_keys` mismatch is an error.
* Uni-Mol: a dict with `embedding` ([n_heavy_atoms, 512]), `symbols` and `coords`, taken from the pose, not
  from a conformer regenerated from SMILES. Symbols and coordinates are re-checked (1e-3 Å) against the
  ligand just read, and the aromatic virtual nodes are derived as the mean over their ring atoms.
* A missing cache file is zero-filled and warned about rather than raising, so the warnings matter: such a
  complex is scored without its pretrained features and is otherwise indistinguishable from a normal
  prediction.

## 4. SAIR metadata (`data/sair_metadata_accepted.csv.gz`)

[SAIR](https://huggingface.co/datasets/SandboxAQ/SAIR) pairs Boltz-1x co-folded structures with IC50
labels; the structures themselves are not here. This table is the metadata for the complexes our
reliability filter accepts, which is the subset the scaling experiments treat as structurally
trustworthy, one row per complex.

SAIR is released by SandboxAQ under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and this
table is redistributed under the same licence. `entry_id`, `shard`, `protein`, `family`, `SMILES`, `n_res`,
`n_lig_heavy`, `pIC50` and `n_meas` are SAIR's own columns, unchanged; the rest are ours.

## 5. Runs N' Poses, PoseBench, ChEMBL

* **Runs N' Poses** ([paper](https://arxiv.org/abs/2502.03642), data on Zenodo) — crystal structures and
  predictions from several co-folding models. We re-folded all of it with Boltz-1 and calibrated the
  filter on those predictions.
* **PoseBench** — external validation of the filter, restricted to systems released after the Boltz-1
  training cut-off.
* **ChEMBL** (v37) and **BindingDB** — activity labels, and the novelty distribution the filter's
  precision is projected onto.
* `cofold_filter/reference/ccd_components.csv.gz` — PDB chemical components with their release dates
  (`pdbx_initial_date`), which is what `L_tc_ccd` compares against.

`P_mm` and `J_mm` additionally need the chains of `pdb_seqres` released on or before the cut-off, and
BioLiP's table of which biologically relevant ligands each PDB chain binds. Both are public.
