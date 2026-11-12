# Data

Nothing here is a redistribution of a public dataset. What is here is derived: split assignments, metadata
tables, and 64 featurized complexes.

| dataset | used for | size |
|---|---|---|
| PDBbind v2020 | the main benchmark, training and test | 19,443 complexes |
| CASF-2016 | scoring / ranking / docking / screening evaluation | 285 core complexes |
| SAIR | the scaling experiments and the case study | 1.05 M complexes, 152 GB |
| Runs N' Poses | calibration of the co-folding filter | 2,579 systems |
| PoseBench | external validation of the filter | PoseBusters subset, 130 targets |
| LIT-PCBA | zero-shot virtual screening | 15 targets, 5 used |
| ChEMBL / BindingDB | activity labels, the novelty projection | — |

## PDBbind v2020

From [pdbbind.org.cn](http://www.pdbbind.org.cn/), registration required. We use `refined` (5,316) and
`general_minus_refined` (14,127), which do not overlap. Labels are `pK = -log10(K / M)`; our index recomputes
pK from the raw affinity string and agrees with PDBbind's value to within 0.005.

Four properties of the distribution that affect how the index is built:

1. The `relation` column refers to the concentration K, not pK: `Ki < 10 nM` means pK > 8. Getting it
   backwards inverts every censored label. The loader keeps `relation == '='` only.
2. The 285 CASF-2016 core complexes are also in PDBbind (266 refined, 19 general) and must be out of
   training before any CASF evaluation. The split files here already exclude them.
3. The CASF distribution processes the same PDB IDs differently: different atom counts and protonation.
   Score CASF from the CASF files only.
4. PDBbind's `smiles` is a protonated form, about 40% carrying a formal charge. Structure prediction wants
   a neutralized form, and the two should not be mixed.

In addition, ~2,600 "ligands" are peptides (`ligand_name` like `(9-mer)`), 304 entries are NMR structures with no
resolution, 354 `uniprot_id` values are the placeholder `------` (we blank them, so that grouping by target
does not treat them as one protein), and 30 ligands have an SDF and a MOL2 that RDKit cannot read.

The index CSV the code expects: `pdb_id`, `pK`, `relation`, `affinity_type`, `uniprot_id`, `pocket_pdb`,
`ligand_sdf`, `ligand_mol2`. The last three are paths, absolute or relative to `data.root`.
`smoke/index.csv` is a 64-row example.

## Splits (`data/splits/`)

Two-column `pdb_id,split` CSVs. A complex absent from a file is unused, which is how the CASF-2016 core is
held out. Censored labels are dropped by the loader instead, so the complexes actually scored are slightly
fewer than `n_test`: 1,609 of the 1,652 test rows of `S3_cold_target_seq30cc/seed0`, the other 43 carrying
`<`, `>` or `~`.

| setting | held out | seeds |
|---|---|---|
| `S1_random` | nothing, random assignment | 5 |
| `S2_cold_ligand_scaffold` | Bemis–Murcko scaffolds | 5 |
| `S2b_cold_ligand_ecfp40` | ligands with ECFP Tanimoto > 0.4 to any training ligand | 5 |
| `S3_cold_target_seq30cc` | proteins in the same single-linkage 30%-identity component | 5 |
| `S3b_cold_target_mmseqs30` | proteins in the same MMseqs2 30% cluster | 5 |
| `S3c_cold_target_seq90` | proteins in the same 90% cluster (a leaky control) | 5 |
| `S3d_cold_target_strict` | 30% sequence and 0.6 pocket structural similarity | 5 |
| `S4_cold_both` | cold ligand and cold target at once | 5 |
| `S5_time` | everything deposited after a cut-off year | 1 |
| `S6_family` | one protein family at a time | 6 families |
| `S7_within_target` | ligands of targets that are also in training | 5 |
| `VS_litpcba_holdout` | the LIT-PCBA screening targets | 3 |

`seq30cc` and `mmseqs30` are different objects, not two names for one thing: `S3` uses connected components
of "identity ≥ 0.3 and alignment ≥ 50 residues", which is single-linkage and strictly more conservative than
`S3b`'s set-cover clusters. The paper reports both.

`summary.csv` has the sizes and the nearest-neighbor statistics per split and seed, including
`frac_prot_nn_ge0.3`, the fraction of test complexes with a training protein still at ≥ 30% identity. The
LIT-PCBA holdout is defined by its target list rather than by those statistics and has no row there.

Rebuilding the splits needs MMseqs2 (`easy-cluster -c 0.8 --cov-mode 0 -s 7.5`, and `easy-search` for the
all-vs-all identities) and Foldseek (`easy-search --exhaustive-search 1` on the 10 Å pockets, pocket
similarity `min(qtmscore, ttmscore)`, gated on alignment length ≥ 20 and `prob` ≥ 0.9). Those runs take a
few hours, which is why the assignments ship here.

## ESM-2 and Uni-Mol caches

Node inputs are concatenations: 38 geometric features + 1280 from ESM-2 650M
(`facebook/esm2_t33_650M_UR50D`) per residue, 43 + 512 from Uni-Mol (`unimolv1`) per ligand heavy atom, both
built offline into `<id>.pt` files.

The caches are far too large to ship, and the bundled smoke graphs already carry the features. Featurizing
new complexes does need them, so this is the layout the featurizer checks and will refuse to proceed without:

* ESM: a dict with `embedding` ([n_residues, 1280], every residue of the structure, not just the pocket) and
  `residue_keys`, `chain:resnum[icode]` in exactly the order `data.protein_atom_table` produces: waters
  dropped, metals kept, file order. Pocket rows are selected by index; a `residue_keys` mismatch is an error.
* Uni-Mol: a dict with `embedding` ([n_heavy_atoms, 512]), `symbols` and `coords`, taken from the pose, not
  from a conformer regenerated from SMILES. Symbols and coordinates are re-checked (1e-3 Å) against the
  ligand just read, and the aromatic virtual nodes are derived as the mean over their ring atoms.
* A missing cache file is zero-filled and warned about rather than raising, so the warnings matter: such a
  complex is scored without its pretrained features and is otherwise indistinguishable from a normal
  prediction.

Ligand perception depends on the RDKit version. The cached graphs here were built with RDKit 2024.03.6.

## SAIR metadata (`data/sair_metadata_accepted.csv.gz`)

[SAIR](https://huggingface.co/datasets/SandboxAQ/SAIR) is 1.05 M complexes with Boltz-1x co-folded
structures and IC50 labels; the structures are 152 GB and not here. This table is the metadata for the
118,351 complexes our reliability filter accepts (11.3% of SAIR), which is the subset the scaling
experiments treat as structurally trustworthy. One row per complex:

| column | |
|---|---|
| `entry_id`, `shard` | SAIR's identifiers; `shard` is where the structure lives in the release |
| `protein`, `family`, `n_res` | UniProt accession, protein family, residue count |
| `SMILES`, `ik14`, `n_lig_heavy` | ligand; `ik14` is the InChIKey skeleton layer, ignoring salt and stereochemistry |
| `pIC50`, `n_meas`, `pIC50_span`, `label_weight` | the label, how many measurements it aggregates, their spread, the reliability weight we train with |
| `nov_P_mm`, `nov_L_tc_ccd`, `nov_J_mm`, `J_tier` | novelty, as defined in `cofold_filter/` |
| `q_lig_plddt`, `q_p_success` | the quality axis and the calibrated probability the pose is within 2 Å |

The full sequences (79 MB on their own), the per-measurement statistics and the intermediate processing
columns are not included; neither is the state column, which is constant because every row is accepted.

`ik14` collapses salts and stereoisomers, so deduplicating on it is conservative. `pIC50` aggregates across
assays, and `pIC50_span` reports the spread, which is what flags an unreliable aggregate.

## Runs N' Poses, PoseBench, LIT-PCBA, ChEMBL

* **Runs N' Poses** ([paper](https://arxiv.org/abs/2502.03642), data on Zenodo) — 2,579 systems with crystal
  structures and predictions from several co-folding models. We re-folded all of it with Boltz-1 at two MSA
  seeds and calibrated the filter on those 5,244 predictions. The three cases in `cofold_filter/demo/` come
  from that re-fold.
* **PoseBench** — external validation of the filter, restricted to systems released after the Boltz-1
  training cut-off.
* **LIT-PCBA** — zero-shot screening on 5 targets; holdout split in `data/splits/VS_litpcba_holdout/`.
* **ChEMBL** (v37) and **BindingDB** — activity labels, and the novelty distribution the filter's precision
  is projected onto. `cofold_filter/reference/ccd_components.csv.gz` is derived from the PDB chemical
  component dictionary (`pdbx_initial_date`) and is included, since it is small and it makes ligand novelty
  computable without any download; 37,033 of its 51,389 components predate the 2021-09-30 cutoff.

`P_mm` and `J_mm` additionally need the chains of `pdb_seqres` filtered to entries released on or before the
cut-off (128,749 unique sequences in our run) and BioLiP's table of which biologically relevant ligands each
PDB chain binds. Both are public, neither is small.
