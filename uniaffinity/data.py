# structures -> pocket crop -> three graphs -> interaction fingerprint -> ESM-2 / Uni-Mol -> cache.
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
from torch import Tensor
from torch.utils.data import Dataset

from .ops import torch_load

RDLogger.DisableLog("rdApp.warning")

# one-hot vocabularies. metals stay in the pocket as nodes, waters do not.
AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
ELEMENTS = ["C", "N", "O", "S", "P", "F", "CL", "BR", "I", "B", "SE"]
ATOM_NAMES = ["N", "CA", "C", "O", "CB", "S", "P", "OTHER"]
METALS = {"ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU", "NA", "K", "CD", "HG"}
WATERS = {"HOH", "WAT", "DOD", "H2O"}
BACKBONE = {"N", "CA", "C", "O", "OXT"}
FEATURIZER_VERSION = 7   # bump this when the features change, or stale caches get reused silently

# residue: CA->N, CA->C, their normal, CA->sidechain centroid.  ligand: outward, ring normal, 0, 0
VECTOR_CHANNELS = 4

# Pharmacophore channels shared by protein and ligand atoms.
ATOM_TYPES = ["donor", "acceptor", "positive", "negative", "hydrophobic", "aromatic", "metal"]
# Geometric interaction-fingerprint channels (ProLIF/PLIP-lite).
INTERACTIONS = ["hbond", "hydrophobic", "salt_bridge", "halogen", "metal", "pi_stack", "cation_pi"]

# (resname, atom name) -> pharmacophore type
_PROT_DONORS = {"ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"}, "HIS": {"ND1", "NE2"},
                "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "TRP": {"NE1"}, "CYS": {"SG"}}
_PROT_ACCEPTORS = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"}, "GLN": {"OE1"},
                   "HIS": {"ND1", "NE2"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"}, "MET": {"SD"}}
_PROT_POSITIVE = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}}
_PROT_NEGATIVE = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
# Carbons bonded to N/O are not counted as hydrophobic.
_PROT_POLAR_CARBONS = {"SER": {"CB"}, "THR": {"CB"}, "ASP": {"CG"}, "GLU": {"CD"}, "ASN": {"CG"},
                       "GLN": {"CD"}, "LYS": {"CE"}, "ARG": {"CD", "CZ"}, "TYR": {"CZ"}, "HIS": {"CG", "CD2", "CE1"},
                       "TRP": {"CD1", "CE2"}, "PRO": {"CD"}}
_PROT_RINGS = {"PHE": [("CG", "CD1", "CE1", "CZ", "CE2", "CD2")],
               "TYR": [("CG", "CD1", "CE1", "CZ", "CE2", "CD2")],
               "TRP": [("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"), ("CG", "CD1", "NE1", "CE2", "CD2")],
               "HIS": [("CG", "ND1", "CE1", "NE2", "CD2")]}


def one_hot(value: Any, choices: list[Any]) -> list[float]:
    return [float(value == item) for item in choices] + [float(value not in choices)]


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32) if norm > 1e-8 else np.zeros(3, dtype=np.float32)


def ring_normal(points: np.ndarray) -> np.ndarray:
    """Newell normal of an ordered ring. Deterministic in atom order, hence rotation-equivariant
    (unlike an SVD normal, whose sign is arbitrary)."""
    centered = points - points.mean(axis=0)
    return unit(np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0))


def outward_vectors(pos: np.ndarray, edges: np.ndarray, n: int) -> np.ndarray:
    """-sum of unit bond vectors; dotted with a contact direction it proxies the H-bond angle."""
    out = np.zeros((n, 3), dtype=np.float32)
    if len(edges):
        src, dst = edges[:, 0], edges[:, 1]
        bond = pos[dst] - pos[src]
        bond /= np.linalg.norm(bond, axis=1, keepdims=True).clip(1e-8)
        np.add.at(out, src, -bond)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(norm > 1e-6, out / norm.clip(1e-8), 0.0).astype(np.float32)


def torsion(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """Return sin/cos of a signed dihedral; invariant to proper rotations and translations."""
    b0, b1, b2 = a - b, c - b, d - c
    b1 = unit(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-8 or np.linalg.norm(w) < 1e-8:
        return 0.0, 0.0
    angle = math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))
    return math.sin(angle), math.cos(angle)


# same as ops.GaussianRBF but numpy, for the precomputed edge features
def rbf(distance: np.ndarray, cutoff: float, bins: int) -> np.ndarray:
    centers = np.linspace(0.0, cutoff, bins, dtype=np.float32)
    gamma = (bins / cutoff) ** 2
    values = np.exp(-gamma * (distance[:, None] - centers[None]) ** 2)
    envelope = 0.5 * (np.cos(np.pi * np.minimum(distance, cutoff) / cutoff) + 1.0)
    return (values * envelope[:, None]).astype(np.float32)


def cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def radius_edges(pos: np.ndarray, cutoff: float, max_neighbors: int,
                 exclude: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    if len(pos) < 2:
        return []
    distances = cdist(pos, pos)
    result: list[tuple[int, int]] = []
    exclude = exclude or set()
    for dst in range(len(pos)):
        candidates = np.where((distances[:, dst] < cutoff) & (distances[:, dst] > 1e-6))[0]
        candidates = candidates[np.argsort(distances[candidates, dst])[:max_neighbors]]
        result.extend((int(src), dst) for src in candidates if (int(src), dst) not in exclude)
    return result


def bipartite_edges(contact: np.ndarray, cutoff: float, max_neighbors: int) -> list[tuple[int, int]]:
    """Protein->ligand edges selected on a [Np, Nl] contact-distance matrix."""
    result: list[tuple[int, int]] = []
    for ligand_idx in range(contact.shape[1]):
        candidates = np.where(contact[:, ligand_idx] < cutoff)[0]
        candidates = candidates[np.argsort(contact[candidates, ligand_idx])[:max_neighbors]]
        result.extend((int(p), ligand_idx) for p in candidates)
    if not result and contact.size:
        p, l = np.unravel_index(np.argmin(contact), contact.shape)
        result.append((int(p), int(l)))
    return result


def inter_geometry_cutoff(graph_cfg: dict[str, Any]) -> float:
    """RBF cutoff for the node-node (e.g. CA-ligand) distance on inter edges.
    With residue nodes and min-atom contacts, CA can be several Angstrom further than the contact."""
    return float(graph_cfg["inter_cutoff"]) + float(graph_cfg["residue_contact_margin"])


def edge_tensors(pos_a: np.ndarray, pos_b: np.ndarray, edges: list[tuple[int, int]],
                 cutoff: float, bins: int, edge_types: np.ndarray) -> tuple[Tensor, Tensor]:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, bins + edge_types.shape[1]), dtype=torch.float32)
    index = np.asarray(edges, dtype=np.int64).T
    distance = np.linalg.norm(pos_a[index[0]] - pos_b[index[1]], axis=-1).astype(np.float32)
    attr = np.concatenate([rbf(distance, cutoff, bins), edge_types.astype(np.float32)], axis=1)
    return torch.from_numpy(index), torch.from_numpy(attr)


# this is what gets pickled into the cache
@dataclass
class ComplexGraph:
    protein_x: Tensor
    protein_v: Tensor
    protein_pos: Tensor
    protein_edge_index: Tensor
    protein_edge_attr: Tensor
    ligand_x: Tensor
    ligand_v: Tensor
    ligand_pos: Tensor
    ligand_edge_index: Tensor
    ligand_edge_attr: Tensor
    inter_edge_index: Tensor
    inter_edge_attr: Tensor
    inter_types: Tensor
    inter_contact: Tensor  # [E] closest heavy-atom distance of the edge
    y: Tensor
    pdb_id: str


@dataclass
class ComplexBatch:
    protein_x: Tensor
    protein_v: Tensor
    protein_pos: Tensor
    protein_edge_index: Tensor
    protein_edge_attr: Tensor
    protein_batch: Tensor
    ligand_x: Tensor
    ligand_v: Tensor
    ligand_pos: Tensor
    ligand_edge_index: Tensor
    ligand_edge_attr: Tensor
    ligand_batch: Tensor
    inter_edge_index: Tensor
    inter_edge_attr: Tensor
    inter_types: Tensor
    inter_contact: Tensor
    y: Tensor
    pdb_id: list[str]
    num_graphs: int

    def to(self, device: torch.device | str) -> "ComplexBatch":
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if isinstance(value, Tensor) else value
        return ComplexBatch(**values)


def collate_graphs(items: list[ComplexGraph | None]) -> ComplexBatch | None:
    graphs = [item for item in items if item is not None]
    if not graphs:
        return None
    tensor_lists: dict[str, list[Tensor]] = {name: [] for name in [
        "protein_x", "protein_v", "protein_pos", "protein_edge_index", "protein_edge_attr",
        "ligand_x", "ligand_v", "ligand_pos", "ligand_edge_index", "ligand_edge_attr",
        "inter_edge_index", "inter_edge_attr", "inter_types", "inter_contact"]}
    protein_batch, ligand_batch, ys, ids = [], [], [], []
    p_offset = l_offset = 0
    for graph_idx, graph in enumerate(graphs):
        for name in tensor_lists:
            value = getattr(graph, name)
            if name == "protein_edge_index":
                value = value + p_offset
            elif name == "ligand_edge_index":
                value = value + l_offset
            elif name == "inter_edge_index":
                value = value + value.new_tensor([[p_offset], [l_offset]])
            tensor_lists[name].append(value)
        protein_batch.append(torch.full((graph.protein_x.shape[0],), graph_idx, dtype=torch.long))
        ligand_batch.append(torch.full((graph.ligand_x.shape[0],), graph_idx, dtype=torch.long))
        ys.append(graph.y.reshape(1))
        ids.append(graph.pdb_id)
        p_offset += graph.protein_x.shape[0]
        l_offset += graph.ligand_x.shape[0]
    values = {name: torch.cat(parts, dim=1 if "edge_index" in name else 0) for name, parts in tensor_lists.items()}
    return ComplexBatch(**values, protein_batch=torch.cat(protein_batch), ligand_batch=torch.cat(ligand_batch),
                        y=torch.cat(ys), pdb_id=ids, num_graphs=len(graphs))


# --------------------------------------------------------------------------------------------
# Protein
# --------------------------------------------------------------------------------------------

# ATOM/HETATM only, altloc in {'', A, 1}, no H/D. resid = (chain, resnum + icode).
def parse_pdb(path: Path) -> list[dict[str, Any]]:
    atoms = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A", "1"):
            continue
        element = (line[76:78].strip() or "".join(c for c in line[12:16] if c.isalpha())[:1]).upper()
        if element in ("H", "D"):
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
        except ValueError:
            continue
        atoms.append(dict(record=line[:6].strip(), name=line[12:16].strip().upper(), resname=line[17:20].strip().upper(),
                          chain=line[21].strip(), resid=(line[21].strip(), line[22:27].strip()),
                          element=element, pos=xyz,
                          bfactor=float(line[60:66]) if line[60:66].strip() else 0.0))
    if not atoms:
        raise ValueError(f"No heavy atoms parsed from {path}")
    return atoms


def residue_key(atom: dict[str, Any]) -> str:
    """Stable residue identity shared with the ESM alignment tool: chain:number+icode:resname."""
    chain, number = atom["resid"]
    return f"{chain}:{number}:{atom['resname']}"


# metals count as positive as well as metal
def protein_atom_types(atom: dict[str, Any]) -> list[float]:
    res, name, element = atom["resname"], atom["name"], atom["element"]
    metal = element in METALS
    donor = (name == "N" and res != "PRO") or name in _PROT_DONORS.get(res, ())
    acceptor = name in ("O", "OXT") or name in _PROT_ACCEPTORS.get(res, ())
    positive = metal or name in _PROT_POSITIVE.get(res, ())
    negative = name == "OXT" or name in _PROT_NEGATIVE.get(res, ())
    hydrophobic = (element == "C" and name not in ("C", "CA") and name not in _PROT_POLAR_CARBONS.get(res, ())) \
        or (res == "MET" and name == "SD")
    aromatic = any(name in ring for ring in _PROT_RINGS.get(res, ()))
    return [float(donor), float(acceptor), float(positive), float(negative), float(hydrophobic), float(aromatic), float(metal)]


@dataclass
class ProteinAtoms:
    pos: np.ndarray            # [Na, 3]
    types: np.ndarray          # [Na, len(ATOM_TYPES)]
    residue: np.ndarray        # [Na] residue index
    atoms: list[dict[str, Any]]
    residues: list[list[int]]           # atom indices per residue
    rings: list[tuple[int, list[int]]]  # (residue index, atom indices)


# waters out, metals in: a metal in the pocket is part of the binding site
def protein_atom_table(atoms: list[dict[str, Any]]) -> ProteinAtoms:
    selected = [a for a in atoms if a["resname"] not in WATERS]
    if not selected:
        raise ValueError("no protein atoms left after dropping waters")
    order: dict[tuple[Any, str], int] = {}
    residues: list[list[int]] = []
    residue_index = []
    for i, atom in enumerate(selected):
        key = (atom["resid"], atom["resname"])
        if key not in order:
            order[key] = len(order)
            residues.append([])
        residues[order[key]].append(i)
        residue_index.append(order[key])
    rings = []
    for r, members in enumerate(residues):
        named = {selected[i]["name"]: i for i in members}
        for ring in _PROT_RINGS.get(selected[members[0]]["resname"], ()):
            if all(name in named for name in ring):
                rings.append((r, [named[name] for name in ring]))
    return ProteinAtoms(pos=np.asarray([a["pos"] for a in selected], np.float32),
                        types=np.asarray([protein_atom_types(a) for a in selected], np.float32),
                        residue=np.asarray(residue_index, np.int64),
                        atoms=selected, residues=residues, rings=rings)


# no B-factor on purpose: it leaks crystal quality into a model that also scores predicted poses
def protein_residue_nodes(P: ProteinAtoms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, v, pos = [], [], []
    named_groups = []
    for members in P.residues:
        group = [P.atoms[i] for i in members]
        named = {a["name"]: a for a in group}
        named_groups.append(named)
        center = named["CA"]["pos"] if "CA" in named else np.mean([a["pos"] for a in group], axis=0)
        axis_n = unit(named["N"]["pos"] - center) if "N" in named else np.zeros(3, np.float32)
        axis_c = unit(named["C"]["pos"] - center) if "C" in named else np.zeros(3, np.float32)
        normal = unit(np.cross(axis_n, axis_c))
        side = [a["pos"] for a in group if a["name"] not in BACKBONE and a["element"] not in METALS]
        side_dir = unit(np.mean(side, axis=0) - center) if side and "CA" in named else np.zeros(3, np.float32)
        types = P.types[members].max(axis=0)  # residue-level pharmacophore presence
        feat = one_hot(group[0]["resname"], AA) + [float("N" in named), float("CA" in named), float("C" in named),
                                                    float(group[0]["record"] == "HETATM")] + types.tolist()
        feat += [0.0] * 6  # sin/cos of phi, psi, omega; filled in the next loop
        x.append(feat); v.append(np.stack([axis_n, axis_c, normal, side_dir])); pos.append(center)
    # only across a real peptide bond (same chain, C-N < 2 A); zero over chain breaks
    for i, named in enumerate(named_groups):
        angles = [(0.0, 0.0)] * 3
        chain = P.atoms[P.residues[i][0]]["chain"]
        same_prev = i > 0 and P.atoms[P.residues[i - 1][0]]["chain"] == chain
        same_next = i + 1 < len(named_groups) and P.atoms[P.residues[i + 1][0]]["chain"] == chain
        if same_prev and all(k in named_groups[i - 1] for k in ("CA", "C")) and all(k in named for k in ("N", "CA", "C")):
            prev = named_groups[i - 1]
            if np.linalg.norm(prev["C"]["pos"] - named["N"]["pos"]) < 2.0:
                angles[0] = torsion(prev["C"]["pos"], named["N"]["pos"], named["CA"]["pos"], named["C"]["pos"])
                angles[2] = torsion(prev["CA"]["pos"], prev["C"]["pos"], named["N"]["pos"], named["CA"]["pos"])
        if same_next and all(k in named for k in ("N", "CA", "C")) and "N" in named_groups[i + 1]:
            if np.linalg.norm(named["C"]["pos"] - named_groups[i + 1]["N"]["pos"]) < 2.0:
                angles[1] = torsion(named["N"]["pos"], named["CA"]["pos"], named["C"]["pos"], named_groups[i + 1]["N"]["pos"])
        x[i][-6:] = [component for pair in angles for component in pair]
    return np.asarray(x, np.float32), np.asarray(v, np.float32), np.asarray(pos, np.float32)


# --------------------------------------------------------------------------------------------
# Ligand
# --------------------------------------------------------------------------------------------

_FEATURE_FACTORY = None


def feature_factory():
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    return _FEATURE_FACTORY


# heavy atoms + one virtual node per aromatic ring. types has one column more than the protein's: halogen.
@dataclass
class LigandGraph:
    x: np.ndarray
    v: np.ndarray
    pos: np.ndarray            # nodes: heavy atoms then aromatic virtual nodes
    n_atoms: int               # number of real heavy atoms
    types: np.ndarray          # [n_atoms, len(ATOM_TYPES)] + halogen column
    bonds: list[tuple[int, int]]
    virtual_edges: list[tuple[int, int]]
    rings: list[tuple[int, ...]]           # aromatic rings (heavy-atom indices)
    ring_nodes: list[int | None]           # virtual-node index per ring (None if disabled)
    symbols: list[str]                     # RDKit element symbols of the heavy atoms (node order)
    source: str                            # file actually parsed (SDF, or MOL2 after fallback)


def load_ligand(path: Path) -> Chem.Mol:
    if path.suffix.lower() == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    else:
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse {path}")
    return mol


def ligand_graph(path: Path) -> LigandGraph:
    mol = load_ligand(path)
    conf = mol.GetConformer()
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    mapping = {old: new for new, old in enumerate(heavy)}
    family: dict[str, set[int]] = {}
    for feature in feature_factory().GetFeaturesForMol(mol):
        family.setdefault(feature.GetFamily(), set()).update(feature.GetAtomIds())
    x, pos, types = [], [], []
    for old in heavy:
        atom = mol.GetAtomWithIdx(old)
        symbol = atom.GetSymbol().upper()
        heavy_degree = sum(n.GetAtomicNum() > 1 for n in atom.GetNeighbors())
        n_h = atom.GetTotalNumHs(includeNeighbors=True)  # explicit H atoms are neighbors, not implicit Hs
        polar_neighbor = any(n.GetSymbol() in ("N", "O") for n in atom.GetNeighbors())
        t = [old in family.get("Donor", ()), old in family.get("Acceptor", ()),
             old in family.get("PosIonizable", ()) or atom.GetFormalCharge() > 0,
             old in family.get("NegIonizable", ()) or atom.GetFormalCharge() < 0,
             # hydrophobic: C/S/halogen, no N/O neighbor, uncharged -- or RDKit says Hydrophobe
             (symbol in ("C", "CL", "BR", "I", "S") and not polar_neighbor and atom.GetFormalCharge() == 0)
             or old in family.get("Hydrophobe", ()),
             atom.GetIsAromatic(), False, symbol in ("CL", "BR", "I")]
        types.append([float(value) for value in t])
        feat = one_hot(symbol, ELEMENTS) + one_hot(min(heavy_degree, 5), list(range(6)))
        feat += one_hot(min(n_h, 3), [0, 1, 2, 3])
        feat += one_hot(max(-2, min(2, atom.GetFormalCharge())), [-2, -1, 0, 1, 2])
        feat += one_hot(str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D"])
        feat += [float(atom.IsInRing()), *[float(value) for value in t[:6]], 0.0]  # last: virtual flag
        p = conf.GetAtomPosition(old)
        x.append(feat); pos.append([p.x, p.y, p.z])
    n_atoms = len(x)
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in mapping and bond.GetEndAtomIdx() in mapping:
            a, b = mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
            bonds.extend([(a, b), (b, a)])
    pos_arr = np.asarray(pos, dtype=np.float32)
    v = np.zeros((n_atoms, VECTOR_CHANNELS, 3), np.float32)
    v[:, 0] = outward_vectors(pos_arr, np.asarray(bonds, np.int64).reshape(-1, 2), n_atoms)
    rings: list[tuple[int, ...]] = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring) and all(i in mapping for i in ring):
            rings.append(tuple(mapping[i] for i in ring))  # AtomRings is in ring-walk order
    ring_nodes: list[int | None] = []
    virtual_edges, extra_x, extra_pos, extra_v = [], [], [], []
    for ring in rings:
        normal = ring_normal(pos_arr[list(ring)])
        v[list(ring), 1] = normal
        ring_idx = n_atoms + len(extra_x)
        ring_nodes.append(ring_idx)
        ring_feat = np.asarray(x, np.float32)[list(ring)].mean(axis=0)
        ring_feat[-1] = 1.0
        extra_x.append(ring_feat.tolist()); extra_pos.append(pos_arr[list(ring)].mean(axis=0))
        ev = np.zeros((VECTOR_CHANNELS, 3), np.float32); ev[1] = normal; extra_v.append(ev)
        for atom_idx in ring:
            virtual_edges.extend([(ring_idx, atom_idx), (atom_idx, ring_idx)])
    if extra_x:
        x = x + extra_x
        pos_arr = np.concatenate([pos_arr, np.asarray(extra_pos, np.float32)])
        v = np.concatenate([v, np.asarray(extra_v, np.float32)])
    return LigandGraph(x=np.asarray(x, np.float32), v=v, pos=pos_arr.astype(np.float32), n_atoms=n_atoms,
                       types=np.asarray(types, np.float32).reshape(-1, len(ATOM_TYPES) + 1), bonds=bonds,
                       virtual_edges=virtual_edges, rings=rings, ring_nodes=ring_nodes,
                       symbols=[mol.GetAtomWithIdx(i).GetSymbol() for i in heavy], source=str(path))


# --------------------------------------------------------------------------------------------
# Interactions
# --------------------------------------------------------------------------------------------

def geometric_interactions(P: ProteinAtoms, atom_to_node: np.ndarray, n_p: int, L: LigandGraph,
                           cfg: dict[str, Any]) -> np.ndarray:
    """[Np, Nl, 7] interaction fingerprint, decided on atom pairs and max-pooled onto the nodes."""
    n_l = len(L.pos)
    out = np.zeros((n_p, n_l, len(INTERACTIONS)), np.float32)
    kept = atom_to_node >= 0
    if not kept.any() or L.n_atoms == 0:
        return out
    p_idx = np.where(kept)[0]
    p_pos, p_t, p_node = P.pos[p_idx], P.types[p_idx], atom_to_node[p_idx]
    l_pos, l_t = L.pos[:L.n_atoms], L.types
    D = cdist(p_pos, l_pos)
    don, acc, pos_, neg, hyd, _, metal = (p_t[:, k][:, None] > 0 for k in range(7))
    ldon, lacc, lpos, lneg, lhyd, _, _, lhal = (l_t[:, k][None, :] > 0 for k in range(8))
    hb = float(cfg.get("hbond_cutoff", 3.5))
    pairs = {
        "hbond": ((don & lacc) | (acc & ldon)) & (D <= hb),
        "hydrophobic": hyd & lhyd & (D <= float(cfg.get("hydrophobic_cutoff", 4.0))),
        # metals excluded from salt bridges, they have their own channel
        "salt_bridge": ((pos_ & ~metal & lneg) | (neg & lpos)) & (D <= float(cfg.get("salt_bridge_cutoff", 4.0))),
        "halogen": acc & lhal & (D <= hb),
        "metal": metal & (lacc | lneg) & (D <= float(cfg.get("metal_cutoff", 2.8))),
    }
    for k, name in enumerate(INTERACTIONS[:5]):
        a, l = np.nonzero(pairs[name])
        out[p_node[a], l, k] = 1.0

    def ligand_targets(ring_id: int) -> list[int]:
        node = L.ring_nodes[ring_id]
        return [node] if node is not None else list(L.rings[ring_id])

    p_rings = [(np.asarray(atoms), P.pos[atoms].mean(axis=0), ring_normal(P.pos[atoms])) for _, atoms in P.rings
               if (atom_to_node[atoms] >= 0).all()]
    l_rings = [(L.pos[list(r)].mean(axis=0), ring_normal(L.pos[list(r)])) for r in L.rings]
    stack_cut, cation_cut = float(cfg.get("pi_stack_cutoff", 5.5)), float(cfg.get("cation_pi_cutoff", 6.0))
    for atoms, pc, pn in p_rings:
        nodes = np.unique(atom_to_node[atoms])
        for r, (lc, ln) in enumerate(l_rings):
            d = np.linalg.norm(pc - lc)
            angle = math.degrees(math.acos(min(1.0, abs(float(np.dot(pn, ln))))))
            # parallel <= 30 deg at pi_stack_cutoff; T-shaped >= 60 deg gets 1 A more
            if (d <= stack_cut and angle <= 30.0) or (d <= stack_cut + 1.0 and angle >= 60.0):  # face-to-face / T-shaped
                for node in nodes:
                    out[node, ligand_targets(r), 5] = 1.0
        # protein ring <-> ligand cation
        cations = np.where((l_t[:, 2] > 0) & (np.linalg.norm(l_pos - pc, axis=1) <= cation_cut))[0]
        for node in nodes:
            out[node, cations, 6] = 1.0
    # protein cation <-> ligand ring
    cation_atoms = np.where((p_t[:, 2] > 0) & (p_t[:, 6] == 0))[0]
    for r, (lc, _) in enumerate(l_rings):
        close = cation_atoms[np.linalg.norm(p_pos[cation_atoms] - lc, axis=1) <= cation_cut]
        for a in close:
            out[p_node[a], ligand_targets(r), 6] = 1.0
    return out


class ComplexFeaturizer:
    def __init__(self, config: dict[str, Any], data_root: Path):
        self.cfg, self.g, self.root = config, config["graph"], data_root
        base = self.base = Path(__file__).resolve().parents[1]
        extras = [Path(path) for path in config["data"]["structure_fallback_roots"]]
        self.roots = [data_root] + [path if path.is_absolute() else (base / path).resolve() for path in extras]

    def resolve_path(self, relative: str | Path) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        for root in self.roots:
            candidate = root / path
            if candidate.exists():
                return candidate
        return self.roots[0] / path

    def project_path(self, value: str | Path) -> Path:
        """Config paths (cache dirs) are relative to the project directory, not the shell's cwd."""
        path = Path(value)
        return path if path.is_absolute() else (self.base / path).resolve()

    # raises on any mismatch rather than shifting features onto the wrong nodes, which would be invisible
    # in the loss curve. Cache layout in DATA.md.
    def _pretrained(self, pdb_id: str, kind: str, n: int, node_to_source: np.ndarray | None = None,
                    virtual_groups: list[tuple[int, ...]] | None = None,
                    expected_keys: list[str] | None = None, ligand: "LigandGraph | None" = None) -> np.ndarray:
        cfg = self.cfg["pretrained"][kind]
        dim = int(cfg["dimension"])
        if not cfg["enabled"] or dim == 0:
            return np.zeros((n, 0), dtype=np.float32)
        path = self.project_path(cfg["cache_dir"]) / f"{pdb_id}.pt" if cfg["cache_dir"] else Path(f"<no {kind} cache_dir>")
        if path.exists():
            value = torch_load(path, map_location="cpu", weights_only=False)
            if isinstance(value, dict):
                if expected_keys is not None and "residue_keys" in value and list(value["residue_keys"]) != expected_keys:
                    raise ValueError(f"{kind} embedding {path} was built for a different residue list; "
                                     f"rebuild it from the current structure")
                if ligand is not None and "symbols" in value:
                    same = list(value["symbols"]) == ligand.symbols and np.allclose(
                        np.asarray(value["coords"], np.float32), ligand.pos[:ligand.n_atoms], atol=1e-3)
                    if not same:
                        raise ValueError(f"{kind} embedding {path} was built from a different ligand file, pose or "
                                         f"atom order (expected {ligand.source}); rebuild it")
                value = value["embedding"]
            value = torch.as_tensor(value, dtype=torch.float32).numpy()
            if value.ndim != 2 or value.shape[1] != dim:
                raise ValueError(f"{kind} embedding {path} has {value.shape}; second dimension must be {dim}")
            if node_to_source is not None and len(node_to_source) == n and node_to_source.max(initial=-1) < value.shape[0]:
                return value[node_to_source]
            if value.shape[0] == n:
                return value
            if virtual_groups and value.shape[0] + len(virtual_groups) == n:
                virtual = np.stack([value[list(group)].mean(axis=0) for group in virtual_groups])
                return np.concatenate([value, virtual], axis=0)
            raise ValueError(f"{kind} embedding {path} has {value.shape[0]} nodes, cannot align to {n}")
        # TODO this should probably raise rather than warn
        print(f"[data] warning: no {kind} features for {pdb_id} ({path}); using zeros", flush=True)
        return np.zeros((n, dim), dtype=np.float32)

    # SDF first; ~30 PDBbind SDFs are unreadable by RDKit while their MOL2 is fine
    def ligand(self, row: dict[str, Any]) -> LigandGraph:
        """The ligand exactly as the featurizer sees it, including the SDF->MOL2 fallback."""
        ligand_path = self.resolve_path(str(row[self.cfg["data"]["ligand_column"]]))
        try:
            return ligand_graph(ligand_path)
        except (ValueError, OSError):
            fallback_column = self.cfg["data"].get("ligand_fallback_column")
            fallback = row.get(fallback_column) if fallback_column else None
            if not fallback or pd.isna(fallback):
                raise
            return ligand_graph(self.resolve_path(str(fallback)))

    def __call__(self, row: dict[str, Any]) -> ComplexGraph:
        g = self.g
        pdb_id = str(row["pdb_id"])
        protein_path = self.resolve_path(str(row[self.cfg["data"]["structure_column"]]))
        L = self.ligand(row)
        P = protein_atom_table(parse_pdb(protein_path))

        # any heavy atom within pocket_cutoff, not the CA: cropping on CA drops long side chains
        atom_lig = cdist(P.pos, L.pos[:L.n_atoms])  # [Na, Nl_atoms]
        atom_nearest = atom_lig.min(axis=1)
        px, pv, ppos = protein_residue_nodes(P)
        node_nearest = np.asarray([atom_nearest[m].min() for m in P.residues])
        node_atoms = P.residues
        node_residue = np.arange(len(P.residues), dtype=np.int64)
        # keep the nearest max_protein_nodes, then restore sequence order
        keep = np.where(node_nearest <= g["pocket_cutoff"])[0]
        if not len(keep):
            keep = np.argsort(node_nearest)[:1]
        keep = np.sort(keep[np.argsort(node_nearest[keep])[: g["max_protein_nodes"]]])  # keep sequence order
        px, pv, ppos = px[keep], pv[keep], ppos[keep]
        node_residue = node_residue[keep]
        atom_to_node = np.full(len(P.atoms), -1, np.int64)
        for new, old in enumerate(keep):
            atom_to_node[node_atoms[old]] = new
        n_p, n_l = len(keep), len(L.pos)

        # pretrained features go on as the last columns of x
        residue_keys = [residue_key(P.atoms[members[0]]) for members in P.residues]
        # optional row["esm_id"]: many poses of one protein share its ESM file (CASF screening); default pdb_id
        esm_id = str(row.get("esm_id") or pdb_id)
        px = np.concatenate([px, self._pretrained(esm_id, "esm", n_p, node_to_source=node_residue,
                                                  expected_keys=residue_keys)], axis=1)
        lx = np.concatenate([L.x, self._pretrained(pdb_id, "unimol", n_l,
                                                   virtual_groups=L.rings,
                                                   ligand=L)], axis=1)

        bins = g["rbf_bins"]
        # ---- protein intra graph. edge type = [sequence neighbors?, 1]
        pcut, pmax = g["protein_cutoff"], g["protein_max_neighbors"]
        pedges = radius_edges(ppos, pcut, pmax)
        ptypes = np.asarray([[float(abs(int(node_residue[i]) - int(node_residue[j])) == 1), 1.0]
                             for i, j in pedges], np.float32).reshape(-1, 2)
        pei, pea = edge_tensors(ppos, ppos, pedges, pcut, bins, ptypes)

        # ---- ligand intra graph
        bond_set = set(L.bonds)
        virtual_set = set(L.virtual_edges)
        ledges = list(L.bonds) + list(L.virtual_edges)
        ledges += radius_edges(L.pos, g["ligand_spatial_cutoff"], g["ligand_max_neighbors"], bond_set | virtual_set)
        ltypes = np.asarray([[float(e in bond_set), float(e in virtual_set), float(e not in bond_set and e not in virtual_set)]
                             for e in ledges], np.float32).reshape(-1, 3)
        # max(): a covalent bond longer than the spatial cutoff must still land inside the RBF support
        lcut = max(float(g["ligand_spatial_cutoff"]), float(g["ligand_rbf_cutoff"]))
        lei, lea = edge_tensors(L.pos, L.pos, ledges, lcut, bins, ltypes)

        # ---- inter graph
        # node_lig[p, l] = shortest heavy-atom distance from residue p to ligand node l
        node_lig = np.full((n_p, n_l), np.inf, np.float32)
        kept_atoms = np.where(atom_to_node >= 0)[0]
        np.minimum.at(node_lig[:, :L.n_atoms], atom_to_node[kept_atoms], atom_lig[kept_atoms].astype(np.float32))
        for r, node in enumerate(L.ring_nodes):
            if node is not None:
                d = np.linalg.norm(P.pos[kept_atoms] - L.pos[node], axis=1).astype(np.float32)
                np.minimum.at(node_lig[:, node], atom_to_node[kept_atoms], d)
        iedges = bipartite_edges(node_lig, g["inter_cutoff"], g["inter_max_neighbors"])
        contact = np.asarray([node_lig[p, l] for p, l in iedges], np.float32)

        dense = geometric_interactions(P, atom_to_node, n_p, L, g)
        itypes = np.asarray([dense[p, l] for p, l in iedges], np.float32).reshape(-1, len(INTERACTIONS))
        # [RBF(node dist), RBF(contact dist), IFP]. model.py also uses inter_contact / inter_types directly.
        iei, iea = edge_tensors(ppos, L.pos, iedges, inter_geometry_cutoff(g), bins, itypes)
        iea = torch.cat([iea[:, :bins], torch.from_numpy(rbf(contact, float(g["inter_cutoff"]), bins)), iea[:, bins:]], dim=1)
        return ComplexGraph(
            protein_x=torch.from_numpy(px), protein_v=torch.from_numpy(pv), protein_pos=torch.from_numpy(ppos),
            protein_edge_index=pei, protein_edge_attr=pea,
            ligand_x=torch.from_numpy(lx), ligand_v=torch.from_numpy(L.v), ligand_pos=torch.from_numpy(L.pos),
            ligand_edge_index=lei, ligand_edge_attr=lea,
            inter_edge_index=iei, inter_edge_attr=iea, inter_types=torch.from_numpy(itypes),
            inter_contact=torch.from_numpy(contact),
            y=torch.tensor(float(row[self.cfg["data"]["label_column"]]), dtype=torch.float32),
            pdb_id=pdb_id)


# Name says PDBbind, but any index csv with the right columns works.
class PDBBindDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, config: dict[str, Any], data_root: Path):
        self.records = frame.to_dict("records")
        self.cfg, self.root = config, data_root
        # cache_dir stays out of the digest -- where the files live does not change what is in them
        pretrained = {kind: {k: v for k, v in block.items() if k != "cache_dir"}
                      for kind, block in config["pretrained"].items()}
        digest = hashlib.sha1(json.dumps({"version": FEATURIZER_VERSION, "graph": config["graph"],
                                          "pretrained": pretrained}, sort_keys=True).encode()).hexdigest()[:12]
        cache = Path(config["data"]["cache_dir"])
        self.cache = (cache if cache.is_absolute() else Path(__file__).resolve().parents[1] / cache) / digest
        self.cache.mkdir(parents=True, exist_ok=True)
        self.featurizer = ComplexFeaturizer(config, data_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ComplexGraph | None:
        row = self.records[index]
        path = self.cache / f"{row['pdb_id']}.pt"
        try:
            if path.exists():
                try:
                    return torch_load(path, map_location="cpu", weights_only=False)
                except Exception:  # truncated / half-written file: rebuild it instead of dropping the sample
                    path.unlink(missing_ok=True)
            graph = self.featurizer(row)
            # tmp + rename: several jobs build the same cache at once and a half-written .pt is unreadable
            tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
            torch.save(graph, tmp)
            tmp.replace(path)
            return graph
        except Exception as exc:
            if self.cfg["data"]["on_error"] == "skip":
                print(f"[data] skipping {row['pdb_id']}: {exc}", flush=True)
                return None
            raise


# anything absent from split_file is unused; that is how the splits drop the CASF-2016 core set
def make_splits(config: dict[str, Any], base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    data_cfg = config["data"]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base_dir / path).resolve()

    root, index_root = resolve(data_cfg["root"]), resolve(data_cfg["index_root"])
    frame = pd.read_csv(index_root / data_cfg["csv"])
    frame = frame.dropna(subset=[data_cfg["label_column"], data_cfg["structure_column"], data_cfg["ligand_column"]])
    if "relation" in frame:
        frame = frame[frame["relation"] == "="]
    assign = pd.read_csv(resolve(data_cfg["split_file"])).set_index("pdb_id")["split"]
    frame = frame.assign(_split=frame["pdb_id"].map(assign))
    test, valid, train = (frame[frame._split == s].drop(columns="_split").copy() for s in ("test", "valid", "train"))
    # shuffle once with the run seed; the index csv arrives sorted
    train = train.iloc[np.random.default_rng(config["experiment"]["seed"]).permutation(len(train))]
    return (train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True), root)
