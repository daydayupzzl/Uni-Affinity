# protein graph -> ligand graph -> inter layers -> pair repr per contact -> attention pool -> MLP head.
from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from .data import inter_geometry_cutoff
from .ops import GaussianRBF, safe_norm, scatter_softmax, scatter_sum


class VectorLinear(nn.Module):
    """Linear mixing over vector channels; never mixes xyz axes and has no bias (equivariance)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, v: Tensor) -> Tensor:
        return torch.einsum("oi,nic->noc", self.weight, v)


class ScalarVectorLayer(nn.Module):
    """PaiNN/GVP-style message passing on (s, V) states, with the V.r angular terms in the message."""

    def __init__(self, scalar_dim: int, vector_dim: int, rbf_dim: int, dropout: float):
        super().__init__()
        self.scalar_dim, self.vector_dim = scalar_dim, vector_dim
        # LazyLinear: edge_attr width differs between the protein / ligand / inter graphs
        self.edge_proj = nn.LazyLinear(rbf_dim)
        in_dim = 2 * scalar_dim + rbf_dim + 2 * vector_dim
        self.message = nn.Sequential(
            nn.Linear(in_dim, 2 * scalar_dim + 2 * vector_dim),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(2 * scalar_dim + 2 * vector_dim, scalar_dim + 2 * vector_dim),
        )
        self.v_source = VectorLinear(vector_dim, vector_dim)
        self.v_update = VectorLinear(vector_dim, vector_dim)
        self.scalar_update = nn.Sequential(
            nn.Linear(2 * scalar_dim + vector_dim, 2 * scalar_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * scalar_dim, scalar_dim),
        )
        self.vector_gate = nn.Linear(scalar_dim, vector_dim)
        self.norm = nn.LayerNorm(scalar_dim)

    def forward(self, s: Tensor, v: Tensor, pos: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        if edge_index.numel() == 0:
            return s, v
        src, dst = edge_index
        delta = pos[src] - pos[dst]
        direction = delta / safe_norm(delta, dim=-1, keepdim=True)
        parts = [s[src], s[dst], self.edge_proj(edge_attr),
                 (v[src] * direction[:, None, :]).sum(-1), (v[dst] * direction[:, None, :]).sum(-1)]
        raw = self.message(torch.cat(parts, dim=-1))
        ms, gate_v, gate_r = torch.split(raw, [self.scalar_dim, self.vector_dim, self.vector_dim], dim=-1)
        mv = gate_v[..., None] * self.v_source(v[src]) + gate_r[..., None] * direction[:, None, :]
        agg_s = scatter_sum(ms, dst, s.shape[0])
        mixed_v = self.v_update(scatter_sum(mv, dst, s.shape[0]))
        s = self.norm(s + self.scalar_update(torch.cat([s, agg_s, safe_norm(mixed_v, dim=-1)], dim=-1)))
        v = v + torch.sigmoid(self.vector_gate(s)).unsqueeze(-1) * mixed_v
        # per-channel RMS norm; pure rescaling, so still equivariant
        v = v / torch.sqrt(v.square().sum(-1).mean(-1, keepdim=True).unsqueeze(-1) + 1e-6)
        return s, v


class GraphEncoder(nn.Module):
    """Input embedding + intra-graph scalar-vector layers."""

    def __init__(self, cfg: dict[str, Any], n_layers: int):
        super().__init__()
        d_s, d_v = cfg["scalar_dim"], cfg["vector_dim"]
        self.scalar_in = nn.LazyLinear(d_s)
        self.vector_in = VectorLinear(cfg["input_vector_channels"], d_v)
        self.layers = nn.ModuleList([
            ScalarVectorLayer(d_s, d_v, cfg["rbf_dim"], cfg["dropout"]) for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, v: Tensor, pos: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        s, v = self.scalar_in(x), self.vector_in(v)
        for layer in self.layers:
            s, v = layer(s, v, pos, edge_index, edge_attr)
        return s, v


class PairContext(nn.Module):
    """Linear-cost attention context over existing inter edges of each complex."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.update = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, pair: Tensor, pair_batch: Tensor, n_graphs: int) -> Tensor:
        weight = scatter_softmax(self.score(pair), pair_batch, n_graphs)
        context = scatter_sum(weight * pair, pair_batch, n_graphs)
        return self.norm(pair + self.update(torch.cat([pair, context[pair_batch]], dim=-1)))


class UniAffinity(nn.Module):
    """Intra/inter scalar-vector network for pose-based affinity prediction."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model = cfg["model"]
        d_s, d_v = model["scalar_dim"], model["vector_dim"]
        self.cfg = cfg
        self.protein_encoder = GraphEncoder(model, model["protein_intra_layers"])
        self.ligand_encoder = GraphEncoder(model, model["ligand_intra_layers"])
        self.inter_layers = nn.ModuleList([
            ScalarVectorLayer(d_s, d_v, model["rbf_dim"], model["dropout"]) for _ in range(model["inter_layers"])
        ])
        # two RBF banks: CA-ligand distance, and the shortest heavy-atom contact distance
        self.inter_rbf = GaussianRBF(model["rbf_dim"], inter_geometry_cutoff(cfg["graph"]))
        self.contact_rbf = GaussianRBF(model["rbf_dim"], float(cfg["graph"]["inter_cutoff"]))
        self.pair_project = nn.Sequential(nn.LazyLinear(d_s), nn.SiLU(), nn.Linear(d_s, d_s))
        self.pair_context = PairContext(d_s, model["dropout"])
        self.pair_attention = nn.Sequential(nn.Linear(d_s, d_s), nn.Tanh(), nn.Linear(d_s, 1))
        self.pair_energy = nn.Sequential(nn.Linear(d_s, d_s), nn.SiLU(), nn.Linear(d_s, d_s))
        # pooled contacts only. A mean-protein / mean-ligand path is how a model ends up answering from
        # target identity alone -- don't add one back without re-running S3/S4.
        self.head = nn.Sequential(
            nn.Linear(d_s, 2 * d_s), nn.SiLU(), nn.Dropout(model["dropout"]),
            nn.Linear(2 * d_s, d_s), nn.SiLU(), nn.Linear(d_s, 1),
        )

    def _augment(self, batch: Any) -> Any:
        """coord_noise + contact_dropout, training only."""
        m = self.cfg["model"]
        sigma, p_drop = float(m["coord_noise"]), float(m["contact_dropout"])
        if not self.training or (sigma <= 0 and p_drop <= 0):
            return batch
        batch = copy.copy(batch)  # shallow: only the replaced tensors differ
        if sigma > 0:
            batch.protein_pos = batch.protein_pos + sigma * torch.randn_like(batch.protein_pos)
            batch.ligand_pos = batch.ligand_pos + sigma * torch.randn_like(batch.ligand_pos)
        if p_drop > 0:
            keep = torch.rand(batch.inter_edge_index.shape[1], device=batch.inter_edge_index.device) >= p_drop
            batch.inter_edge_index = batch.inter_edge_index[:, keep]
            batch.inter_edge_attr, batch.inter_types = batch.inter_edge_attr[keep], batch.inter_types[keep]
            batch.inter_contact = batch.inter_contact[keep]
        return batch

    def forward(self, batch: Any, return_attention: bool = False) -> dict[str, Tensor]:
        # Np = protein nodes in the batch, Nl = ligand nodes, E = contacts, B = complexes
        batch = self._augment(batch)
        ns_p = batch.protein_x.shape[0]
        pos = torch.cat([batch.protein_pos, batch.ligand_pos])
        # add reverse edges; last attr column flags direction (0 = p->l, 1 = l->p)
        p_idx, l_idx = batch.inter_edge_index
        forward = torch.stack([p_idx, l_idx + ns_p])
        inter_index = torch.cat([forward, forward.flip(0)], dim=1)
        flag = batch.inter_edge_attr.new_zeros((p_idx.numel(), 1))
        inter_attr = torch.cat([torch.cat([batch.inter_edge_attr, flag], -1),
                                torch.cat([batch.inter_edge_attr, flag + 1.0], -1)], dim=0)

        # ps = [Np, scalar_dim], pv = [Np, vector_dim, 3]
        ps, pv = self.protein_encoder(batch.protein_x, batch.protein_v, batch.protein_pos,
                                      batch.protein_edge_index, batch.protein_edge_attr)
        ls, lv = self.ligand_encoder(batch.ligand_x, batch.ligand_v, batch.ligand_pos,
                                     batch.ligand_edge_index, batch.ligand_edge_attr)
        for layer in self.inter_layers:
            s, v = layer(torch.cat([ps, ls]), torch.cat([pv, lv]), pos, inter_index, inter_attr)
            ps, pv, ls, lv = s[:ns_p], v[:ns_p], s[ns_p:], v[ns_p:]

        # ---- pair representation, on real contacts only (never an Np x Nl matrix)
        delta = batch.ligand_pos[l_idx] - batch.protein_pos[p_idx]   # [E, 3], protein -> ligand
        distance = safe_norm(delta, dim=-1)
        direction = delta / distance[:, None]
        # p_dir, l_dir = [E, vector_dim] : each side's vectors against the contact axis (H-bond angle etc.)
        # pl_dot        = [E, vector_dim] : relative orientation (ring normal vs ring normal -> pi-stacking)
        p_dir = (pv[p_idx] * direction[:, None, :]).sum(-1)
        l_dir = (lv[l_idx] * direction[:, None, :]).sum(-1)
        pl_dot = (pv[p_idx] * lv[l_idx]).sum(-1)
        geom = self.inter_rbf(distance), self.contact_rbf(batch.inter_contact)
        pair_raw = torch.cat([ps[p_idx], ls[l_idx], p_dir, l_dir, pl_dot, *geom, batch.inter_types], dim=-1)
        pair = self.pair_project(pair_raw)                           # [E, scalar_dim]
        pair_batch = batch.protein_batch[p_idx]                      # [E], which complex each contact is in
        n_graphs = batch.num_graphs
        pair = self.pair_context(pair, pair_batch, n_graphs)

        # ---- readout
        attention = scatter_softmax(self.pair_attention(pair), pair_batch, n_graphs)   # [E, 1]
        head_in = scatter_sum(attention * self.pair_energy(pair), pair_batch, n_graphs)  # [B, scalar_dim]
        affinity = self.head(head_in).squeeze(-1)
        out = {"affinity": affinity, "pair_energy": head_in, "head_in": head_in}
        if return_attention:
            out.update(attention=attention.squeeze(-1), pair_batch=pair_batch,
                       inter_edge_index=batch.inter_edge_index)
        return out
