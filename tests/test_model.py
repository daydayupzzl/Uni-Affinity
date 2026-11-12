"""python -m unittest discover -s tests -v

Runs on the graphs bundled in smoke/, so nothing has to be downloaded.
"""
from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

import torch

from uniaffinity.data import INTERACTIONS, ComplexFeaturizer, collate_graphs
from uniaffinity.losses import affinity_loss
from uniaffinity.model import ScalarVectorLayer, UniAffinity
from uniaffinity.ops import torch_load
from uniaffinity.trainer import load_weights

BASE = Path(__file__).resolve().parents[1]
SMOKE = BASE / "smoke"


def random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A proper rotation: QR can return a reflection, which a chiral pocket is not invariant to."""
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = torch_load(SMOKE / "checkpoint" / "best.pt", map_location="cpu", weights_only=False)
        cls.config = checkpoint["config"]
        cache = next((SMOKE / "cache").iterdir())
        cls.graphs = [torch_load(path, map_location="cpu", weights_only=False)
                      for path in sorted(cache.glob("*.pt"))[:4]]
        cls.state = checkpoint["model"]
        cls.model = UniAffinity(cls.config)
        cls.model.load_state_dict(checkpoint["model"])
        cls.model.eval()

    def test_graphs_look_like_complexes(self):
        for graph in self.graphs:
            self.assertEqual(graph.inter_types.shape[1], len(INTERACTIONS))
            self.assertGreater(graph.inter_edge_index.shape[1], 0, "a pocket with no contact edge")
            self.assertGreater(float(graph.inter_types.sum()), 0.0, "no interaction type fired at all")
            self.assertEqual(graph.protein_v.shape[1:], (4, 3))

    def test_prediction_is_invariant_under_rigid_motion(self):
        generator = torch.Generator().manual_seed(0)
        batch = collate_graphs(self.graphs)
        with torch.no_grad():
            before = self.model(batch)["affinity"]
        for _ in range(3):
            q, shift = random_rotation(generator), torch.randn(3, generator=generator) * 10
            moved = collate_graphs(self.graphs)
            moved.protein_pos = moved.protein_pos @ q + shift
            moved.ligand_pos = moved.ligand_pos @ q + shift
            moved.protein_v = moved.protein_v @ q      # the vector inputs rotate with the structure
            moved.ligand_v = moved.ligand_v @ q
            with torch.no_grad():
                after = self.model(moved)["affinity"]
            torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)

    def test_layer_is_equivariant(self):
        generator = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        layer = ScalarVectorLayer(16, 4, 8, 0.0).eval()
        n = 7
        s, v, pos = torch.randn(n, 16), torch.randn(n, 4, 3), torch.randn(n, 3) * 3
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 1], [1, 2, 3, 4, 5, 6, 0, 5]])
        edge_attr = torch.randn(edge_index.shape[1], 5)
        q = random_rotation(generator)
        with torch.no_grad():
            s1, v1 = layer(s, v, pos, edge_index, edge_attr)
            s2, v2 = layer(s, v @ q, pos @ q + 1.5, edge_index, edge_attr)
        torch.testing.assert_close(s1, s2, atol=1e-5, rtol=1e-5)       # scalars: invariant
        torch.testing.assert_close(v1 @ q, v2, atol=1e-5, rtol=1e-5)   # vectors: equivariant

    def test_backward_pass_produces_finite_gradients(self):
        model = UniAffinity(self.config)
        batch = collate_graphs(self.graphs)
        model.train()
        loss, parts = affinity_loss(model(batch)["affinity"], batch.y, self.config["loss"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"loss", "huber", "relative", "pearson_loss"})
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads, "no parameter received a gradient")
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_featurizer_builds_the_same_graph_shape_from_structures(self):
        """The cached graphs skip featurization, so this is the only coverage of PDB/SDF -> ComplexGraph."""
        case = BASE / "cofold_filter" / "demo" / "structures" / "high_quality_low_hallucination"
        data = self.config["data"]
        row = {"pdb_id": "demo_pose",
               data["structure_column"]: str(case / "predicted_protein.pdb"),
               data["ligand_column"]: str(case / "predicted_ligand.sdf"),
               data["label_column"]: 0.0, "relation": "="}
        with contextlib.redirect_stdout(io.StringIO()):   # no esm/unimol cache here, so it warns and zero-fills
            graph = ComplexFeaturizer(self.config, case)(row)
        reference = self.graphs[0]
        self.assertEqual(graph.protein_x.shape[1], reference.protein_x.shape[1])
        self.assertEqual(graph.ligand_x.shape[1], reference.ligand_x.shape[1])
        self.assertEqual(graph.protein_v.shape[1:], (4, 3))
        self.assertGreater(graph.inter_edge_index.shape[1], 0, "no contact between pocket and ligand")
        self.assertEqual(graph.inter_types.shape[1], len(INTERACTIONS))
        with torch.no_grad():
            prediction = self.model(collate_graphs([graph]))["affinity"]
        self.assertTrue(torch.isfinite(prediction).all())


    def test_released_checkpoint_matches_its_reference_predictions(self):
        reference = json.loads((SMOKE / "expected.json").read_text())["predictions"]
        batch = collate_graphs(self.graphs)
        with torch.no_grad():
            predicted = self.model(batch)["affinity"]
        for pdb_id, value in zip(batch.pdb_id, predicted.tolist()):
            self.assertAlmostEqual(value, reference[pdb_id], places=3, msg=pdb_id)

    def test_extra_tensors_are_dropped_and_missing_ones_are_fatal(self):
        """A checkpoint trained from a superset of this architecture must still load.

        The trained checkpoints carry an optional label offset and an alternative contact read-out
        that the released package does not define; strict loading rejected them outright. Dropping
        them has to leave every loaded parameter untouched, while an *absent* parameter must stay a
        hard error -- that one would keep its random initialisation and quietly skew predictions.
        """
        fat = dict(self.state)
        fat["label_offset"] = torch.zeros(())
        for module in ("contact_energy", "contact_gate"):
            for index in (0, 2):
                fat[f"{module}.{index}.weight"] = torch.randn(8, 8)
                fat[f"{module}.{index}.bias"] = torch.randn(8)

        model = UniAffinity(self.config)
        with self.assertWarns(RuntimeWarning):
            dropped = load_weights(model, fat, source="superset.pt")
        self.assertEqual(len(dropped), 9, dropped)
        loaded = model.state_dict()
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(loaded[key], value), f"{key} changed when the extras were dropped")

        thin = dict(self.state)
        victim = sorted(thin)[len(thin) // 2]
        thin.pop(victim)
        with self.assertRaises(RuntimeError) as caught:
            load_weights(UniAffinity(self.config), thin, source="truncated.pt")
        self.assertIn(victim, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
