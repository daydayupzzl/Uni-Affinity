# python -m uniaffinity.cli {prepare,train,evaluate,predict} --config configs/smoke.yaml
# Run from the repository root.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .config import load_config
from .data import ComplexFeaturizer, PDBBindDataset, collate_graphs, make_splits
from .ops import torch_load
from .trainer import evaluate, get_device, load_model, make_loaders, train, write_predictions


# relative paths in a config resolve against the repo root, not the cwd
BASE = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Uni-Affinity training and inference")
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        p = sub.add_parser(command)
        p.add_argument("--config", default=str(BASE / "configs" / "smoke.yaml"))
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config entry, e.g. --set training.epochs=10")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True); p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--output", default=None, help="CSV path (default: <run>/predictions/eval_<split>.csv)")
    p = sub.add_parser("predict", help="Score one pose (--protein/--ligand) or many poses (--csv)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--protein"); p.add_argument("--ligand"); p.add_argument("--pdb-id", default="query")
    p.add_argument("--csv", help="CSV with columns id,protein,ligand (paths absolute or relative to the CSV)")
    p.add_argument("--output", default=None, help="Output CSV for --csv mode")
    p.add_argument("--top-k", type=int, default=10)
    return root


def _featurize_pose(config: dict, protein: Path, ligand: Path, pdb_id: str):
    row = {"pdb_id": pdb_id, config["data"]["structure_column"]: str(protein.resolve()),
           config["data"]["ligand_column"]: str(ligand.resolve()), config["data"]["label_column"]: 0.0, "relation": "="}
    return ComplexFeaturizer(config, protein.parent)(row)


def main() -> None:
    args = parser().parse_args()
    if args.command in ("prepare", "train"):
        config = load_config(args.config, args.set)
    if args.command == "train":
        train(config, BASE)
    elif args.command == "prepare":
        frames = make_splits(config, BASE)
        root = frames[3]
        for name, frame in zip(("train", "valid", "test"), frames[:3]):
            dataset = PDBBindDataset(frame, config, root)
            ok = sum(dataset[i] is not None for i in range(len(dataset)))
            print(f"[prepare] {name}: {ok}/{len(dataset)} cached in {dataset.cache}")
    elif args.command == "evaluate":
        checkpoint = Path(args.checkpoint).resolve()
        config = torch_load(checkpoint, map_location="cpu", weights_only=False)["config"]
        device = get_device(config["training"]["device"])
        model, _ = load_model(checkpoint, device)
        loaders, _, _ = make_loaders(config, BASE)
        result, records = evaluate(model, loaders[args.split], device, config["loss"])
        output = Path(args.output) if args.output else checkpoint.parents[1] / "predictions" / f"eval_{args.split}.csv"
        write_predictions(records, output)
        print(json.dumps({"split": args.split, "metrics": result, "predictions_csv": str(output)}, indent=2))
    # NOTE pretrained features come from the cache_dir inside the checkpoint config, keyed by id; a missing
    # file is zero-filled with a warning rather than an error
    elif args.command == "predict":
        checkpoint = Path(args.checkpoint).resolve()
        config = torch_load(checkpoint, map_location="cpu", weights_only=False)["config"]
        device = get_device(config["training"]["device"])
        model, _ = load_model(checkpoint, device)
        if args.csv:
            table = pd.read_csv(args.csv)
            base = Path(args.csv).resolve().parent
            rows = []
            for item in table.to_dict("records"):
                protein, ligand = Path(str(item["protein"])), Path(str(item["ligand"]))
                protein = protein if protein.is_absolute() else base / protein
                ligand = ligand if ligand.is_absolute() else base / ligand
                try:
                    graph = _featurize_pose(config, protein, ligand, str(item["id"]))
                    with torch.no_grad():
                        value = float(model(collate_graphs([graph]).to(device))["affinity"][0])
                    rows.append({"id": item["id"], "predicted_pK": value, "error": ""})
                except Exception as exc:  # keep screening when one pose is broken
                    rows.append({"id": item["id"], "predicted_pK": float("nan"), "error": str(exc)})
            output = Path(args.output) if args.output else Path(args.csv).with_name(Path(args.csv).stem + "_predictions.csv")
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"[predict] {len(rows)} poses -> {output}")
            return
        if not (args.protein and args.ligand):
            raise SystemExit("predict needs --protein and --ligand, or --csv")
        graph = _featurize_pose(config, Path(args.protein), Path(args.ligand), args.pdb_id)
        with torch.no_grad():
            out = model(collate_graphs([graph]).to(device), return_attention=True)
        score = out.get("contribution", out["attention"])
        top = torch.topk(score.abs(), min(args.top_k, score.numel())).indices.cpu()
        edges = out["inter_edge_index"][:, top].cpu().T.tolist()
        print(json.dumps({"pdb_id": args.pdb_id, "predicted_pK": float(out["affinity"][0]),
                          "top_contacts": [{"protein_node": e[0], "ligand_node": e[1],
                                            "attention": float(out["attention"][idx]),
                                            **({"contribution": float(out["contribution"][idx])} if "contribution" in out else {})}
                                           for idx, e in zip(top.tolist(), edges)]}, indent=2))


if __name__ == "__main__":
    main()
