# AdamW + warmup/cosine, fp32, early stopping on valid rmse. A run writes config.yaml, code_snapshot/,
# train.log, metrics.jsonl, checkpoints/, predictions/, summary.json.
from __future__ import annotations

import csv
import json
import math
import random
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from .config import save_config
from .data import FEATURIZER_VERSION, PDBBindDataset, collate_graphs, make_splits
from .losses import affinity_loss
from .model import UniAffinity
from .ops import torch_load


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Logger:
    """print() + append to <run_dir>/train.log."""

    def __init__(self, run_dir: Path | None = None):
        self.file = (run_dir / "train.log").open("a", encoding="utf-8") if run_dir else None

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        print(message, flush=True)
        if self.file:
            self.file.write(f"{stamp} {message}\n"); self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()


def metrics(y: list[float], pred: list[float]) -> dict[str, float]:
    """MAE / RMSE / Pearson / Spearman over one split."""
    ya, pa = np.asarray(y), np.asarray(pred)
    if not len(ya):
        return {"mae": math.nan, "rmse": math.nan, "pearson": math.nan, "spearman": math.nan}
    ok = len(ya) > 1 and np.std(ya) > 0 and np.std(pa) > 0
    corr_p = float(pearsonr(ya, pa)[0]) if ok else 0.0
    corr_s = float(spearmanr(ya, pa)[0]) if ok else 0.0
    return {"mae": float(np.abs(ya - pa).mean()), "rmse": float(np.sqrt(np.square(ya - pa).mean())),
            "pearson": corr_p, "spearman": corr_s, "n": int(len(ya))}


PT_MIN = 5  # per-target metrics: only targets with >= PT_MIN complexes in the evaluated split count


# for pt_spearman: are ligands ranked *within* a target, or are easy targets just split from hard ones
def target_groups(*frames: pd.DataFrame) -> dict[str, int]:
    keys: dict[str, int] = {}
    out: dict[str, int] = {}
    for frame in frames:
        uni = frame["uniprot_id"] if "uniprot_id" in frame else pd.Series([None] * len(frame), index=frame.index)
        for pdb_id, u in zip(frame["pdb_id"], uni):
            key = str(u) if isinstance(u, str) and u else f"pdb:{pdb_id}"
            out[pdb_id] = keys.setdefault(key, len(keys))
    return out


@torch.no_grad()
def evaluate(model: UniAffinity, loader: DataLoader, device: torch.device, loss_cfg: dict,
             group_of: dict[str, int] | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval(); targets, predictions, records, losses, gids = [], [], [], [], []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        pred = model(batch)["affinity"]
        loss, _ = affinity_loss(pred, batch.y, loss_cfg)
        losses.append(float(loss))
        for pdb_id, y, p in zip(batch.pdb_id, batch.y.cpu().tolist(), pred.cpu().tolist()):
            records.append({"pdb_id": pdb_id, "target": y, "prediction": p})
            targets.append(y); predictions.append(p)
            gids.append(group_of[pdb_id] if group_of is not None else -1)
    result = metrics(targets, predictions)
    result["loss"] = float(np.mean(losses)) if losses else math.nan
    result["pt_spearman"], result["pt_targets"] = math.nan, 0
    if group_of is not None and targets:
        frame = pd.DataFrame({"g": gids, "y": targets, "p": predictions})
        rhos = [spearmanr(d.y, d.p)[0] for _, d in frame.groupby("g")
                if len(d) >= PT_MIN and d.y.std() > 0 and d.p.std() > 0]
        if rhos:  # mean over targets, so a few large targets cannot dominate
            result["pt_spearman"], result["pt_targets"] = float(np.mean(rhos)), len(rhos)
    return result, records


def write_predictions(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "target", "prediction"])
        writer.writeheader(); writer.writerows(records)


def make_loaders(config: dict, base_dir: Path) -> tuple[dict[str, DataLoader], Path, dict]:
    train_frame, valid, test, root = make_splits(config, base_dir)
    frames = {"train": train_frame, "valid": valid, "test": test}
    loaders = {}
    for name, frame in frames.items():
        dataset = PDBBindDataset(frame, config, root)
        # RandomSampler rejects a zero-length dataset, and evaluate() has no train split
        loaders[name] = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                                   shuffle=name == "train" and len(dataset) > 0,
                                   num_workers=config["data"]["num_workers"], collate_fn=collate_graphs,
                                   pin_memory=torch.cuda.is_available())
    print("[data] " + ", ".join(f"{name}={len(loader.dataset)}" for name, loader in loaders.items()), flush=True)
    return loaders, root, frames


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> Any:
    warmup, epochs = int(cfg["warmup_epochs"]), int(cfg["epochs"])
    min_ratio = float(cfg["min_lr_ratio"])

    def factor(epoch: int) -> float:  # epoch is the 0-based count of finished epochs
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def snapshot_code(run_dir: Path) -> None:
    """Copy the exact model/data code into the run, so every checkpoint can be traced to its architecture."""
    source = Path(__file__).resolve().parent
    target = run_dir / "code_snapshot" / "uniaffinity"
    target.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*.py"):
        shutil.copy2(file, target / file.name)


def train(config: dict, base_dir: Path) -> Path:
    seed_everything(config["experiment"]["seed"])
    device = get_device(config["training"]["device"])
    tcfg = config["training"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(config["experiment"]["output_dir"])
    output_root = output_root if output_root.is_absolute() else base_dir / output_root
    tag = config["experiment"].get("tag")
    run_dir = output_root / (f"{config['experiment']['name']}" + (f"_{tag}" if tag else "") + f"_{stamp}")
    if config["experiment"].get("run_dir"):  # fixed location when a script drives a grid of runs
        run_dir = Path(config["experiment"]["run_dir"])
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    save_config(config, run_dir / "config.yaml")
    if config["experiment"]["snapshot_code"]:
        snapshot_code(run_dir)
    logger = Logger(run_dir)
    logger.log(f"[run] {run_dir}")
    logger.log(f"[run] command: {' '.join(sys.argv)}")
    loaders, _, frames = make_loaders(config, base_dir)
    group_of = target_groups(*frames.values())  # per-target metrics are always per UniProt target

    model = UniAffinity(config).to(device)
    # materialize the Lazy layers before the optimizer sees the parameters
    first = next((b for b in loaders["valid"] if b is not None), None) or next(b for b in loaders["train"] if b is not None)
    with torch.no_grad():
        model.eval(); model(first.to(device))
    n_params = sum(p.numel() for p in model.parameters())
    logger.log(f"[model] parameters={n_params:,} device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    scheduler = build_scheduler(optimizer, tcfg)
    monitor = tcfg["monitor"]
    maximize = tcfg["monitor_mode"] == "max"
    best, best_epoch, stale = (-math.inf if maximize else math.inf), 0, 0
    history_path = run_dir / "metrics.jsonl"
    for epoch in range(1, tcfg["epochs"] + 1):
        start = time.time()
        model.train(); totals: dict[str, list[float]] = {}; grad_norms = []
        for batch in loaders["train"]:
            if batch is None:
                continue
            batch = batch.to(device); optimizer.zero_grad(set_to_none=True)
            loss, parts = affinity_loss(model(batch)["affinity"], batch.y, config["loss"])
            loss.backward()
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])))
            optimizer.step()
            for key, value in parts.items():
                totals.setdefault(key, []).append(float(value))
        train_stats = {key: float(np.mean(values)) for key, values in totals.items()}
        valid_stats, _ = evaluate(model, loaders["valid"], device, config["loss"], group_of)
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "lr": lr, "time_s": round(time.time() - start, 1),
               "grad_norm": float(np.mean(grad_norms)) if grad_norms else math.nan,
               **{f"train/{k}": v for k, v in train_stats.items()},
               **{f"valid/{k}": v for k, v in valid_stats.items()}}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=True) + "\n")
        score = valid_stats[monitor]
        improved = math.isfinite(score) and ((score > best) if maximize else (score < best))
        state = {"epoch": epoch, "config": config, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "valid": valid_stats,
                 "featurizer_version": FEATURIZER_VERSION}
        torch.save(state, run_dir / "checkpoints" / "last.pt")
        # always write on the first epoch so best.pt exists
        if improved or best_epoch == 0:
            best, best_epoch, stale = (score if math.isfinite(score) else best), epoch, 0
            torch.save(state, run_dir / "checkpoints" / "best.pt")
        else:
            stale += 1
        logger.log(f"[epoch {epoch:03d}] train_loss={train_stats.get('loss', math.nan):.4f} "
                   f"valid_rmse={valid_stats['rmse']:.4f} valid_r={valid_stats['pearson']:.4f} "
                   f"valid_ptrho={valid_stats['pt_spearman']:.4f} "
                   f"lr={lr:.2e} {row['time_s']}s{' *best*' if best_epoch == epoch else ''}")
        scheduler.step()
        if stale >= tcfg["patience"]:
            logger.log(f"[train] early stopping after {epoch} epochs (best epoch {best_epoch})")
            break
    checkpoint = torch_load(run_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    summary: dict[str, Any] = {"run_dir": str(run_dir), "best_epoch": best_epoch, f"best_valid_{monitor}": best,
                               "parameters": n_params, "featurizer_version": FEATURIZER_VERSION}
    for split in ("valid", "test"):
        stats, records = evaluate(model, loaders[split], device, config["loss"], group_of)
        write_predictions(records, run_dir / "predictions" / f"{split}.csv")
        summary[split] = stats
    (run_dir / "test_metrics.json").write_text(json.dumps(summary["test"], indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.log(f"[test] {json.dumps(summary['test'])}")
    logger.log(f"[run] finished: {run_dir}")
    logger.close()
    return run_dir


def load_weights(model: UniAffinity, state: dict[str, Any], source: Any = "checkpoint") -> list[str]:
    """Load a state dict, tolerating tensors for branches this build does not define.

    A checkpoint trained from a superset of this architecture -- an alternative read-out, an optional
    label offset -- carries parameters that have no module here to receive them. Those are inert: the
    code that would read them does not exist in this package, so they are dropped and named in a
    warning. A *missing* parameter is the dangerous case, because the module would silently keep its
    random initialisation and the predictions would be quietly wrong, so that is a hard error.
    Shape mismatches are raised by torch itself and are never tolerated.

    Returns the dropped keys.
    """
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"{source}: {len(incompatible.missing_keys)} parameter(s) absent from the checkpoint, so "
            f"they would keep their random initialisation: {sorted(incompatible.missing_keys)}. "
            "The checkpoint does not match this architecture -- check the config it was trained with."
        )
    if incompatible.unexpected_keys:
        dropped = sorted(incompatible.unexpected_keys)
        warnings.warn(
            f"{source}: ignoring {len(dropped)} tensor(s) for modules this build does not define: "
            f"{dropped}. They are unused here; the loaded parameters are complete.",
            RuntimeWarning, stacklevel=2,
        )
        return dropped
    return []


# config travels inside the checkpoint, so no yaml needed
def load_model(checkpoint_path: Path, device: torch.device) -> tuple[UniAffinity, dict]:
    checkpoint = torch_load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = UniAffinity(config).to(device)
    load_weights(model, checkpoint["model"], source=Path(checkpoint_path).name)
    model.eval()
    return model, config
