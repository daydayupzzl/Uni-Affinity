#!/usr/bin/env python
"""Redraw cases.png from the files run_demo.py reads.

One row per case; columns are quality by scope, novelty by scope, the decision, the ground-truth RMSD.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from filter import apply_to_frame  # noqa: E402
from ligand_plddt import ligand_plddt  # noqa: E402

MM = 1 / 25.4
W = 122.0
INK, MUTED, GREY, FRAME = "#0b0b0b", "#52514e", "#d9d7d2", "#a9a7a2"
BLUE, ORANGE = "#2a78d6", "#eb6834"
VERDICT = {"accepted": "#1baf7a", "uncertain": "#eda100", "rejected": "#e34948"}
GLOBAL_THRESHOLD = 0.84   # single ligand-pLDDT cut-off calibrated to the same 90% target precision
ANG = "Å"
BOX = dict(fc="white", ec="none", alpha=0.82, pad=0.7)

plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 5.4,
                     "axes.labelsize": 5.2, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8})

X0, PW, GAP = 2.0, 26.0, 4.67
PH, HEAD, ROWGAP, TOP = 15.0, 4.4, 4.6, 10.6
TITLES = ["Quality: pLDDT, by scope", "Hallucination: novelty, by scope",
          "Decision: does it clear its bar?", "Ground truth (never an input)"]


def col_x(j):
    return X0 + j * (PW + GAP)


def panel(fig, fig_h, col, row_y, h=PH):
    ax = fig.add_axes([col_x(col) / W, 1 - (row_y + h) / fig_h, PW / W, h / fig_h])
    ax.tick_params(length=1.6, width=0.4, pad=1.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4), ax.spines["bottom"].set_color(FRAME)
    ax.grid(False)
    ax.set_yticks([])
    return ax


def labelled_bars(ax, labels, values, colours, xmax, fmt):
    """Three bars with the category name above each one -- no left gutter, so nothing can overflow."""
    for k, (lab, v, c) in enumerate(zip(labels, values, colours)):
        y = len(labels) - 1 - k
        ax.barh([y], [v], 0.40, color=c, zorder=3)
        ax.text(0.0, y + 0.34, lab, fontsize=4.4, va="bottom", ha="left", color=MUTED)
        ax.text(v + xmax * 0.02, y, fmt.format(v), va="center", fontsize=4.6, color=INK)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.45, len(labels) - 0.15)


cases = apply_to_frame(pd.read_csv(HERE / "cases.csv"), policy="strict")

# recompute the three pLDDT scopes from the bundled Boltz output, so the figure cannot drift from the data
scopes = []
for row in cases.itertuples():
    base = HERE / "structures" / row.case / "boltz_output"
    q = ligand_plddt(base / "predictions" / row.system_id,
                     base / "processed" / "structures" / f"{row.system_id}.npz", row.model_ligand_chain)
    assert abs(q["lig_plddt"] - row.lig_plddt) < 1e-4, row.case
    scopes.append(q)

FIG_H = TOP + len(cases) * (HEAD + PH) + (len(cases) - 1) * ROWGAP + 7.4
fig = plt.figure(figsize=(W * MM, FIG_H * MM))

for j, title in enumerate(TITLES):
    fig.text(col_x(j) / W, 1 - 2.0 / FIG_H, title, fontsize=5.4, fontweight="bold", va="top", ha="left", color=INK)
fig.text(X0 / W, 1 - 5.1 / FIG_H,
         "The novelty tier sets the quality bar, so the same pLDDT passes in one tier and not in another.",
         fontsize=4.9, va="top", ha="left", color=MUTED)
fig.text(X0 / W, 1 - 7.9 / FIG_H,
         f"orange line = threshold for this tier    dashed = one global threshold ({GLOBAL_THRESHOLD:.2f}) "
         f"instead of per-tier    black line = 2 {ANG} success cut-off",
         fontsize=4.4, va="top", ha="left", color=MUTED)

for i, (row, q) in enumerate(zip(cases.itertuples(), scopes)):
    y = TOP + i * (HEAD + PH + ROWGAP)
    fig.text(X0 / W, 1 - (y + 2.5) / FIG_H, row.case.replace("_", " "), fontsize=5.5, fontweight="bold",
             va="baseline", ha="left", color=INK)
    fig.text((col_x(1) + PW) / W, 1 - (y + 2.5) / FIG_H,
             f"{row.system_id.split('__')[0]} / {row.ligand_ccd} / seed {row.seed}",
             fontsize=4.7, color=MUTED, va="baseline", ha="right")

    # --- (a) quality at three scopes: only the ligand-local one sees the problem
    ax = panel(fig, FIG_H, 0, y + HEAD)
    labelled_bars(ax, ["whole complex", "all non-protein chains", "target ligand"],
                  [q["complex_plddt"], q["nonprotein_plddt"], q["lig_plddt"]], [GREY, GREY, BLUE],
                  1.0, "{:.3f}")
    ax.set_xticks([0, 0.5, 1.0])
    if i == len(cases) - 1:
        ax.set_xlabel("mean pLDDT", fontsize=5.0, labelpad=1)

    # --- (b) novelty at three scopes: only the complex-level one does
    ax = panel(fig, FIG_H, 1, y + HEAD)
    labelled_bars(ax, ["protein alone  $P_{mm}$", "ligand alone  $L_{tc}$", "complex  $J_{mm}$"],
                  [row.P_mm, row.L_tc_ccd * 100, row.J_mm], [GREY, GREY, ORANGE], 100.0, "{:.0f}")
    ax.set_xticks([0, 50, 100])
    if i == len(cases) - 1:
        ax.set_xlabel("similarity to the pre-cutoff PDB (%)", fontsize=5.0, labelpad=1)

    # --- (c) the decision, and the gap that produces it.
    # Rules and bar live in the lower band; everything written stays above them, so nothing overlaps.
    ax = panel(fig, FIG_H, 2, y + HEAD)
    ax.set_xlim(0, 1.0), ax.set_ylim(-0.45, 2.05)
    ax.axvspan(row.threshold_strict, 1.0, ymax=0.40, color=VERDICT["accepted"], alpha=0.12, lw=0, zorder=1)
    ax.barh([0], [row.lig_plddt], 0.40, color=BLUE, zorder=3)
    ax.axvline(row.threshold_strict, ymax=0.40, color=ORANGE, lw=0.9, zorder=4)
    ax.axvline(GLOBAL_THRESHOLD, ymax=0.40, color=MUTED, lw=0.7, ls=(0, (2.2, 1.6)), zorder=4)
    ax.set_xticks([0, 0.5, 1.0])
    ax.text(0.0, 1.46, f"pLDDT {row.lig_plddt:.3f}, tier {row.J_tier} needs {row.threshold_strict:.2f}",
            fontsize=4.4, va="bottom", ha="left", color=MUTED)
    ax.text(0.0, 0.62, f"{row.lig_plddt - row.threshold_strict:+.3f}   {row.state.upper()}",
            fontsize=5.4, fontweight="bold", va="bottom", ha="left", color=VERDICT[row.state])
    if i == len(cases) - 1:
        ax.set_xlabel("ligand pLDDT", fontsize=5.0, labelpad=1)

    # --- (d) what actually happened
    ax = panel(fig, FIG_H, 3, y + HEAD)
    ok = bool(row.true_success)
    ax.set_xscale("log"), ax.set_xlim(0.1, 60), ax.set_ylim(-0.45, 2.05)
    ax.axvspan(0.1, 2.0, ymax=0.40, color=VERDICT["accepted"], alpha=0.12, lw=0, zorder=1)
    ax.barh([0], [row.true_ligand_rmsd], 0.40, color=VERDICT["accepted"] if ok else VERDICT["rejected"], zorder=3)
    ax.axvline(2.0, ymax=0.40, color=INK, lw=0.7, zorder=4)
    ax.set_xticks([0.1, 1, 10]), ax.set_xticklabels(["0.1", "1", "10"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.text(0.105, 1.46, f"ligand RMSD {row.true_ligand_rmsd:.2f} {ANG}", fontsize=4.4,
            va="bottom", ha="left", color=MUTED)
    ax.text(0.105, 0.62, "correct pose" if ok else "wrong pose", fontsize=5.4, fontweight="bold",
            va="bottom", ha="left", color=VERDICT["accepted"] if ok else VERDICT["rejected"])
    if i == len(cases) - 1:
        ax.set_xlabel(f"ligand RMSD ({ANG}, log)", fontsize=5.0, labelpad=1)

out = HERE / "cases"
fig.savefig(f"{out}.png", dpi=320)
plt.close(fig)
print(f"wrote {out}.png")
