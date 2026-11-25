"""Build ROC-curve and confusion-matrix figures from saved test evaluations.

Run from the repository root after evaluation.evaluate has written runs/eval/:
    python paper/figures/make_eval_figures.py

Writes paper/figures/roc_curves.png and paper/figures/confusion_matrices.png.
Everything is read from the per-video predictions in the metrics JSON files;
nothing is re-run. Each seed has its own grouped split, so curves are only
compared within a seed.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

EVAL = "runs/eval"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 7, 123]
# Seed 42 uses the 5-epoch baseline so every model shares the same training budget.
MODELS = [
    ("ResNet18 baseline", {42: "dfdc_baseline_5ep", 7: "dfdc_baseline_s7", 123: "dfdc_baseline_s123"}, "tab:red"),
    ("Hybrid, no Transformer", {42: "dfdc_hybrid_notemporal", 7: "dfdc_hybrid_notemporal_s7",
                                123: "dfdc_hybrid_notemporal_s123"}, "tab:green"),
    ("Hybrid (proposed)", {42: "dfdc_hybrid", 7: "dfdc_hybrid_s7", 123: "dfdc_hybrid_s123"}, "tab:blue"),
]


def load(run, suffix=""):
    path = os.path.join(EVAL, f"{run}_best_test{suffix}.json")
    if not os.path.exists(path):
        print(f"missing {path}, skipped")
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def roc_figure():
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(10.5, 3.9), sharey=True, layout="constrained")
    for ax, seed in zip(axes, SEEDS):
        n = None
        for name, runs, colour in MODELS:
            m = load(runs[seed])
            if m is None:
                continue
            fpr, tpr, _ = roc_curve([p["label"] for p in m["predictions"]],
                                    [p["fake_probability"] for p in m["predictions"]])
            ax.plot(fpr, tpr, color=colour, lw=1.8, label=f"{name} (AUC {m['roc_auc']:.3f})")
            n = m["support"]
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        title = f"Seed {seed}" + (f" ({n['real']} real, {n['fake']} fake)" if n else "")
        ax.set(title=title, xlabel="False positive rate", xlim=(0, 1), ylim=(0, 1.02), aspect="equal")
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel("True positive rate")
    fig.savefig(os.path.join(OUT, "roc_curves.png"), dpi=200)
    plt.close(fig)


def confusion_figure():
    fig, axes = plt.subplots(len(MODELS), len(SEEDS), figsize=(7.5, 7.5))
    for row, (name, runs, _) in enumerate(MODELS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            m = load(runs[seed], "_valthr")
            if m is None:
                ax.axis("off")
                continue
            cm = m["confusion_matrix"]
            grid = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            # Shade by row share so the 11-video REAL row stays readable next to the FAKE row.
            share = [[v / max(sum(r), 1) for v in r] for r in grid]
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, grid[i][j], ha="center", va="center", fontsize=12,
                            color="white" if share[i][j] > 0.6 else "black")
            ax.set_xticks([0, 1], ["REAL", "FAKE"], fontsize=8)
            ax.set_yticks([0, 1], ["REAL", "FAKE"], fontsize=8)
            ax.set_title(f"{name}\nseed {seed}, threshold {m['threshold']:.2f}", fontsize=8)
            if row == len(MODELS) - 1:
                ax.set_xlabel("Predicted", fontsize=8)
            if col == 0:
                ax.set_ylabel("True", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    roc_figure()
    confusion_figure()
    print("wrote roc_curves.png and confusion_matrices.png to", OUT)
