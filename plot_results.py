# ============================================================
# Plot all model results from final_research_outputs/*.csv
# Reads the CSVs produced by full_reliability_pipeline.py and
# writes a full panel set of figures to final_research_outputs/figures/.
#
# Run: python plot_results.py
# Requires: matplotlib, seaborn, pandas, numpy
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "final_research_outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DATASETS = ["readmission", "early_stage_diabetes"]
DATASET_LABELS = {
    "readmission": "Readmission (130-US Hospitals)",
    "early_stage_diabetes": "Early-Stage Diabetes",
}
MECHANISMS = ["MCAR", "MAR", "MNAR"]
APPROACHES = ["Native", "Explicit Imputation"]
MECH_COLORS = {"MCAR": "#2c7bb6", "MAR": "#fdae61", "MNAR": "#d7191c"}
MECH_STYLES = {"MCAR": "-", "MAR": "--", "MNAR": ":"}
MECH_MARKERS = {"MCAR": "o", "MAR": "s", "MNAR": "^"}
APPROACH_STYLES = {"Native": "-", "Explicit Imputation": "--"}
METRIC_LABELS = {
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1": "F1",
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Brier": "Brier score (lower better)",
    "ECE": "ECE (lower better)",
}

DATASET_COLORS = {"readmission": "#8c6bb1", "early_stage_diabetes": "#1a9850"}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def plot_line_band(ax, x, mean, sd, color, label, style="-", marker=None):
    ax.plot(x, mean, color=color, ls=style, marker=marker, label=label, lw=1.8, ms=5)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.18, lw=0)


def fig_auroc_ece():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "cross_dataset_comparison.csv"))
    fig, axes = plt.subplots(
        len(DATASETS), 2, figsize=(12, 8), sharex=True, sharey="col"
    )
    for row, ds in enumerate(DATASETS):
        sub = df[df["Dataset"] == ds]
        for approach in APPROACHES:
            a = sub[sub["Approach"] == approach]
            for mech in MECHANISMS:
                m = a[a["Mechanism"] == mech].sort_values("Missingness")
                plot_line_band(
                    axes[row, 0], m["Missingness"], m["AUROC_Mean"], m["AUROC_SD"],
                    MECH_COLORS[mech],
                    f"{approach} / {mech}",
                    style=APPROACH_STYLES[approach], marker=MECH_MARKERS[mech],
                )
                plot_line_band(
                    axes[row, 1], m["Missingness"], m["ECE_Mean"], m["ECE_SD"],
                    MECH_COLORS[mech],
                    f"{approach} / {mech}",
                    style=APPROACH_STYLES[approach], marker=MECH_MARKERS[mech],
                )
        axes[row, 0].set_title(f"{DATASET_LABELS[ds]} - AUROC (mean +/- SD)")
        axes[row, 1].set_title(f"{DATASET_LABELS[ds]} - ECE (mean +/- SD)")
    for ax in axes.flat:
        ax.set_xlabel("Missingness (%)")
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.autoscale(tight=True)
    fig.suptitle("Impact of missingness on model reliability across mechanisms/approaches",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGURES_DIR, "fig_auroc_ece.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def fig_all_metrics():
    for ds in DATASETS:
        df = pd.read_csv(os.path.join(
            OUTPUT_DIR, f"robustness_results_{ds}.csv"))
        metric_cols = ["Accuracy", "Precision", "Recall", "F1",
                       "AUROC", "AUPRC", "Brier", "ECE"]
        agg = (
            df.groupby(["Approach", "Mechanism", "Missingness"])[metric_cols]
            .agg(["mean", "std"])
        )
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        agg = agg.reset_index()

        fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
        for ax, metric in zip(axes.flat, metric_cols):
            for approach in APPROACHES:
                a = agg[agg["Approach"] == approach]
                for mech in MECHANISMS:
                    m = a[a["Mechanism"] == mech].sort_values("Missingness")
                    plot_line_band(
                        ax, m["Missingness"], m[f"{metric}_mean"],
                        m[f"{metric}_std"], MECH_COLORS[mech],
                        f"{approach} / {mech}",
                        style=APPROACH_STYLES[approach], marker=MECH_MARKERS[mech],
                    )
            ax.set_title(METRIC_LABELS[metric], fontsize=11)
            ax.set_xlabel("Missingness (%)")
            ax.set_xticks([10, 20, 30, 40, 50])
        axes.flat[0].legend(fontsize=7, loc="best")
        fig.suptitle(f"{DATASET_LABELS[ds]} - all metrics vs missingness "
                     "(mean +/- SD over 10 seeds)", fontsize=14, y=1.0)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(FIGURES_DIR, f"fig_all_metrics_{ds}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out}")


def fig_cross_dataset():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "cross_dataset_comparison.csv"))
    fig, axes = plt.subplots(len(APPROACHES), 2, figsize=(13, 8), sharex=True)
    for row, approach in enumerate(APPROACHES):
        a = df[df["Approach"] == approach]
        for col, metric, title in [
            (0, "AUROC", "AUROC (higher better)"),
            (1, "ECE", "ECE (lower better)"),
        ]:
            for ds in DATASETS:
                sub = a[a["Dataset"] == ds]
                for mech in MECHANISMS:
                    m = sub[sub["Mechanism"] == mech].sort_values("Missingness")
                    axes[row, col].plot(
                        m["Missingness"], m[f"{metric}_Mean"],
                        color=DATASET_COLORS[ds],
                        ls=MECH_STYLES[mech], marker=MECH_MARKERS[mech],
                        lw=1.8, ms=5,
                        label=f"{DATASET_LABELS[ds]} / {mech}",
                    )
            axes[row, col].set_title(f"{approach} - {title}", fontsize=11)
            axes[row, col].set_xlabel("Missingness (%)")
            axes[row, col].set_xticks([10, 20, 30, 40, 50])
        axes[row, 0].legend(fontsize=7, loc="best", ncol=2)
    fig.suptitle("Cross-dataset comparison: does the reliability pattern generalize?",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIGURES_DIR, "fig_cross_dataset.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def fig_divergence():
    for ds in DATASETS:
        df = pd.read_csv(os.path.join(OUTPUT_DIR, f"mar_vs_mcar_divergence_{ds}.csv"))
        fig, ax = plt.subplots(figsize=(7, 5))
        for approach in APPROACHES:
            a = df[df["Approach"] == approach].sort_values("Missingness")
            ax.plot(a["Missingness"], a["Absolute_Gap"], marker="o",
                    label=approach, lw=1.8)
        ax.set_xlabel("Missingness (%)")
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_ylabel("|AUROC_MCAR - AUROC_MAR|")
        ax.set_title(f"{DATASET_LABELS[ds]} - MAR vs MCAR divergence (sanity check)")
        ax.legend(fontsize=9)
        out = os.path.join(FIGURES_DIR, f"fig_mar_mcar_divergence_{ds}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    fig_auroc_ece()
    fig_all_metrics()
    fig_cross_dataset()
    fig_divergence()
    print(f"All figures written to {FIGURES_DIR}")