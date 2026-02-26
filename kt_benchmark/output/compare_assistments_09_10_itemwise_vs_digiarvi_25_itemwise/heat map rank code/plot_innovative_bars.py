from __future__ import annotations

"""
Innovative publication-ready bar charts for benchmark table with two datasets.

Usage
-----
Option A: Run with your own tidy CSV (recommended)
    python -m kt_benchmark.plot_innovative_bars --input path/to/bench_long.csv

CSV schema (long/tidy):
    Model,Metric,Dataset,Value
Examples of values:
    BKT,Accuracy,ASSISTments 09-10,0.681
    BKT,Accuracy,DigiArvi 2025,0.633
    ...

Option B: Quickly preview with the embedded example
    python -m kt_benchmark.plot_innovative_bars  # no --input provided
Then replace DEFAULT_DATA below or provide a CSV later.

Figures generated
-----------------
Per metric: a two-panel figure split into 'Classical baselines' and 'Advanced models',
with fixed model colors, dataset-specific hatch patterns, per-bar value labels, ranked order
within each panel, and a shared legend under the panels. Outputs PNG and PDF to:
    kt_benchmark/plots/innovative_bars/

This script mirrors the grouped two-panel style used in trajectories and (by preference)
ROC/PR plots: consistent colors, family-based grouping, per-model identity across figures.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib as mpl

# ------------------------------------------------------------
# Naming and styling helpers (kept local for standalone usage)
# ------------------------------------------------------------

def _sanitize_model_name(name: str) -> str:
    import re
    s = name
    s = re.sub(r"\s*\((?:minimal|coral)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\blite\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip().strip("-_ ")
    return s


def _display_model_name(name: str) -> str:
    key = _sanitize_model_name(name).lower()
    mapping = {"logisticregression": "LR", "rasch1pl": "Rasch 1PL"}
    return mapping.get(key, _sanitize_model_name(name))


def _family_of_model(name: str) -> str:
    n = _sanitize_model_name(name).lower()
    if n in {"dkt", "fkt"}:
        return "deep"
    if n in {"bkt", "rasch1pl", "tirt", "logisticregression"}:
        return "classical"
    if n in {"clkt", "adaptkt"}:
        return "hybrid"
    if n in {"gkt"}:
        return "graph"
    return "other"


def _fixed_colors() -> Dict[str, str]:
    # Consistent with kt_benchmark.plot_results
    return {
        "dkt": "#CC79A7",          # magenta
        "fkt": "#009E73",          # green
        "clkt": "#0072B2",         # blue
        "adaptkt": "#D55E00",      # dark orange
        "gkt": "#E69F00",          # orange/yellow
        "bkt": "#56B4E9",          # light blue
        "rasch1pl": "#0072B2",     # blue
        "logisticregression": "#009E73",  # green
        "tirt": "#D55E00",         # dark orange
    }


# ------------------------------------------------------------
# Default embedded data (edit or provide a CSV via --input)
# ------------------------------------------------------------
# Leave empty by default; user can fill or pass a CSV.
DEFAULT_DATA: List[Dict[str, object]] = []

# Example structure if you want to hardcode (uncomment and fill):
# DEFAULT_DATA = [
#     {"Model": "BKT", "Metric": "Accuracy", "Dataset": "ASSISTments 09-10", "Value": 0.681},
#     {"Model": "BKT", "Metric": "Accuracy", "Dataset": "DigiArvi 2025", "Value": 0.633},
#     ...
# ]

# Display order for panels and metrics
FAMILY_PANELS = {
    "Classical baselines": ["Rasch 1PL", "BKT", "LogisticRegression", "TIRT"],
    "Advanced models": ["DKT", "FKT", "CLKT", "AdaptKT", "GKT"],
}

METRICS_ORDER = [
    "Accuracy",
    "ROC-AUC",
    "Average Precision",
    "F1 Score",
    "Log Loss",
    # "Rank",  # optional; handled separately
]

DATASETS_ORDER = ["ASSISTments 09-10", "DigiArvi 2025"]
HATCH_BY_DATASET = {
    "ASSISTments 09-10": "\\\\",
    "DigiArvi 2025": "..",
}

# Axis ranges for metrics (min, max) to keep scales consistent across panels
METRIC_RANGES: Dict[str, Optional[tuple]] = {
    "Accuracy": (0.3, 1.0),
    "ROC-AUC": (0.3, 1.0),
    "Average Precision": (0.3, 1.0),
    "F1 Score": (0.3, 1.0),
    "Log Loss": None,  # auto; smaller is better
}


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_long_df(input_csv: Optional[str]) -> pd.DataFrame:
    # Priority: explicit path -> packaged default CSV -> embedded DEFAULT_DATA
    if input_csv:
        df = pd.read_csv(input_csv)
    else:
        default_path = Path(__file__).resolve().parent / "benchmark_summary_long.csv"
        if default_path.exists():
            df = pd.read_csv(default_path)
        elif DEFAULT_DATA:
            df = pd.DataFrame(DEFAULT_DATA)
        else:
            raise SystemExit(
                "No --input provided and no default CSV found at kt_benchmark/data/benchmark_summary_long.csv, "
                "and DEFAULT_DATA is empty. Please provide a CSV or fill DEFAULT_DATA."
            )
    # basic normalizations
    req_cols = {"Model", "Metric", "Dataset", "Value"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["Model"] = df["Model"].astype(str)
    df["Metric"] = df["Metric"].astype(str)
    df["Dataset"] = df["Dataset"].astype(str)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])  # only keep numeric
    return df


# ------------------------------------------------------------
# Plotting core
# ------------------------------------------------------------

def _model_color(model: str, palette_fallback: List) -> str:
    key = _sanitize_model_name(model).lower()
    return _fixed_colors().get(key, palette_fallback[hash(key) % len(palette_fallback)])


def _sort_models_for_metric(df_m: pd.DataFrame, models: List[str], metric: str) -> List[str]:
    # sort by average across datasets (desc), except Log Loss (asc)
    sub = df_m[df_m["Model"].isin(models)]
    agg = sub.groupby("Model")["Value"].mean().to_dict()
    if metric.lower().strip() == "log loss":
        order = sorted(models, key=lambda m: (agg.get(m, np.inf), _sanitize_model_name(m).lower()))
    else:
        order = sorted(models, key=lambda m: (-(agg.get(m, -np.inf)), _sanitize_model_name(m).lower()))
    return order


def _draw_two_panel_bars(df: pd.DataFrame, metric: str, outdir: Path) -> None:
    # Prepare
    palette_fb = sns.color_palette("colorblind", n_colors=12)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.2), sharex=False, sharey=False)

    # panel loop
    for ax, (panel_name, panel_models) in zip(axes, FAMILY_PANELS.items()):
        present_models = [m for m in panel_models if m in set(df["Model"]) ]
        if not present_models:
            ax.axis("off")
            continue
        order_models = _sort_models_for_metric(df[df["Metric"] == metric], present_models, metric)

        # positions
        y = np.arange(len(order_models))
        height = 0.8
        width = 0.38  # per dataset
        offsets = np.linspace(-width/2, width/2, num=len(DATASETS_ORDER))

        # draw per dataset
        for i, dataset in enumerate(DATASETS_ORDER):
            sub = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]
            vals = [float(sub[sub["Model"] == m]["Value"].mean()) if not sub[sub["Model"] == m].empty else np.nan for m in order_models]
            colors = [_model_color(m, palette_fb) for m in order_models]
            bars = ax.barh(y + offsets[i], vals, height=width, color=colors, edgecolor="black", linewidth=0.6,
                           hatch=HATCH_BY_DATASET.get(dataset, None), alpha=0.95, label=dataset)
            # annotate values
            for rect, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(rect.get_width() + 0.01 if metric != "Log Loss" else rect.get_width() + 0.02,
                            rect.get_y() + rect.get_height()/2,
                            f"{v:.3f}", va="center", ha="left", fontsize=9)

        # aesthetics
        ax.set_yticks(y)
        ax.set_yticklabels([_display_model_name(m) for m in order_models])
        ax.set_title(panel_name)
        ax.grid(axis="x", alpha=0.25)
        rng = METRIC_RANGES.get(metric)
        if rng is not None:
            ax.set_xlim(*rng)
        if metric == "Log Loss":
            ax.set_xlabel("Log Loss (lower is better)")
        else:
            ax.set_xlabel(f"{metric}")

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=min(3, len(labels)), frameon=True)
        fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    else:
        fig.tight_layout(rect=[0, 0.08, 1, 0.98])

    fig.suptitle(f"{metric} — Classical vs Advanced", y=0.995)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"bars_{metric.replace(' ', '_').lower()}__grouped.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"bars_{metric.replace(' ', '_').lower()}__grouped.pdf", bbox_inches="tight")
    plt.close(fig)


def _draw_rank_bars(df: pd.DataFrame, outdir: Path) -> None:
    # optional plot if 'Rank' exists
    if "Rank" not in set(df["Metric"].unique()):
        return
    metric = "Rank"
    palette_fb = sns.color_palette("colorblind", n_colors=12)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.2), sharex=False, sharey=False)
    for ax, (panel_name, panel_models) in zip(axes, FAMILY_PANELS.items()):
        present_models = [m for m in panel_models if m in set(df["Model"]) ]
        if not present_models:
            ax.axis("off")
            continue
        # For rank: lower is better; sort ascending by average rank
        sub = df[df["Metric"] == metric]
        avg_rank = sub.groupby("Model")["Value"].mean().to_dict()
        order_models = sorted(present_models, key=lambda m: (avg_rank.get(m, np.inf), _sanitize_model_name(m)))
        y = np.arange(len(order_models))
        width = 0.38
        offsets = np.linspace(-width/2, width/2, num=len(DATASETS_ORDER))
        for i, dataset in enumerate(DATASETS_ORDER):
            s = sub[sub["Dataset"] == dataset]
            vals = [float(s[s["Model"] == m]["Value"].mean()) if not s[s["Model"] == m].empty else np.nan for m in order_models]
            colors = [_model_color(m, palette_fb) for m in order_models]
            bars = ax.barh(y + offsets[i], vals, height=width, color=colors, edgecolor="black", linewidth=0.6,
                           hatch=HATCH_BY_DATASET.get(dataset, None), alpha=0.95, label=dataset)
            for rect, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(rect.get_width() + 0.05, rect.get_y() + rect.get_height()/2, f"{int(v)}", va="center", ha="left", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([_display_model_name(m) for m in order_models])
        ax.set_title(panel_name)
        ax.set_xlabel("Rank (lower is better)")
        ax.grid(axis="x", alpha=0.25)
        ax.invert_xaxis()  # visually emphasize lower is better
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=min(3, len(labels)), frameon=True)
        fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    else:
        fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    fig.suptitle("Rank — Classical vs Advanced", y=0.995)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "bars_rank__grouped.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "bars_rank__grouped.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Single-page multi-metric grid (all metrics in one figure)
# ------------------------------------------------------------
def _draw_all_metrics_grid(df: pd.DataFrame, outdir: Path) -> None:
    metrics_present = [m for m in METRICS_ORDER if m in set(df["Metric"])]
    if "Rank" in set(df["Metric"]):
        metrics_present.append("Rank")
    if not metrics_present:
        return

    nrows = len(metrics_present)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, max(2.2 * nrows, 8.0)), sharex=False, sharey=False)
    if nrows == 1:
        axes = np.array([axes])

    palette_fb = sns.color_palette("colorblind", n_colors=12)

    # Top-row titles
    axes[0, 0].set_title("Classical baselines")
    axes[0, 1].set_title("Advanced models")

    for r, metric in enumerate(metrics_present):
        for c, (panel_name, panel_models) in enumerate(FAMILY_PANELS.items()):
            ax = axes[r, c]
            present_models = [m for m in panel_models if m in set(df["Model"]) ]
            if not present_models:
                ax.axis("off"); continue
            sub_metric = df[df["Metric"] == metric]
            if metric == "Rank":
                # For Rank, lower is better
                avg_rank = sub_metric.groupby("Model")["Value"].mean().to_dict()
                order_models = sorted(present_models, key=lambda m: (avg_rank.get(m, np.inf), _sanitize_model_name(m)))
            else:
                order_models = _sort_models_for_metric(sub_metric, present_models, metric)

            y = np.arange(len(order_models))
            width = 0.34
            offsets = np.linspace(-width/2, width/2, num=len(DATASETS_ORDER))

            for i, dataset in enumerate(DATASETS_ORDER):
                s = sub_metric[sub_metric["Dataset"] == dataset]
                vals = [float(s[s["Model"] == m]["Value"].mean()) if not s[s["Model"] == m].empty else np.nan for m in order_models]
                colors = [_model_color(m, palette_fb) for m in order_models]
                bars = ax.barh(y + offsets[i], vals, height=width, color=colors, edgecolor="black", linewidth=0.5,
                               hatch=HATCH_BY_DATASET.get(dataset, None), alpha=0.95, label=dataset)
                # compact annotations
                for rect, v in zip(bars, vals):
                    if np.isfinite(v):
                        txt = f"{int(v)}" if metric == "Rank" else f"{v:.3f}"
                        ax.text(rect.get_width() + (0.03 if metric == "Rank" else 0.01),
                                rect.get_y() + rect.get_height()/2,
                                txt, va="center", ha="left", fontsize=7)

            # axis cosmetics
            if c == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([_display_model_name(m) for m in order_models], fontsize=9)
            else:
                ax.set_yticks(y)
                ax.set_yticklabels([_display_model_name(m) for m in order_models], fontsize=9)
            rng = METRIC_RANGES.get(metric)
            if rng is not None and metric != "Rank":
                ax.set_xlim(*rng)
            if metric == "Log Loss":
                ax.set_xlabel("Log Loss (lower is better)", fontsize=9)
            elif metric == "Rank":
                ax.set_xlabel("Rank (lower is better)", fontsize=9)
                ax.invert_xaxis()
            else:
                ax.set_xlabel(metric, fontsize=9)
            ax.grid(axis="x", alpha=0.25)
            ax.set_title(metric if r == 0 else "", fontsize=10)

    # Shared legend for datasets at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=min(3, len(labels)), frameon=True)
        fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    else:
        fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    fig.suptitle("Benchmark summary — all metrics (Classical vs Advanced)", y=0.995)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "bars_all_metrics__single_page.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "bars_all_metrics__single_page.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Single-page grid WITHOUT grouping (all models in each subplot)
# ------------------------------------------------------------
def _draw_all_metrics_grid_unified(df: pd.DataFrame, outdir: Path) -> None:
    metrics_present = [m for m in METRICS_ORDER if m in set(df["Metric"])]
    if "Rank" in set(df["Metric"]):
        metrics_present.append("Rank")
    if not metrics_present:
        return

    # 3x2 grid to fit 6 subplots on a single page
    n = len(metrics_present)
    nrows, ncols = (3, 2) if n >= 6 else (int(np.ceil(n/2)), 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(9.0, 3.2 * nrows)), sharex=False, sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    palette_fb = sns.color_palette("colorblind", n_colors=12)

    # Determine global list of models present in data for stability
    all_models = sorted(df["Model"].unique(), key=lambda m: _sanitize_model_name(m).lower())

    def order_models_for_metric(metric: str) -> List[str]:
        sub = df[df["Metric"] == metric]
        if metric in {"Log Loss", "Rank"}:
            agg = sub.groupby("Model")["Value"].mean().to_dict()
            return sorted(all_models, key=lambda m: (agg.get(m, np.inf), _sanitize_model_name(m).lower()))
        else:
            agg = sub.groupby("Model")["Value"].mean().to_dict()
            return sorted(all_models, key=lambda m: (-(agg.get(m, -np.inf)), _sanitize_model_name(m).lower()))

    for idx, metric in enumerate(metrics_present):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        order_models = [m for m in order_models_for_metric(metric) if m in set(df["Model"]) ]
        y = np.arange(len(order_models))
        width = 0.34
        offsets = np.linspace(-width/2, width/2, num=len(DATASETS_ORDER))

        sub_metric = df[df["Metric"] == metric]
        for i, dataset in enumerate(DATASETS_ORDER):
            s = sub_metric[sub_metric["Dataset"] == dataset]
            vals = [float(s[s["Model"] == m]["Value"].mean()) if not s[s["Model"] == m].empty else np.nan for m in order_models]
            colors = [_model_color(m, palette_fb) for m in order_models]
            bars = ax.barh(y + offsets[i], vals, height=width, color=colors, edgecolor="black", linewidth=0.5,
                           hatch=HATCH_BY_DATASET.get(dataset, None), alpha=0.95, label=dataset)
            for rect, v in zip(bars, vals):
                if np.isfinite(v):
                    txt = f"{int(v)}" if metric == "Rank" else f"{v:.3f}"
                    ax.text(rect.get_width() + (0.03 if metric == "Rank" else 0.01),
                            rect.get_y() + rect.get_height()/2, txt, va="center", ha="left", fontsize=7)

        # Axes cosmetics
        ax.set_yticks(y)
        # Show ytick labels only on left column to save space
        if c == 0:
            ax.set_yticklabels([_display_model_name(m) for m in order_models], fontsize=9)
        else:
            ax.set_yticklabels(["" for _ in order_models])

        rng = METRIC_RANGES.get(metric)
        if rng is not None and metric != "Rank":
            ax.set_xlim(*rng)
        if metric == "Log Loss":
            ax.set_xlabel("Log Loss (lower is better)", fontsize=9)
        elif metric == "Rank":
            ax.set_xlabel("Rank (lower is better)", fontsize=9)
            ax.invert_xaxis()
        else:
            ax.set_xlabel(metric, fontsize=9)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(metric, fontsize=10)

    # Remove any unused axes
    total_axes = nrows * ncols
    for j in range(n, total_axes):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r, c])

    # Legend once at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles and ncols > 1:
        handles, labels = axes[0, 1].get_legend_handles_labels()
    outdir.mkdir(parents=True, exist_ok=True)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(labels)), frameon=True)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    else:
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    fig.suptitle("Benchmark summary — all metrics (All models, two datasets)", y=0.995)
    fig.savefig(outdir / "bars_all_metrics__single_page_unified.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "bars_all_metrics__single_page_unified.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# One-figure colorful heatmap (ALL models x ALL metrics for both datasets)
# ------------------------------------------------------------
def _draw_single_heatmap_unified(df: pd.DataFrame, outdir: Path) -> None:
    # Columns: MultiIndex (Metric, Dataset) in a fixed order
    metrics_present = [m for m in METRICS_ORDER if m in set(df["Metric"])]
    if "Rank" in set(df["Metric"]):
        metrics_present.append("Rank")
    if not metrics_present:
        return

    # Build pivot with full set of models
    models = sorted(df["Model"].unique(), key=lambda m: _sanitize_model_name(m).lower())
    cols = []
    for metric in metrics_present:
        for dataset in DATASETS_ORDER:
            cols.append((metric, dataset))

    # Fill wide table
    tbl = pd.DataFrame(index=models, columns=pd.MultiIndex.from_tuples(cols, names=["Metric", "Dataset"]))
    for (metric, dataset) in tbl.columns:
        s = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]
        mvals = s.set_index("Model")["Value"].to_dict()
        for m in models:
            tbl.loc[m, (metric, dataset)] = mvals.get(m, np.nan)

    # Prepare a normalized copy for colormap scaling (0..1 within each metric)
    norm_tbl = tbl.copy().astype(float)
    for metric in metrics_present:
        sub = norm_tbl[metric]
        # metric-wise min-max, with inversion for loss/rank
        arr = sub.to_numpy(dtype=float)
        mmin = np.nanmin(arr)
        mmax = np.nanmax(arr)
        if not np.isfinite(mmin) or not np.isfinite(mmax) or mmax == mmin:
            # avoid division by zero; set to 0.5
            norm_tbl[metric] = 0.5
            continue
        scaled = (sub - mmin) / (mmax - mmin)
        if metric in {"Log Loss", "Rank"}:  # lower is better
            scaled = 1.0 - scaled
        norm_tbl[metric] = scaled

    # Sort models by average of normalized scores across all columns
    model_order = norm_tbl.mean(axis=1).sort_values(ascending=False).index.tolist()
    tbl = tbl.loc[model_order]
    norm_tbl = norm_tbl.loc[model_order]

    # Create a single heatmap figure
    fig_h, ax_h = plt.subplots(figsize=(14, max(6.5, 0.48 * len(model_order) + 2)))

    # Compose a simple string column header: f"{metric}\n{dataset}"
    display_tbl = norm_tbl.copy()
    display_tbl.columns = [f"{m}\n{d}" for m, d in display_tbl.columns]

    # Annotations with original values
    annot_tbl = tbl.copy()
    def fmt_val(m: str, v: float) -> str:
        if pd.isna(v):
            return ""
        if m == "Rank":
            try:
                return f"{int(round(float(v)))}"
            except Exception:
                return f"{v}"
        else:
            return f"{float(v):.4f}"
    annot_df = pd.DataFrame({
        f"{m}\n{d}": [fmt_val(m, annot_tbl.loc[mdl, (m, d)]) for mdl in annot_tbl.index]
        for (m, d) in annot_tbl.columns
    }, index=annot_tbl.index)

    # Heatmap
    cmap = sns.color_palette("Spectral", as_cmap=True)
    sns.heatmap(
        display_tbl,
        ax=ax_h,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        annot=annot_df,
        fmt="",
        annot_kws={"fontsize": 7.5, "color": "black"},
        cbar_kws={"label": "Normalized performance (per metric)"},
        linewidths=0.4,
        linecolor="white",
        square=False,
    )
    ax_h.set_ylabel("")
    ax_h.set_xlabel("")
    ax_h.set_yticklabels([_display_model_name(m) for m in model_order], rotation=0)
    ax_h.set_title("All models × all metrics (two datasets) — higher colors are better\n(Log Loss and Rank normalized with lower=better)", pad=12)

    outdir.mkdir(parents=True, exist_ok=True)
    fig_h.tight_layout()
    fig_h.savefig(outdir / "all_metrics__heatmap_unified.png", dpi=300, bbox_inches="tight")
    fig_h.savefig(outdir / "all_metrics__heatmap_unified.pdf", bbox_inches="tight")
    plt.close(fig_h)

# ------------------------------------------------------------
# One-figure ranked heatmap: colors by rank (1=best .. N=worst)
# ------------------------------------------------------------
def _draw_single_heatmap_ranked(df: pd.DataFrame, outdir: Path) -> None:
    metrics_present = [m for m in METRICS_ORDER if m in set(df["Metric"])]
    if "Rank" in set(df["Metric"]):
        metrics_present.append("Rank")
    if not metrics_present:
        return

    models = sorted(df["Model"].unique(), key=lambda m: _sanitize_model_name(m).lower())
    cols = [(m, d) for m in metrics_present for d in DATASETS_ORDER]

    # wide table of values
    tbl = pd.DataFrame(index=models, columns=pd.MultiIndex.from_tuples(cols, names=["Metric", "Dataset"]))
    for (metric, dataset) in tbl.columns:
        s = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]
        mvals = s.set_index("Model")["Value"].to_dict()
        for m in models:
            tbl.loc[m, (metric, dataset)] = mvals.get(m, np.nan)
    tbl = tbl.astype(float)

    # compute per-column ranks (1 best)
    rank_tbl = pd.DataFrame(index=models, columns=tbl.columns)
    for (metric, dataset) in tbl.columns:
        col = tbl[(metric, dataset)].astype(float)
        ascending = metric in {"Log Loss", "Rank"}  # lower is better
        # Dense ranking: ties share a rank and the next rank increments by 1 (no gaps)
        ranks = col.rank(method="dense", ascending=ascending)
        ranks = ranks.fillna(len(models))
        rank_tbl[(metric, dataset)] = ranks

    # Fixed display order (top to bottom)
    FIXED_MODEL_ORDER = ["TIRT", "LogisticRegression", "DKT", "BKT", "FKT", "AdaptKT", "CLKT", "GKT", "Rasch 1PL"]
    order = [m for m in FIXED_MODEL_ORDER if m in models]
    # Append any models not in the fixed list at the end
    order += [m for m in models if m not in order]
    rank_tbl = rank_tbl.loc[order]
    tbl = tbl.loc[order]

    # display/annotation tables
    disp = rank_tbl.copy()
    # Show full metric and dataset, but split dataset into brand and year on separate lines
    def _split_dataset_two_lines(ds: str) -> str:
        import re
        mobj = re.match(r"^(.*?)[\s\-]*(\d{2}\-\d{2}|\d{4})$", ds)
        if mobj:
            brand = mobj.group(1).strip()
            year = mobj.group(2).strip()
            return f"{brand}\n{year}"
        # fallback: keep as one line
        return ds
    disp.columns = [f"{m}\n{_split_dataset_two_lines(d)}" for m, d in disp.columns]
    n_models = len(order)
    annot_df = pd.DataFrame({
        f"{m}\n{d}": [
            (
                ""
                if pd.isna(tbl.loc[mdl, (m, d)])
                else f"{tbl.loc[mdl, (m, d)]:.4f}\nRank {int(n_models - rank_tbl.loc[mdl, (m, d)] + 1)}"
            )
            for mdl in rank_tbl.index
        ]
        for (m, d) in rank_tbl.columns
    }, index=rank_tbl.index)

    # Red→Blue gradient, rank 1 = red (best), worst = blue
    colors_by_rank = sns.color_palette("RdBu", n_colors=n_models)
    cmap = ListedColormap(colors_by_rank)
    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(1, n_models + 2), ncolors=n_models)

    # Wide landscape for readability; increase height scaling so bigger annotations fit
    fig, ax = plt.subplots(figsize=(36, max(12.0, 0.8 * n_models + 2)))
    sns.heatmap(
        disp,
        ax=ax,
        cmap=cmap,
        norm=norm,
        annot=annot_df,
        fmt="",
        # Bigger, bold in-box details for readability
        annot_kws={"fontsize": 23, "fontweight": "bold", "color": "black"},
        cbar_kws={"label": "Rank"},
        linewidths=0.4,
        linecolor="white",
        square=False,
    )
    # Make colorbar ticks and label large and bold (match ~19pt axis text)
    try:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_label("Rank", fontsize=19, fontweight="bold")
            cbar.ax.tick_params(labelsize=19)
            for lbl in cbar.ax.get_yticklabels():
                lbl.set_fontweight("bold")
            cbar.ax.invert_yaxis()  # Flip: Rank 1 at top, highest rank at bottom
    except Exception:
        pass
    # Horizontal x labels (three-line max), bigger and bold
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=19, fontweight="bold")
    ax.tick_params(axis="x", which="major", pad=26)
    # Larger, bold y labels
    ax.set_yticklabels([_display_model_name(m) for m in order], rotation=0, fontsize=19, fontweight="bold")
    # Emphasize top-3 per column with visible black borders (ties included)
    for c_idx, col_name in enumerate(disp.columns):
        col_ranks = disp[col_name].astype(float).values
        for r_idx, rk in enumerate(col_ranks):
            if np.isfinite(rk) and rk <= 3:
                lw = 2.2 if rk == 1 else 1.8 if rk == 2 else 1.5
                rect = Rectangle((c_idx, r_idx), 1, 1, fill=False, edgecolor="black", linewidth=lw, joinstyle="miter", capstyle="projecting", zorder=3)
                ax.add_patch(rect)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticklabels([_display_model_name(m) for m in order], rotation=0)

    outdir.mkdir(parents=True, exist_ok=True)
    # Reserve more bottom margin for multi-line x tick labels
    plt.subplots_adjust(bottom=0.45)
    fig.tight_layout()
    # Save at very high resolution
    fig.savefig(outdir / "all_metrics__heatmap_ranked.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "all_metrics__heatmap_ranked.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Ranked, sorted-per-column mini-heatmaps (each column independently sorted)
# ------------------------------------------------------------
def _draw_ranked_sorted_per_column(df: pd.DataFrame, outdir: Path) -> None:
    metrics_present = [m for m in METRICS_ORDER if m in set(df["Metric"])]
    if "Rank" in set(df["Metric"]):
        metrics_present.append("Rank")
    if not metrics_present:
        return

    models = sorted(df["Model"].unique(), key=lambda m: _sanitize_model_name(m).lower())
    n_models = len(models)

    # Red→Blue gradient for rank (1=best red → worst blue)
    colors_by_rank = sns.color_palette("RdBu", n_colors=n_models)
    cmap = ListedColormap(colors_by_rank)
    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(1, n_models + 2), ncolors=n_models)

    # columns to render
    cols = [(m, d) for m in metrics_present for d in DATASETS_ORDER]
    total = len(cols)
    ncols = min(6, total)
    nrows = int(np.ceil(total / ncols))
    fig_w = max(12, 2.2 * ncols)
    fig_h = max(6.0, 0.35 * n_models * nrows + 1.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (metric, dataset) in enumerate(cols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        s = df[(df["Metric"] == metric) & (df["Dataset"] == dataset)]
        col = s.set_index("Model")["Value"].reindex(models)
        ascending = metric in {"Log Loss", "Rank"}
        # Dense ranking avoids gaps after ties: 1,2,2,3 (not 1,2,2,4)
        ranks = col.rank(method="dense", ascending=ascending)
        ranks = ranks.fillna(n_models)
        # order by rank ascending for this column
        order = ranks.sort_values().index.tolist()
        disp = pd.DataFrame(ranks.loc[order].astype(float))
        disp.columns = [f"{metric}\n{dataset}"]
        # annotations: show #rank + value
        def atext(m):
            v = col.loc[m]
            rk = int(ranks.loc[m]) if pd.notna(ranks.loc[m]) else n_models
            disp_rk = int(n_models - rk + 1)
            if pd.isna(v):
                return f"Rank {disp_rk}"
            if metric == "Rank":
                try:
                    vtxt = f"{int(round(float(v)))}"
                except Exception:
                    vtxt = str(v)
            else:
                vtxt = f"{float(v):.4f}"
            return f"Rank {disp_rk}\n{vtxt}"
        annot = pd.DataFrame({disp.columns[0]: [atext(m) for m in order]}, index=order)

        sns.heatmap(
            disp,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 7.0, "color": "black"},
            cbar=False,
            linewidths=0.4,
            linecolor="white",
            square=False,
        )
        # y labels only on first column of each row
        if c == 0:
            ax.set_yticklabels([_display_model_name(m) for m in order], rotation=0, fontsize=8)
        else:
            ax.set_yticklabels(["" for _ in order])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # remove any empty axes
    for j in range(total, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r, c])

    # shared colorbar
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label("Rank (1=best)")

    fig.tight_layout(rect=[0.02, 0.02, 0.9, 0.98])
    outdir.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Per-column sorted ranks (each subplot independently sorted)", y=0.995)
    fig.savefig(outdir / "all_metrics__ranked_sorted_per_column.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "all_metrics__ranked_sorted_per_column.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Innovative grouped bar charts for benchmark metrics")
    parser.add_argument("--input", type=str, default=None, help="Path to long/tidy CSV: Model,Metric,Dataset,Value")
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parent / "plots" / "innovative_bars"), help="Output directory for figures")
    args = parser.parse_args()

    df = load_long_df(args.input)

    # Ensure order columns are categorical for stable plotting
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=DATASETS_ORDER, ordered=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Only generate the ranked, medal-colored heatmap figure
    _draw_single_heatmap_ranked(df, outdir)


if __name__ == "__main__":
    main()
