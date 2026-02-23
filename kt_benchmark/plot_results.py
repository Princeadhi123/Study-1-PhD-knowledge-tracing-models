from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve

from . import config
from .utils import load_itemwise_df, add_time_index


def name_to_tag(name: str) -> str:
    # mimic run_benchmark.py tagging
    return name.lower().replace(" ", "_").replace("/", "-")


def _sanitize_model_name(name: str) -> str:
    """Mirror the name normalization in run_benchmark._sanitize_model_name so
    plots can find the correct preds files regardless of '(minimal)', '(CORAL)', or 'lite' suffixes.
    """
    import re
    s = name
    s = re.sub(r"\s*\((?:minimal|coral)\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\blite\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip("-_ ")
    return s


def _display_model_name(name: str) -> str:
    """Short, presentation-friendly model display names.
    Currently maps 'LogisticRegression' -> 'LR'. Falls back to sanitized name.
    """
    s = _sanitize_model_name(name)
    key = s.lower()
    mapping = {
        "logisticregression": "LR",
    }
    return mapping.get(key, s)


def _ensure_plot_dirs(base: Path) -> Dict[str, Path]:
    plots_dir = base / "plots"
    per_model_dir = plots_dir / "per_model"
    summary_dir = plots_dir / "summary"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    return {"base": plots_dir, "per_model": per_model_dir, "summary": summary_dir}


def _load_summary(outdir: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    metrics_csv = outdir / "metrics_summary.csv"
    details_json = outdir / "details.json"
    metrics = pd.read_csv(metrics_csv)
    if details_json.exists():
        details = json.loads(details_json.read_text(encoding="utf-8"))
    else:
        details = {}
    return metrics, details


def _load_predictions(outdir: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    tag = name_to_tag(model_name)
    p = outdir / f"preds_{tag}.csv"
    if not p.exists():
        # Some names include parentheses in tag; try a looser search
        candidates = list(outdir.glob(f"preds_{tag}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No prediction file for model '{model_name}' (expected {p})")
        p = candidates[0]
    df = pd.read_csv(p)
    y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy()
    y_prob = pd.to_numeric(df["y_prob"], errors="coerce").to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    return y_true[mask].astype(int), y_prob[mask].astype(float)


def _load_predictions_with_index(outdir: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions and optional original row indices if present."""
    tag = name_to_tag(model_name)
    # Prefer the viz file if present for fuller coverage in visualizations
    p_viz = outdir / f"preds_{tag}__viz.csv"
    p = p_viz if p_viz.exists() else (outdir / f"preds_{tag}.csv")
    if not p.exists():
        candidates = list(outdir.glob(f"preds_{tag}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No prediction file for model '{model_name}' (expected {p})")
        # Prefer a candidate ending with __viz if available
        viz_candidates = [c for c in candidates if c.name.endswith("__viz.csv")]
        p = viz_candidates[0] if viz_candidates else candidates[0]
    df = pd.read_csv(p)
    y_true = pd.to_numeric(df.get("y_true"), errors="coerce").to_numpy()
    y_prob = pd.to_numeric(df.get("y_prob"), errors="coerce").to_numpy()
    idx = pd.to_numeric(df.get("df_row_index"), errors="coerce").to_numpy() if "df_row_index" in df.columns else None
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if idx is not None:
        mask = mask & np.isfinite(idx)
        return y_true[mask].astype(int), y_prob[mask].astype(float), idx[mask].astype(int)
    else:
        return y_true[mask].astype(int), y_prob[mask].astype(float), np.array([], dtype=int)


def _collect_model_predictions_with_dfrows(metrics: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {model: {y_true, y_prob, rows}} only for models that have row indices stored."""
    out = {}
    for _, row in metrics.iterrows():
        name = row["model"]
        try:
            yt, yp, idx = _load_predictions_with_index(config.OUTPUT_DIR, name)
            if idx.size > 0:
                out[name] = {"y_true": yt, "y_prob": yp, "rows": idx}
        except Exception:
            continue
    return out


def plot_student_trajectories_all_models(
    metrics: pd.DataFrame,
    outdir: Path,
    max_students: int = 2,
    min_len: int = 30,
    max_len: int = 60,
    max_models: int = 2,
    models: Optional[List[str]] = None,
):
    """
    For the current dataset, pick up to max_students students and plot trajectories of
    predicted probabilities over time for ALL models in one figure (dataset-wise).
    Correct vs. incorrect points highlighted by marker face color (green=correct, red=wrong) per model.
    """
    df = add_time_index(load_itemwise_df())
    if config.COL_ID not in df.columns or config.COL_ORDER not in df.columns:
        return
    model_preds = _collect_model_predictions_with_dfrows(metrics)
    if not model_preds:
        return
    # Map df row index -> (student, order)
    id_map = df[config.COL_ID].astype(str)
    order_map = df[config.COL_ORDER]

    # Candidate students: limit by interaction counts in dataset
    sizes = df.groupby(df[config.COL_ID].astype(str))[config.COL_ID].size()
    allowed = set(sizes[(sizes >= min_len) & (sizes <= max_len)].index.astype(str).tolist())

    # If user provided explicit models, prefer students present in ALL of them
    intersection_students: Optional[set] = None
    if models:
        models_norm = [_sanitize_model_name(m) for m in models]
        present_sets = []
        for mdl in models_norm:
            if mdl in model_preds:
                sids = set(id_map.loc[model_preds[mdl]["rows"]].astype(str))
                present_sets.append(sids)
        if present_sets:
            inter_full = set.intersection(*present_sets)
            inter_allowed = inter_full.intersection(allowed)
            intersection_students = inter_allowed if inter_allowed else inter_full

    # Score by overlap coverage across models
    sid_counts: Dict[str, int] = {}
    for m in model_preds.values():
        sids = id_map.loc[m["rows"]].astype(str)
        for s in sids:
            if s in allowed:
                sid_counts[s] = sid_counts.get(s, 0) + 1

    if intersection_students:
        # Choose from intersection first, sorted by coverage (desc)
        top_students = [s for s, _ in sorted(((s, sid_counts.get(s, 0)) for s in intersection_students), key=lambda x: -x[1])][:max_students]
        # If intersection too small, pad with top by coverage
        if len(top_students) < max_students:
            remainder = [s for s, _ in sorted(sid_counts.items(), key=lambda x: -x[1]) if s not in top_students]
            top_students += remainder[: (max_students - len(top_students))]
    else:
        top_students = [s for s, _ in sorted(sid_counts.items(), key=lambda x: -x[1])][:max_students]
    if not top_students:
        return

    # Choose models: explicit list from caller/config if provided, else top by metric order
    if models:
        # Normalize provided names to match saved/sanitized keys and include ALL provided
        models_norm = [_sanitize_model_name(m) for m in models]
        model_list = [m for m in models_norm if m in model_preds]
        if not model_list:
            ordered_models = [row["model"] for _, row in metrics.iterrows() if row["model"] in model_preds]
            model_list = ordered_models if ordered_models else list(model_preds.keys())
    else:
        ordered_models = [row["model"] for _, row in metrics.iterrows() if row["model"] in model_preds]
        model_list = ordered_models[:max_models] if ordered_models else list(model_preds.keys())[:max_models]
    # Highly distinct colors across hues (one unique hue per model)
    colors = sns.color_palette("husl", n_colors=len(model_list) if len(model_list) > 0 else 8)
    linestyles = ["solid"]  # keep all lines solid for easier tracing
    color_by_model = {mdl: colors[i % len(colors)] for i, mdl in enumerate(model_list)}
    style_by_model = {mdl: linestyles[i % len(linestyles)] for i, mdl in enumerate(model_list)}
    # Per-line markers for additional distinguishability
    markers = ["D", "o", "s", "^", "v", "P", "X", "*", "<", ">"]
    marker_by_model = {mdl: markers[i % len(markers)] for i, mdl in enumerate(model_list)}

    n = len(top_students)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 2.7 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, top_students):
        # Ground truth points: show ONLY test rows available for any selected model for this student
        # This makes the dots end where the model lines end.
        rows_union_list = []
        for mdl in model_list:
            data = model_preds[mdl]
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if mask_sid.any():
                rows_union_list.append(data["rows"][mask_sid])
        if rows_union_list:
            rows_union = np.unique(np.concatenate(rows_union_list))
            ord_u = order_map.loc[rows_union].to_numpy()
            sort_idx = np.argsort(ord_u)
            rows_union = rows_union[sort_idx]
            ord_true = order_map.loc[rows_union].to_numpy()
            gt_vals = pd.to_numeric(df.loc[rows_union, config.COL_RESP], errors="coerce").to_numpy()
            m = np.isin(gt_vals, [0, 1]) & np.isfinite(ord_true)
            ax.scatter(
                ord_true[m],
                gt_vals[m].astype(int),
                s=35,
                c="black",
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
                label="Ground truth",
            )
        else:
            # Fallback: if no test rows found (unlikely), skip dots for this student
            pass

        # Model prediction lines (thicker, higher-contrast)
        for mdl in model_list:
            data = model_preds[mdl]
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if not mask_sid.any():
                continue
            rows_sid = data["rows"][mask_sid]
            ord_sid = order_map.loc[rows_sid].to_numpy()
            order_idx = np.argsort(ord_sid)
            ord_sid = ord_sid[order_idx]
            yp = data["y_prob"][mask_sid][order_idx]
            ax.plot(
                ord_sid + _x_jitter_for_model(mdl),
                yp,
                color=color_by_model[mdl],
                lw=2.2,
                alpha=0.9,
                linestyle=style_by_model[mdl],
                marker=marker_by_model[mdl],
                markevery=5,
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=color_by_model[mdl],
                markeredgewidth=0.9,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
                label=_display_model_name(mdl),
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel("p(correct)")
        ax.set_title(f"Student {sid}")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.02)
    axes[-1].set_xlabel("Interaction order", labelpad=0)
    # Legend just below all plots (outside axes) to save space
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01),
                   ncol=min(8, len(labels)), fontsize=9, frameon=True)
        fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    else:
        fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.suptitle("Example Student Trajectories", y=0.985)
    fig.savefig(outdir / "trajectories_all_models.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "trajectories_all_models.pdf", bbox_inches="tight")
    plt.close(fig)


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


def _x_jitter_for_model(name: str) -> float:
    """Small x-offset to visually separate nearly overlapping lines.
    We only jitter the two most similar classical baselines in our plots:
    - LogisticRegression: -0.12
    - CLKT: -0.12
    - TIRT: +0.12
    All others: 0.0
    """
    key = _sanitize_model_name(name).lower()
    if key == "logisticregression":
        return -0.12
    if key == "clkt":
        return -0.12
    if key == "tirt":
        return +0.12
    return 0.0


def plot_student_trajectories_grouped(
    metrics: pd.DataFrame,
    outdir: Path,
    max_students: int = 4,
    min_len: int = 20,
    max_len: int = 60,
):
    """
    Plot less cluttered trajectories by splitting models into two panels:
      Panel A: classical baselines (BKT, Rasch1PL, LogisticRegression, TIRT)
      Panel B: advanced models (DKT, FKT, CLKT, AdaptKT, GKT)
    Also: larger ground-truth dots and light-grey shading for incorrect answers.
    Produces two figures:
      (1) grouped_single_student: two stacked panels for the best-covered student
      (2) grouped_multi_students_[classical|advanced]: 2x2 grid (up to 4 students) per panel
    """
    df = add_time_index(load_itemwise_df())
    if config.COL_ID not in df.columns or config.COL_ORDER not in df.columns:
        return
    model_preds = _collect_model_predictions_with_dfrows(metrics)
    if not model_preds:
        return

    # Maps from df rows to (student, order)
    id_map = df[config.COL_ID].astype(str)
    order_map = df[config.COL_ORDER]

    # Candidate students by length
    sizes = df.groupby(df[config.COL_ID].astype(str))[config.COL_ID].size()
    allowed = set(sizes[(sizes >= min_len) & (sizes <= max_len)].index.astype(str).tolist())

    # Coverage-based selection
    sid_counts: Dict[str, int] = {}
    for m in model_preds.values():
        sids = id_map.loc[m["rows"]].astype(str)
        for s in sids:
            if s in allowed:
                sid_counts[s] = sid_counts.get(s, 0) + 1
    top_students = [s for s, _ in sorted(sid_counts.items(), key=lambda x: -x[1])][:max_students]
    if not top_students:
        return

    # Define groups (use sanitized keys, then map back to existing model names in preds)
    group_defs = {
        "Classical baselines": ["BKT", "Rasch1PL", "LogisticRegression", "TIRT"],
        "Advanced models": ["DKT", "FKT", "CLKT", "AdaptKT", "GKT"],
    }
    # map sanitized -> original key present in model_preds
    key_by_sanitized = {_sanitize_model_name(k): k for k in model_preds.keys()}

    # Colors and styles
    # Fixed high-contrast colors per model (sanitized key, lowercase)
    fixed_colors = {
        "dkt": "#CC79A7",       # magenta
        "fkt": "#009E73",       # green
        "clkt": "#0072B2",      # blue
        "adaptkt": "#D55E00",   # dark orange (high contrast)
        "gkt": "#E69F00",       # orange/yellow
        "bkt": "#56B4E9",       # light blue
        "rasch1pl": "#0072B2",  # blue (labelled as IRT)
        "logisticregression": "#009E73",  # green
        "tirt": "#D55E00",      # dark orange
    }
    # Fallback palette for any extra models
    all_models_ordered = [k for k in model_preds.keys()]  # preserve metric sort order
    palette = sns.color_palette("colorblind", n_colors=max(8, len(all_models_ordered)))
    color_by_model = {}
    for i, k in enumerate(all_models_ordered):
        key = _sanitize_model_name(k).lower()
        color_by_model[k] = fixed_colors.get(key, palette[i % len(palette)])

    def linestyle_for(name: str) -> str:
        fam = _family_of_model(name)
        if fam == "deep":
            return "solid"
        if fam == "classical":
            return "dashed"
        if fam == "hybrid":
            return "dotted"
        if fam == "graph":
            return "dashdot"
        return "solid"

    markers = ["D", "o", "s", "^", "v", "P", "X", "<", ">"]

    def _plot_panel(ax: plt.Axes, sid: str, panel_name: str, mdl_sanitized_list: List[str]):
        # pick models present
        mdl_keys = [key_by_sanitized[m] for m in mdl_sanitized_list if m in key_by_sanitized]
        if not mdl_keys:
            ax.axis("off")
            return
        # Ground-truth: union of test rows across these models for this student
        rows_union_list = []
        for k in mdl_keys:
            data = model_preds[k]
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if mask_sid.any():
                rows_union_list.append(data["rows"][mask_sid])
        if rows_union_list:
            rows_union = np.unique(np.concatenate(rows_union_list))
            ords = order_map.loc[rows_union].to_numpy()
            args = np.argsort(ords)
            rows_union = rows_union[args]
            ords = ords[args]
            gt_vals = pd.to_numeric(df.loc[rows_union, config.COL_RESP], errors="coerce").to_numpy()
            # Shade incorrect (0) bands lightly
            if len(ords) > 0:
                for o, g in zip(ords, gt_vals):
                    if g == 0:
                        ax.axvspan(float(o) - 0.5, float(o) + 0.5, color="#000000", alpha=0.08, zorder=0)
            m = np.isin(gt_vals, [0, 1]) & np.isfinite(ords)
            ax.scatter(
                ords[m],
                gt_vals[m].astype(int),
                s=48,  # slightly larger
                c="black",
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
                label="Ground truth",
            )
        # Model lines
        # Debug: print coverage counts per model in this panel
        try:
            cov_summary = { _sanitize_model_name(k): int((id_map.loc[model_preds[k]["rows"]].astype(str).values == sid).sum()) for k in mdl_keys }
            print(f"[Traj Grouped] Student {sid} coverage in '{panel_name}': {cov_summary}")
        except Exception:
            pass

        for i, k in enumerate(mdl_keys):
            data = model_preds[k]
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if not mask_sid.any():
                continue
            rows_sid = data["rows"][mask_sid]
            ord_sid = order_map.loc[rows_sid].to_numpy()
            args = np.argsort(ord_sid)
            ord_sid = ord_sid[args]
            yp = data["y_prob"][mask_sid][args]
            ax.plot(
                ord_sid + _x_jitter_for_model(k),
                yp,
                color=color_by_model.get(k, palette[i % len(palette)]),
                lw=2.2,
                alpha=0.98,
                linestyle=linestyle_for(k),
                marker=markers[i % len(markers)],
                markerfacecolor="white",
                markeredgecolor=color_by_model.get(k, palette[i % len(palette)]),
                markeredgewidth=1.0,
                markersize=(6 if _sanitize_model_name(k).lower()=="adaptkt" else 5),
                markevery=6,
                zorder=5,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
                # Legend label: just model name (sanitized), no extras
                label=_display_model_name(k),
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel("p(correct)")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.02)
        ax.set_title(f"{panel_name}")

    # Figure 1: single student with two panels (prefer a student with AdaptKT coverage)
    adv_key = key_by_sanitized.get("adaptkt")
    if adv_key in model_preds:
        has_adapt = []
        for s in top_students:
            try:
                cnt = int((id_map.loc[model_preds[adv_key]["rows"]].astype(str).values == s).sum())
            except Exception:
                cnt = 0
            if cnt > 0:
                has_adapt.append((s, cnt))
        sid0 = has_adapt[0][0] if has_adapt else top_students[0]
    else:
        sid0 = top_students[0]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6.2), sharex=True)
    # Panel titles show display names for models actually present in that panel
    panel1_models = [m for m in group_defs["Classical baselines"] if m in key_by_sanitized]
    panel2_models = [m for m in group_defs["Advanced models"] if m in key_by_sanitized]
    panel1_disp = [_display_model_name(m) for m in panel1_models]
    panel2_disp = [_display_model_name(m) for m in panel2_models]
    title1 = f"Student {sid0} — " + ", ".join(panel1_disp) if panel1_disp else f"Student {sid0}"
    title2 = f"Student {sid0} — " + ", ".join(panel2_disp) if panel2_disp else f"Student {sid0}"
    _plot_panel(axes[0], sid0, title1, group_defs["Classical baselines"])
    _plot_panel(axes[1], sid0, title2, group_defs["Advanced models"])
    axes[-1].set_xlabel("Interaction order", labelpad=0)
    # Legend outside bottom (combine from last panel)
    # Collect labels from both panels to avoid missing ones
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    # Deduplicate preserving order
    seen = set()
    handles_dedup = []
    labels_dedup = []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l)
            handles_dedup.append(h)
            labels_dedup.append(l)
    if handles_dedup:
        fig.legend(handles_dedup, labels_dedup, loc="lower center", bbox_to_anchor=(0.5, -0.012),
                   ncol=min(9, len(labels_dedup)), fontsize=9, frameon=True)
        fig.tight_layout(rect=[0, 0.045, 1, 0.985])
    else:
        fig.tight_layout(rect=[0, 0.045, 1, 0.985])
    # No suptitle per request (remove 'Grouped by Model Family')
    fig.savefig(outdir / "trajectories_grouped_single_student.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "trajectories_grouped_single_student.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figures 2 & 3: multi-student grids, one per panel
    def _plot_grid(panel_key: str, filename_stub: str):
        n = len(top_students)
        n = min(n, max_students)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2.8 * nrows), sharex=False)
        axes = np.array(axes).reshape(nrows, ncols)
        for i in range(n):
            r = i // ncols
            c = i % ncols
            ax = axes[r, c]
            sid = top_students[i]
            _plot_panel(ax, sid, f"Student {sid}", group_defs[panel_key])
        # remove unused axes
        for j in range(n, nrows * ncols):
            r = j // ncols
            c = j % ncols
            fig.delaxes(axes[r, c])
        # Safely set x-label on a visible bottom axis
        try:
            axes_flat = [ax for ax in axes.ravel() if ax in fig.axes]
            if axes_flat:
                axes_flat[min(max(0, n - 1), len(axes_flat) - 1)].set_xlabel("Interaction order", labelpad=0)
        except Exception:
            pass
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.012),
                       ncol=min(6, len(labels)), fontsize=9, frameon=True)
            fig.tight_layout(rect=[0, 0.045, 1, 0.985])
        else:
            fig.tight_layout(rect=[0, 0.045, 1, 0.985])
        fig.suptitle(f"Trajectories — {panel_key}", y=0.99)
        fig.savefig(outdir / f"trajectories_grouped_multi_students__{filename_stub}.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / f"trajectories_grouped_multi_students__{filename_stub}.pdf", bbox_inches="tight")
        plt.close(fig)

    _plot_grid("Classical baselines", "classical")
    _plot_grid("Advanced models", "advanced")


def plot_student_trajectories_best_vs_worst(
    metrics: pd.DataFrame,
    outdir: Path,
    by_metric: str = "accuracy",
    min_len: int = 20,
    max_len: int = 60,
):
    """
    Single-student, two-panel trajectories:
      Top panel: best 3 models by `by_metric` present in predictions with row indices
      Bottom panel: worst 3 models by `by_metric`
    Saves: trajectories_best_vs_worst_single_student.[png|pdf]
    """
    df = add_time_index(load_itemwise_df())
    if config.COL_ID not in df.columns or config.COL_ORDER not in df.columns:
        return
    model_preds = _collect_model_predictions_with_dfrows(metrics)
    if not model_preds:
        return

    # Ensure metric exists; fallback to 'ap' if missing
    metric_col = by_metric if by_metric in metrics.columns else ("ap" if "ap" in metrics.columns else None)
    if metric_col is None:
        return

    # Only consider models we have rows for
    metr_filt = metrics[metrics["model"].isin(model_preds.keys())].copy()
    if metr_filt.empty:
        return

    # Rank and pick top3 / bottom3
    metr_filt = metr_filt.sort_values(metric_col, ascending=False)
    best3 = [row["model"] for _, row in metr_filt.head(3).iterrows()]
    worst3 = [row["model"] for _, row in metr_filt.tail(3).iterrows()]

    # Maps for alignment
    id_map = df[config.COL_ID].astype(str)
    order_map = df[config.COL_ORDER]

    # Candidate students by length
    sizes = df.groupby(df[config.COL_ID].astype(str))[config.COL_ID].size()
    allowed = set(sizes[(sizes >= min_len) & (sizes <= max_len)].index.astype(str).tolist())

    # Prefer intersection coverage across all top-3 models so each line appears
    sids_by_model: Dict[str, set] = {}
    for m in best3:
        data = model_preds.get(m)
        if data is None:
            continue
        sids = set(id_map.loc[data["rows"]].astype(str))
        sids_by_model[m] = {s for s in sids if s in allowed}
    inter: set = set.intersection(*sids_by_model.values()) if sids_by_model and len(sids_by_model) == 3 else set()

    sid0: str
    if inter:
        # Choose the student in the intersection with the largest total coverage across the three
        def total_cov(s: str) -> int:
            tot = 0
            for m in best3:
                data = model_preds.get(m)
                if data is None:
                    continue
                tot += int((id_map.loc[data["rows"]].astype(str).values == s).sum())
            return tot
        sid0 = max(inter, key=total_cov)
    else:
        # Fallback: original coverage-based selection (at least one best model covers the student)
        coverage: Dict[str, int] = {}
        for m in best3:
            data = model_preds.get(m)
            if data is None:
                continue
            sids = id_map.loc[data["rows"]].astype(str)
            for s in sids:
                if s in allowed:
                    coverage[s] = coverage.get(s, 0) + 1
        if not coverage:
            return
        sid0 = max(coverage.items(), key=lambda x: x[1])[0]

    # Colors: ensure unique colors within this figure to avoid ambiguous overlaps
    fixed_colors = {
        "dkt": "#CC79A7",  # magenta
        "fkt": "#009E73",  # green
        "clkt": "#0072B2", # blue
        "adaptkt": "#D55E00", # orange
        "gkt": "#E69F00",  # yellow/orange
        "bkt": "#56B4E9",  # light blue
        "rasch1pl": "#0072B2", # blue (IRT)
        "logisticregression": "#009E73", # green (LR)
        "tirt": "#D55E00", # orange (temporal IRT)
    }
    all_models = best3 + [m for m in worst3 if m not in best3]
    # Build a collision-free color map for this figure
    base_palette = sns.color_palette("husl", n_colors=max(8, len(all_models)))
    used = set()
    color_by_model = {}
    for i, name in enumerate(all_models):
        key = _sanitize_model_name(name).lower()
        c = fixed_colors.get(key)
        if c is None or c in used:
            # Assign an unused palette color
            c = base_palette[i % len(base_palette)]
            # Ensure uniqueness by stepping if needed
            j = 0
            while c in used and j < len(base_palette):
                i = (i + 1) % len(base_palette)
                c = base_palette[i]
                j += 1
        color_by_model[name] = c
        used.add(c)
    def color_for(name: str, i: int) -> tuple:
        return color_by_model.get(name, base_palette[i % len(base_palette)])

    def linestyle_for(name: str) -> str:
        fam = _family_of_model(name)
        if fam == "deep":
            return "solid"
        if fam == "classical":
            return "dashed"
        if fam == "hybrid":
            return "dotted"
        if fam == "graph":
            return "dashdot"
        return "solid"

    markers = ["D", "o", "s", "^", "v", "P"]

    def _plot_panel(ax: plt.Axes, sid: str, mdl_list: List[str], title: str):
        # Union of rows across models for this student for GT dots
        rows_union_list = []
        for k in mdl_list:
            data = model_preds.get(k)
            if data is None:
                continue
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if mask_sid.any():
                rows_union_list.append(data["rows"][mask_sid])
        if rows_union_list:
            rows_union = np.unique(np.concatenate(rows_union_list))
            ords = order_map.loc[rows_union].to_numpy()
            args = np.argsort(ords)
            rows_union = rows_union[args]
            ords = ords[args]
            gt_vals = pd.to_numeric(df.loc[rows_union, config.COL_RESP], errors="coerce").to_numpy()
            # light shade incorrect
            if len(ords) > 0:
                for o, g in zip(ords, gt_vals):
                    if g == 0:
                        ax.axvspan(float(o) - 0.5, float(o) + 0.5, color="#000000", alpha=0.08, zorder=0)
            m = np.isin(gt_vals, [0, 1]) & np.isfinite(ords)
            ax.scatter(ords[m], gt_vals[m].astype(int), s=48, c="black", edgecolors="white", linewidths=0.7, zorder=4, label="Ground truth")

        # Lines per model
        for i, k in enumerate(mdl_list):
            data = model_preds.get(k)
            if data is None:
                continue
            mask_sid = id_map.loc[data["rows"]].astype(str).values == sid
            if not mask_sid.any():
                continue
            rows_sid = data["rows"][mask_sid]
            ord_sid = order_map.loc[rows_sid].to_numpy()
            args = np.argsort(ord_sid)
            ord_sid = ord_sid[args]
            yp = data["y_prob"][mask_sid][args]
            c = color_for(k, i)
            # Extra jitter: symmetric x/y offsets by rank within this panel
            # Helps separate very close trajectories visually
            _xo_raw = (-0.40, 0.0, 0.40)[i % 3] if len(mdl_list) <= 3 else (i - (len(mdl_list)-1)/2) * 0.12
            # Keep combined x-jitter (model-specific + panel) inside a single time bin
            _xo = float(np.clip(_x_jitter_for_model(k) + _xo_raw, -0.45, 0.45) - _x_jitter_for_model(k))
            _yj = (0.080, 0.0, -0.080)[i % 3] if len(mdl_list) <= 3 else (i - (len(mdl_list)-1)/2) * 0.020
            # Targeted vertical nudge for classical trio across any panel
            # Ensures LR/CLKT/TIRT are visually separated even when overlapping
            ksan = _sanitize_model_name(k).lower()
            if ksan == "logisticregression":
                _yj += 0.080
            elif ksan == "tirt":
                _yj -= 0.080
            elif ksan == "clkt":
                _yj += 0.060
            elif ksan == "rasch1pl":
                # Do not apply vertical jitter to Rasch1PL to preserve edge values hitting 0/1
                _yj = 0.0
            x_vals = ord_sid + _x_jitter_for_model(k) + _xo
            # Apply vertical jitter only for mid-range probabilities to avoid altering extremes
            mid_mask = (yp >= 0.02) & (yp <= 0.98)
            y_vals = np.clip(yp + (_yj * mid_mask.astype(float)), 0.0, 1.0)
            ax.plot(
                x_vals, y_vals,
                color=c, lw=2.2, alpha=0.98,
                linestyle=linestyle_for(k),
                marker=markers[i % len(markers)],
                markerfacecolor="white",
                markeredgecolor=c, markeredgewidth=1.0,
                markersize=6,
                markevery=4,
                zorder=5,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
                label=_display_model_name(k),
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel("p(correct)")
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.02)
        ax.set_title(title)

    # Build figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6.2), sharex=True)
    _plot_panel(axes[0], sid0, best3, f"Student {sid0} — Best 3 by {metric_col}")
    _plot_panel(axes[1], sid0, worst3, f"Student {sid0} — Worst 3 by {metric_col}")
    axes[-1].set_xlabel("Interaction order", labelpad=0)

    # Shared legend inside bottom axis (combine both panels)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    seen = set(); handles_dedup = []; labels_dedup = []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l); handles_dedup.append(h); labels_dedup.append(l)
    if handles_dedup:
        fig.legend(handles_dedup, labels_dedup, loc="lower center", bbox_to_anchor=(0.5, -0.012),
                   ncol=min(8, len(labels_dedup)), fontsize=9, frameon=True)
        fig.tight_layout(rect=[0, 0.045, 1, 0.985])
    else:
        fig.tight_layout(rect=[0, 0.045, 1, 0.985])

    fig.savefig(outdir / "trajectories_best_vs_worst_single_student.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "trajectories_best_vs_worst_single_student.pdf", bbox_inches="tight")
    plt.close(fig)
   
def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_val = -np.inf
    best_stats = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            val = f1_score(y_true, y_pred, zero_division=0)
        else:
            # default to F1
            val = f1_score(y_true, y_pred, zero_division=0)
        if val > best_val:
            best_val = val
            best_t = float(t)
            best_stats = {
                "f1": float(val),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
    return best_t, best_stats


def plot_roc_all(metrics: pd.DataFrame, outdir: Path):
    """Top-3 ROC wrapper (legacy name).
    We now only present Top-3 ROC; this function delegates and returns early.
    """
    return plot_roc_top3(metrics, outdir)

def plot_pr_all(metrics: pd.DataFrame, outdir: Path):
    """Top-3 PR wrapper (legacy name).
    We now only present Top-3 PR; this function delegates and returns early.
    """
    return plot_pr_top3(metrics, outdir)
    
def plot_pr_top3(metrics: pd.DataFrame, outdir: Path):
    """Single square PR chart with ONLY the top-3 models (by AP).
    Lines are drawn thicker with white outlines and sparse markers so close curves
    remain distinguishable.
    Saves: pr_top3.png / pr_top3.pdf
    """
    sns.set(style="whitegrid", context="paper")

    # Collect PR data
    curves = []  # (name, ap, recall, precision)
    base_vals = []
    for _, row in metrics.iterrows():
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            if len(y_true) == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            curves.append((name, float(ap), rec, prec))
            base_vals.append(float(np.mean(y_true)))
        except Exception as e:
            print(f"[PR-top3] Skipping {name}: {e}")

    if not curves:
        return
    curves.sort(key=lambda x: -x[1])
    top3 = curves[:3]

    # Colors: fixed mapping + fallback
    fixed_colors = {
        "dkt": "#CC79A7",                 # magenta
        "fkt": "#009E73",                 # green
        "clkt": "#0072B2",                # blue
        "adaptkt": "#D55E00",             # orange
        "gkt": "#E69F00",                 # yellow-orange
        "bkt": "#56B4E9",                 # sky blue
        "rasch1pl": "#999999",            # gray (distinct)
        "logisticregression": "#2CA02C",  # green (distinct from CLKT blue)
        "tirt": "#9467BD",                # purple (distinct from AdaptKT)
    }
    palette = sns.color_palette("colorblind", n_colors=3)
    def color_for(name: str, i: int):
        key = _sanitize_model_name(name).lower()
        return fixed_colors.get(key, palette[i % len(palette)])

    # Styles: distinct linestyles + sparse markers to separate close curves
    markers = ["o", "s", "D"]
    linestyles = ["solid", "dashed", "dashdot"]

    base = float(np.median(base_vals)) if base_vals else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.grid(True, color="#d0d0d0", alpha=0.4)
    ax.hlines(base, 0, 1, colors="gray", linestyles="--", lw=1.2, alpha=0.9, label=f"Class prior (≈ {base:.3f})")

    # Jitter: top curve up, bottom down (keep within [0,1])
    jitter = {0: +0.004, 1: 0.0, 2: -0.004}
    for i, (name, ap, rec, prec) in enumerate(top3):
        c = color_for(name, i)
        prec_j = np.clip(prec + jitter.get(i, 0.0), 0.0, 1.0)
        ax.plot(
            rec,
            prec_j,
            color=c,
            lw=3.0,
            alpha=1.0,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5.2,
            markevery=max(12, len(rec) // 20),
            markerfacecolor="white",
            markeredgecolor=c,
            markeredgewidth=1.4,
            path_effects=[patheffects.withStroke(linewidth=4.5, foreground="white")],
            label=f"{_display_model_name(name)} (AP={ap:.4f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # Zoomed inset on MIDDLE recall region with stronger magnification
    try:
        ins = ax.inset_axes([0.34, 0.12, 0.56, 0.56])
        ins.grid(True, color="#d0d0d0", alpha=0.3)
        x0, x1 = 0.52, 0.68
        ymins, ymaxs = [], []
        for i, (name, ap, rec, prec) in enumerate(top3):
            c = color_for(name, i)
            prec_j = np.clip(prec + jitter.get(i, 0.0), 0.0, 1.0)
            ins.plot(
                rec,
                prec_j,
                color=c,
                lw=2.0,
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=4.2,
                markevery=max(8, len(rec) // 22),
                markerfacecolor="white",
                markeredgecolor=c,
                markeredgewidth=1.0,
                path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")],
            )
            m = (rec >= x0) & (rec <= x1)
            if np.any(m):
                ymins.append(float(np.nanmin(prec[m])))
                ymaxs.append(float(np.nanmax(prec[m])))
        ins.set_xlim(x0, x1)
        if ymins:
            y0 = max(0.6, min(ymins) - 0.015)
            y1 = min(1.0, max(ymaxs) + 0.015)
            ins.set_ylim(y0, y1)
        ins.tick_params(labelsize=8)
        try:
            ax.indicate_inset_zoom(ins, edgecolor="gray")
        except Exception:
            pass
    except Exception:
        pass

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.012),
            ncol=3,
            fontsize=9,
            frameon=True,
            borderaxespad=0.05,
            labelspacing=0.25,
            columnspacing=0.7,
            handletextpad=0.35,
        )
        fig.tight_layout(rect=[0, 0.045, 1, 0.99])
    else:
        fig.tight_layout(rect=[0, 0.045, 1, 0.99])

    fig.savefig(outdir / "pr_top3.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "pr_top3.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_roc_top3(metrics: pd.DataFrame, outdir: Path):
    """Single square ROC chart with ONLY the top-3 models (by ROC AUC).
    Thick lines, white stroke, sparse markers removed (clean lines), random baseline.
    Saves: roc_top3.png / roc_top3.pdf
    """
    sns.set(style="whitegrid", context="paper")

    curves = []  # (name, auc, fpr, tpr)
    for _, row in metrics.iterrows():
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            if len(y_true) == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            curves.append((name, float(auc_val), fpr, tpr))
        except Exception as e:
            print(f"[ROC-top3] Skipping {name}: {e}")

    if not curves:
        return
    curves.sort(key=lambda x: -x[1])
    top3 = curves[:3]

    fixed_colors = {
        "dkt": "#CC79A7",
        "fkt": "#009E73",
        "clkt": "#0072B2",
        "adaptkt": "#D55E00",
        "gkt": "#E69F00",
        "bkt": "#56B4E9",
        "rasch1pl": "#999999",            # gray to avoid clash with CLKT
        "logisticregression": "#2CA02C",  # green distinct from CLKT
        "tirt": "#9467BD",                # purple distinct from AdaptKT
    }
    palette = sns.color_palette("colorblind", n_colors=3)
    def color_for(name: str, i: int):
        key = _sanitize_model_name(name).lower()
        return fixed_colors.get(key, palette[i % len(palette)])

    # Styles for separation when curves are very close
    markers = ["o", "s", "D"]
    linestyles = ["solid", "dashed", "dashdot"]

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.grid(True, color="#d0d0d0", alpha=0.35)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2, alpha=0.85, label="Random (AUC=0.5)")

    # Small vertical jitter for visual separation: top up, bottom down
    jitter = {0: +0.004, 1: 0.0, 2: -0.004}
    for i, (name, auc_val, fpr, tpr) in enumerate(top3):
        c = color_for(name, i)
        tpr_j = np.clip(tpr + jitter.get(i, 0.0), 0.0, 1.0)
        ax.plot(
            fpr,
            tpr_j,
            color=c,
            lw=3.0,
            alpha=1.0,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5.0,
            markevery=max(12, len(fpr) // 20),
            markerfacecolor="white",
            markeredgecolor=c,
            markeredgewidth=1.2,
            path_effects=[patheffects.withStroke(linewidth=4.5, foreground="white")],
            label=f"{_display_model_name(name)} (AUC={auc_val:.4f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # Zoomed inset on MIDDLE FPR region with stronger magnification
    try:
        ins = ax.inset_axes([0.34, 0.12, 0.56, 0.56])
        ins.grid(True, color="#d0d0d0", alpha=0.3)
        x0, x1 = 0.40, 0.60
        ymins, ymaxs = [], []
        for i, (name, auc_val, fpr, tpr) in enumerate(top3):
            c = color_for(name, i)
            tpr_j = np.clip(tpr + jitter.get(i, 0.0), 0.0, 1.0)
            ins.plot(
                fpr,
                tpr_j,
                color=c,
                lw=2.0,
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=3.6,
                markevery=max(10, len(fpr) // 22),
                markerfacecolor="white",
                markeredgecolor=c,
                markeredgewidth=0.9,
                path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")],
            )
            m = (fpr >= x0) & (fpr <= x1)
            if np.any(m):
                ymins.append(float(np.nanmin(tpr_j[m])))
                ymaxs.append(float(np.nanmax(tpr_j[m])))
        ins.set_xlim(x0, x1)
        if ymins:
            y0 = max(0.6, min(ymins) - 0.015)
            y1 = min(1.0, max(ymaxs) + 0.015)
            ins.set_ylim(y0, y1)
        ins.tick_params(labelsize=8)
        try:
            ax.indicate_inset_zoom(ins, edgecolor="gray")
        except Exception:
            pass
    except Exception:
        pass

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=3, fontsize=9, frameon=True)
        fig.tight_layout(rect=[0, 0.045, 1, 0.99])
    else:
        fig.tight_layout(rect=[0, 0.045, 1, 0.99])

    fig.savefig(outdir / "roc_top3.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "roc_top3.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_calibration_grid(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    m = len(metrics)
    ncols = 3
    nrows = int(np.ceil(m / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    for i, row in metrics.reset_index(drop=True).iterrows():
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
            ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=_display_model_name(name))
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.set_title(_display_model_name(name), fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{name}", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if r == nrows - 1:
            ax.set_xlabel("Predicted probability")
        if c == 0:
            ax.set_ylabel("Observed frequency")
    # Remove empty axes
    for j in range(i + 1, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])
    fig.suptitle("Calibration (Reliability) Curves", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "calibration_grid.png", dpi=300)
    fig.savefig(outdir / "calibration_grid.pdf")
    plt.close(fig)


def plot_confusion_matrices(metrics: pd.DataFrame, outdir: Path):
    """
    Plot a grid of confusion matrices with CONSISTENT colors for TN/FP/FN/TP across all models.
    Color mapping (fixed):
      TN -> blue, FP -> orange, FN -> red, TP -> greenish/teal.
    """
    sns.set(style="white", context="paper")
    m = len(metrics)
    ncols = 3
    nrows = int(np.ceil(m / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 2.9 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    # Fixed color mapping per cell index (0..3) across all matrices
    # cm layout: [[TN, FP], [FN, TP]]
    color_list = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]  # TN, FP, FN, TP
    cmap = ListedColormap(color_list)
    color_index = np.array([[0, 1], [2, 3]])

    # For a shared legend
    patches = [
        mpatches.Patch(color=color_list[0], label="TN"),
        mpatches.Patch(color=color_list[1], label="FP"),
        mpatches.Patch(color=color_list[2], label="FN"),
        mpatches.Patch(color=color_list[3], label="TP"),
    ]

    for i, row in metrics.reset_index(drop=True).iterrows():
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
            t, stats = best_threshold(y_true, y_prob, metric="f1")
            y_pred = (y_prob >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            # Draw colored cells with fixed mapping, independent of count magnitudes
            ax.imshow(color_index, cmap=cmap, vmin=0, vmax=3)
            # Annotate counts
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            counts = np.array([[tn, fp], [fn, tp]])
            for (rr, cc), val in np.ndenumerate(counts):
                ax.text(cc, rr, f"{val}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            # Axes cosmetics
            ax.set_xticks([0, 1], labels=["0", "1"])  # predicted
            ax.set_yticks([0, 1], labels=["0", "1"])  # true
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            # Title: model name only (remove threshold/metrics)
            ax.set_title(f"{_display_model_name(name)}", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n{name}", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])

    # Remove empty axes
    for j in range(i + 1, nrows * ncols):
        r = j // ncols
        c = j % ncols
        fig.delaxes(axes[r, c])

    # Shared legend and title
    fig.legend(handles=patches, loc="lower center", ncol=4, title="Cells", title_fontsize=9, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(outdir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "confusion_matrices.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    # Removed Brier at user's request; keep five metrics
    metric_cols = [
        ("roc_auc", "ROC-AUC (↑)"),
        ("accuracy", "Accuracy (↑)"),
        ("avg_precision", "Average Precision (↑)"),
        ("f1", "F1 Score (↑)"),
        ("log_loss", "Log Loss (↓)"),
    ]
    # Dynamic grid (3 columns)
    n = len(metric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3.0 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, ((col, title)) in enumerate(metric_cols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if col not in metrics.columns:
            ax.axis("off")
            continue
        df = metrics[["model", col]].copy()
        df = df.sort_values(col, ascending=(col in {"log_loss"}))
        # Display names on y-axis
        df["model_disp"] = df["model"].apply(_display_model_name)
        sns.barplot(data=df, x=col, y="model_disp", ax=ax, palette="viridis")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # annotate values
        for i, v in enumerate(df[col].values):
            ax.text(v, i, f" {v:.6f}", va="center", fontsize=8)
    # Hide any remaining empty axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r, c])
    fig.suptitle("Model Comparison on Test Metrics", y=0.995)
    fig.tight_layout()
    fig.savefig(outdir / "metric_bars.png", dpi=300)
    fig.savefig(outdir / "metric_bars.pdf")
    plt.close(fig)


def plot_metric_radar_ranks(metrics: pd.DataFrame, outdir: Path):
    """
    Build ranks (positions) per model over five metrics and plot a radar chart.
    Metrics used (higher-better except log_loss):
      - accuracy, roc_auc, avg_precision, f1, log_loss
    Saves:
      - metric_radar_ranks.[png|pdf]
      - metric_rank_summary.csv (with per-metric rank and total_rank)
    """
    sns.set(style="whitegrid", context="paper")
    metric_defs = [
        ("accuracy", True, "Accuracy"),
        ("roc_auc", True, "ROC-AUC"),
        ("avg_precision", True, "Average Precision"),
        ("f1", True, "F1 Score"),
        ("log_loss", False, "Log Loss"),
    ]
    # Filter to available metrics
    metric_defs = [(c, up, lbl) for (c, up, lbl) in metric_defs if c in metrics.columns]
    if not metric_defs:
        return
    models = metrics["model"].tolist()
    M = len(models)
    # Compute ranks per metric
    ranks = pd.DataFrame(index=models)
    for col, higher_better, label in metric_defs:
        s = metrics.set_index("model")[col].copy()
        # NaNs -> worst (assign max rank)
        s_filled = s.copy()
        if higher_better:
            order = s_filled.rank(ascending=False, method="min")
        else:
            order = s_filled.rank(ascending=True, method="min")
        # Any NaNs rank to worst
        order = order.fillna(M)
        ranks[col] = order.astype(int)
    ranks["total_rank"] = ranks.sum(axis=1)
    # Derive scores so best gets M and worst gets 1
    scores = ranks.drop(columns=["total_rank"]).apply(lambda col: (M - col + 1))
    scores = scores.astype(int)
    scores["total_score"] = scores.sum(axis=1)
    # Save summary CSV with both ranks and scores, plus display names
    out_csv = outdir / "metric_rank_summary.csv"
    disp = pd.DataFrame({"model": models, "model_display": [ _display_model_name(m) for m in models ]}).set_index("model")
    # Interleave columns: metric_rank, then metric_score
    cols_order = []
    for col, _, _ in metric_defs:
        if col in ranks.columns:
            cols_order.append(col)  # rank
        if col in scores.columns:
            cols_order.append(f"{col}_score")
            scores.rename(columns={col: f"{col}_score"}, inplace=True)
    merged = disp.join([ranks, scores])
    # Reorder columns to desired order + totals
    ordered_cols = ["model_display"] + cols_order + ["total_rank", "total_score"]
    merged = merged.reindex(columns=ordered_cols)
    merged = merged.sort_values("total_score", ascending=False)
    merged.to_csv(out_csv, index=True)

    # Radar plot uses rank (1=best). Convert to score so outward is better
    # Higher score = (M - rank + 1)
    theta_labels = [lbl for _, _, lbl in metric_defs]
    K = len(metric_defs)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    # Color mapping consistent with other plots
    fixed_colors = {
        "dkt": "#CC79A7", "fkt": "#009E73", "clkt": "#0072B2", "adaptkt": "#D55E00",
        "gkt": "#E69F00", "bkt": "#56B4E9", "rasch1pl": "#999999", "logisticregression": "#0173B2", "tirt": "#9467BD",
    }
    palette = sns.color_palette("colorblind", n_colors=max(9, M))
    def color_for(name: str, i: int):
        key = _sanitize_model_name(name).lower()
        return fixed_colors.get(key, palette[i % len(palette)])

    def linestyle_for(name: str) -> str:
        fam = _family_of_model(name)
        if fam == "deep":
            return "solid"
        if fam == "classical":
            return "dashed"
        if fam == "hybrid":
            return "dotted"
        if fam == "graph":
            return "dashdot"
        return "solid"

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Grid and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(theta_labels)
    ax.set_yticks(range(0, M + 1))
    ax.set_yticklabels([str(r) for r in range(0, M + 1)])
    ax.set_ylim(0, M)
    ax.yaxis.grid(True, color="#d0d0d0", alpha=0.35)
    ax.xaxis.grid(True, color="#d0d0d0", alpha=0.35)

    # Plot each model
    for i, m in enumerate(models):
        r = []
        for col, _, _ in metric_defs:
            rk = int(ranks.loc[m, col])
            score = M - rk + 1  # outward is better
            r.append(score)
        r += r[:1]
        c = color_for(m, i)
        ax.plot(angles, r, color=c, lw=2.0, linestyle=linestyle_for(m), label=_display_model_name(m),
                path_effects=[patheffects.withStroke(linewidth=3.2, foreground="white")])
        ax.fill(angles, r, color=c, alpha=0.06)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=4,
            fontsize=9,
            frameon=True,
            borderaxespad=0.05,
            labelspacing=0.25,
            columnspacing=0.8,
            handletextpad=0.35,
        )
        fig.tight_layout(rect=[0, 0.075, 1, 0.995])
    else:
        fig.tight_layout(rect=[0, 0.075, 1, 0.995])
    fig.savefig(outdir / "metric_radar_ranks.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "metric_radar_ranks.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_overall_rank_bars(metrics: pd.DataFrame, outdir: Path):
    """
    Plot a horizontal bar chart of overall ranks (sum of positions across
    Accuracy, ROC AUC, Average Precision, F1, Log Loss). Lower is better.
    Saves: metric_rank_overall.[png|pdf]
    """
    sns.set(style="whitegrid", context="paper")
    metric_defs = [
        ("accuracy", True),
        ("roc_auc", True),
        ("avg_precision", True),
        ("f1", True),
        ("log_loss", False),
    ]
    metric_cols = [c for c, _ in metric_defs if c in metrics.columns]
    if not metric_cols:
        return
    models = metrics["model"].tolist()
    M = len(models)
    # Compute ranks and scores per metric as in radar
    ranks = pd.DataFrame(index=models)
    for col, higher_better in [(c, up) for (c, up) in metric_defs if c in metric_cols]:
        s = metrics.set_index("model")[col].copy()
        order = s.rank(ascending=not higher_better, method="min")
        order = order.fillna(M)
        ranks[col] = order.astype(int)
    # Scores: best=M, worst=1
    scores = ranks.apply(lambda col: (M - col + 1))
    scores = scores.astype(int)
    scores["total_score"] = scores.sum(axis=1)
    df = scores.reset_index().rename(columns={"index": "model"})
    df["model_display"] = df["model"].apply(_display_model_name)
    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)

    # Distinct colors for each model (no duplicates). Close to PR/ROC palette.
    fixed_colors = {
        "dkt": "#CC79A7",              # magenta
        "fkt": "#009E73",              # green
        "clkt": "#0072B2",             # blue
        "adaptkt": "#D55E00",          # dark orange
        "gkt": "#E69F00",              # yellow-orange
        "bkt": "#56B4E9",              # sky blue
        "rasch1pl": "#999999",         # gray
        "logisticregression": "#8C564B",# brown (avoid green clash with FKT)
        "tirt": "#9467BD",             # purple
    }
    palette = sns.color_palette("colorblind", n_colors=max(9, len(df)))
    def color_for(name: str, i: int):
        key = _sanitize_model_name(name).lower()
        return fixed_colors.get(key, palette[i % len(palette)])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(df))
    colors = [color_for(n, i) for i, n in enumerate(df["model"].tolist())]
    ax.barh(y, df["total_score"].values, color=colors, edgecolor="white")
    ax.set_yticks(y, labels=df["model_display"].tolist())
    ax.invert_yaxis()  # best (lowest total rank) at top
    ax.set_xlabel("Sum of Ranks")
    ax.set_title("Sum of Ranks across Accuracy, ROC-AUC, AP, F1 Score, Log Loss")
    # Annotate total rank values
    for yi, val in zip(y, df["total_score"].values):
        ax.text(val + 0.2, yi, f"{int(val)}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "metric_rank_overall.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "metric_rank_overall.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_per_model(metrics: pd.DataFrame, outdir: Path):
    sns.set(style="whitegrid", context="paper")
    for _, row in metrics.iterrows():
        name = row["model"]
        try:
            y_true, y_prob = _load_predictions(config.OUTPUT_DIR, name)
        except Exception as e:
            print(f"[Per-model] Skipping {name}: {e}")
            continue
        # ROC + PR + hist in one figure
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.2))
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[0].plot(fpr, tpr, lw=2)
        axes[0].set_title(f"ROC (AUC={auc_val:.4f})")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        base = y_true.mean() if len(y_true) else 0.0
        axes[1].plot(recall, precision, lw=2)
        axes[1].hlines(base, 0, 1, colors="gray", linestyles="--")
        axes[1].set_title(f"PR (AP={ap:.4f})")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        # Histogram
        axes[2].hist(y_prob, bins=25, color="#4c72b0")
        axes[2].set_title("Predicted Probabilities")
        axes[2].set_xlabel("p(y=1)")
        axes[2].set_ylabel("Count")
        fig.suptitle(name)
        fig.tight_layout()
        tag = name_to_tag(name)
        fig.savefig(outdir / f"{tag}__overview.png", dpi=300)
        fig.savefig(outdir / f"{tag}__overview.pdf")
        plt.close(fig)


def main():
    outdir = config.OUTPUT_DIR
    plot_dirs = _ensure_plot_dirs(outdir)
    metrics, details = _load_summary(outdir)

    # Order models by ROC AUC descending where available
    if "roc_auc" in metrics.columns:
        metrics = metrics.sort_values("roc_auc", ascending=False, na_position="last").reset_index(drop=True)

    # Summary plots (Top-3 only)
    plot_roc_top3(metrics, plot_dirs["summary"])  # ROC top-3 overlay
    plot_pr_top3(metrics, plot_dirs["summary"])   # PR top-3 overlay
    plot_metric_bars(metrics, plot_dirs["summary"])  # bar charts for metrics (no Brier)
    plot_metric_radar_ranks(metrics, plot_dirs["summary"])  # radar ranks + CSV summary
    plot_metric_overall_rank_bars(metrics, plot_dirs["summary"])  # overall rank bars
    plot_calibration_grid(metrics, plot_dirs["summary"])  # reliability curves
    plot_confusion_matrices(metrics, plot_dirs["summary"])  # confusion matrices grid

    # Per-model overviews
    plot_per_model(metrics, plot_dirs["per_model"])

    # Qualitative plots using aligned row indices (if available)
    try:
        plot_student_trajectories_all_models(
            metrics,
            plot_dirs["summary"],
            max_students=2,
            models=getattr(config, "TRAJECTORY_MODELS", None),
        )
    except Exception as e:
        print(f"[Trajectories] Skipped: {e}")

    # Grouped trajectories (classical vs advanced) to reduce clutter
    try:
        plot_student_trajectories_grouped(
            metrics,
            plot_dirs["summary"],
            max_students=4,
            min_len=20,
            max_len=60,
        )
    except Exception as e:
        print(f"[Grouped Trajectories] Skipped: {e}")

    # Best vs Worst (by accuracy) single-student trajectories
    try:
        plot_student_trajectories_best_vs_worst(
            metrics,
            plot_dirs["summary"],
            by_metric="accuracy",
            min_len=20,
            max_len=60,
        )
    except Exception as e:
        print(f"[Best-vs-Worst Trajectories] Skipped: {e}")

    print("Saved plots to:", plot_dirs["base"].resolve())


if __name__ == "__main__":
    main()
