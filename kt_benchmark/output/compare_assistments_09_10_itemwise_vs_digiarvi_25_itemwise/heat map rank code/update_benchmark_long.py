from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import pandas as pd

# Use package config for OUTPUT_DIR
try:
    from kt_benchmark import config
except Exception:  # pragma: no cover
    # Fallback if run in unusual contexts
    import importlib
    config = importlib.import_module("kt_benchmark.config")

METRIC_COLS = ["accuracy", "roc_auc", "avg_precision", "f1", "log_loss"]
METRIC_NAMES = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
    "avg_precision": "Average Precision",
    "f1": "F1 Score",
    "log_loss": "Log Loss",
}
MODEL_DISPLAY = {
    "Rasch1PL": "Rasch 1PL",
    # Leave others as-is by default
}

DATASETS = [
    ("ASSISTments 09-10", config.OUTPUT_DIR / "assistments_09_10_itemwise" / "metrics_summary.csv"),
    ("DigiArvi 2025",    config.OUTPUT_DIR / "digiarvi_25_itemwise" / "metrics_summary.csv"),
]

TARGET_CSV = (
    config.OUTPUT_DIR
    / "compare_assistments_09_10_itemwise_vs_digiarvi_25_itemwise"
    / "heat map rank code"
    / "benchmark_summary_long.csv"
)


def _read_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Keep only needed columns if they exist
    keep = [c for c in ["model", *METRIC_COLS] if c in df.columns]
    return df[keep].copy()


def build_long_table() -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for dataset_label, csv_path in DATASETS:
        if not csv_path.exists():
            print(f"[WARN] Missing metrics summary: {csv_path}")
            continue
        df = _read_metrics_csv(csv_path)
        for _, r in df.iterrows():
            model = str(r["model"]) if "model" in r else ""
            model_disp = MODEL_DISPLAY.get(model, model)
            for col in METRIC_COLS:
                if col in df.columns:
                    val = pd.to_numeric(r[col], errors="coerce")
                    if pd.isna(val):
                        continue
                    rows.append(
                        {
                            "Model": model_disp,
                            "Metric": METRIC_NAMES[col],
                            "Dataset": dataset_label,
                            "Value": f"{float(val):.4f}",
                        }
                    )
    long_df = pd.DataFrame(rows, columns=["Model", "Metric", "Dataset", "Value"])
    # Nice sort: Metric, Dataset, Model
    if not long_df.empty:
        long_df = long_df.sort_values(["Metric", "Dataset", "Model"]).reset_index(drop=True)
    return long_df


def main() -> None:
    outdir = TARGET_CSV.parent
    outdir.mkdir(parents=True, exist_ok=True)
    long_df = build_long_table()
    if long_df.empty:
        print("[ERROR] No data assembled; check input CSV paths.")
        return
    long_df.to_csv(TARGET_CSV, index=False)
    print(f"Wrote: {TARGET_CSV}")


if __name__ == "__main__":
    main()
