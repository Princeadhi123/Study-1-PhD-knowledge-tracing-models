from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
import time

from .. import config
from ..utils import prepare_next_step_features_dense_split
from sklearn.model_selection import train_test_split


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "FKT-lite: a small shared MLP predicts correctness (primary) and response time (aux) jointly. "
        "Multi-task learning can improve representation quality and calibration by leveraging correlated auxiliary signals."
    )
    # Optional torch dependency
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        return {"category": "Multi-task", "name": "FKT-lite", "why": why, "error": f"torch not available: {e}"}

    if config.COL_RESP not in df.columns:
        return {"category": "Multi-task", "name": "FKT-lite", "why": why, "error": "response column missing"}

    # Build NEXT-step features and labels; helper includes prev_correct and next-row features
    X_tr_df, X_te_df, y_tr_corr_arr, y_te_corr_arr, rows_tr_next, rows_te_next = prepare_next_step_features_dense_split(
        df,
        train_idx=train_idx,
        test_idx=test_idx,
        use_item=True,
        use_group=True,
        use_sex=True,
        use_time=True,
    )
    if len(y_tr_corr_arr) == 0 or len(y_te_corr_arr) == 0:
        return {"category": "Multi-task", "name": "FKT-lite", "why": why, "error": "no valid next-step train/test rows"}

    # Drop any log_rt column to avoid leakage into RT head if present
    if "log_rt" in X_tr_df.columns:
        X_tr_df = X_tr_df.drop(columns=["log_rt"])  # prevent trivial RT prediction
        X_te_df = X_te_df.drop(columns=["log_rt"])  # keep columns aligned

    X_tr = X_tr_df.values.astype(np.float32)
    X_te = X_te_df.values.astype(np.float32)
    y_tr_corr = y_tr_corr_arr.astype(np.float32)
    y_te_corr = y_te_corr_arr.astype(np.float32)

    # Next-step response time labels (aux task) from NEXT rows
    if config.COL_RT in df.columns:
        y_tr_rt = pd.to_numeric(df.loc[rows_tr_next, config.COL_RT], errors="coerce").values.astype(np.float32)
        y_te_rt = pd.to_numeric(df.loc[rows_te_next, config.COL_RT], errors="coerce").values.astype(np.float32)
    else:
        y_tr_rt = np.full(shape=len(rows_tr_next), fill_value=np.nan, dtype=np.float32)
        y_te_rt = np.full(shape=len(rows_te_next), fill_value=np.nan, dtype=np.float32)
    m_tr_rt = np.isfinite(y_tr_rt)
    m_te_rt = np.isfinite(y_te_rt)

    # Standardize features to zero mean/var
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Torch dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
    yt_corr = torch.tensor(y_tr_corr, dtype=torch.float32, device=device)
    yt_rt = torch.tensor(np.nan_to_num(y_tr_rt, nan=0.0), dtype=torch.float32, device=device)
    mt_rt = torch.tensor(m_tr_rt.astype(np.float32), dtype=torch.float32, device=device)

    class MTL(nn.Module):
        def __init__(self, d_in: int, d_hid: int = 64):
            super().__init__()
            p = float(getattr(config, "MTL_DROPOUT", 0.0))
            self.shared = nn.Sequential(
                nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(p=p),
                nn.Linear(d_hid, d_hid), nn.ReLU(), nn.Dropout(p=p),
            )
            self.head_corr = nn.Sequential(nn.Linear(d_hid, 1))
            self.head_rt = nn.Sequential(nn.Linear(d_hid, 1))
        def forward(self, x):
            h = self.shared(x)
            logit = self.head_corr(h).squeeze(-1)
            rt = self.head_rt(h).squeeze(-1)
            return logit, rt

    model = MTL(d_in=X_tr.shape[1], d_hid=getattr(config, "MTL_HID_DIM", 64)).to(device)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(
        model.parameters(), lr=float(getattr(config, "MTL_LR", 1e-3)), weight_decay=float(getattr(config, "MTL_WEIGHT_DECAY", 0.0))
    )

    # Train (respect time budget) with early stopping via inner validation split
    model.train()
    batch_size = int(getattr(config, "MTL_BATCH", 128))
    n = Xt.shape[0]
    start_time = time.perf_counter()
    budget = getattr(config, "TRAIN_TIME_BUDGET_S", None)
    # Inner split indices
    idx_all = np.arange(n)
    if n >= 5:
        tr_in, va_in = train_test_split(idx_all, test_size=0.2, random_state=config.RANDOM_STATE, stratify=(yt_corr.cpu().numpy().round() if float(yt_corr.numel()) else None))
    else:
        tr_in, va_in = idx_all, np.array([], dtype=int)
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    patience = int(getattr(config, "MTL_PATIENCE", 0))
    wait = 0
    for epoch in range(max(1, config.EPOCHS_MTL)):
        if budget is not None and (time.perf_counter() - start_time) > float(budget):
            break
        perm = torch.randperm(len(tr_in))
        for i in range(0, len(tr_in), batch_size):
            idx_slice = tr_in[perm[i:i+batch_size].cpu().numpy()]
            xb = Xt[idx_slice]
            yb = yt_corr[idx_slice]
            rb = yt_rt[idx_slice]
            mb = mt_rt[idx_slice]
            logit, rt_pred = model(xb)
            loss_corr = bce(logit, yb)
            loss_rt = (mse(rt_pred, rb) * mb).sum() / (mb.sum() + 1e-6)
            loss = loss_corr + 0.2 * loss_rt  # small weight on aux
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Validation step
        if va_in.size > 0:
            model.eval()
            with torch.no_grad():
                xb = Xt[va_in]
                yb = yt_corr[va_in]
                rb = yt_rt[va_in]
                mb = mt_rt[va_in]
                logit, rt_pred = model(xb)
                loss_corr = bce(logit, yb)
                loss_rt = (mse(rt_pred, rb) * mb).sum() / (mb.sum() + 1e-6)
                val_loss = float((loss_corr + 0.2 * loss_rt).item())
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    break
            model.train()

    # Eval
    Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    # Load best if available and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logit, rt_pred = model(Xte_t)
        y_prob = torch.sigmoid(logit).cpu().numpy()
        rt_pred_np = rt_pred.cpu().numpy()

    # Aux metrics where RT available
    if m_te_rt.any():
        rt_err = rt_pred_np[m_te_rt] - y_te_rt[m_te_rt]
        mae = float(np.mean(np.abs(rt_err)))
        rmse = float(np.sqrt(np.mean(rt_err**2)))
    else:
        mae = np.nan
        rmse = np.nan

    return {
        "category": "Multi-task",
        "name": "FKT-lite",
        "why": why,
        "y_true": y_te_corr.astype(int),
        "y_prob": y_prob,
        "test_rows": rows_te_next,
        "aux_mae_rt": mae,
        "aux_rmse_rt": rmse,
        "best_epoch": int(best_epoch) if best_epoch >= 0 else None,
    }
