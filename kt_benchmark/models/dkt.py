from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import time

from .. import config
from ..utils import student_sequences
from sklearn.model_selection import train_test_split


def run(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    why = (
        "DKT (LSTM-based) is the classic deep learning KT model that captures sequential dependencies. "
        "Here we use a minimal version predicting current response from the running hidden state over item embeddings."
    )
    # Torch is optional
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except Exception as e:
        return {"category": "Deep Learning", "name": "DKT (minimal)", "why": why, "error": f"torch not available: {e}"}

    item_col = config.COL_ITEM if config.COL_ITEM in df.columns else config.COL_ITEM_INDEX if config.COL_ITEM_INDEX in df.columns else None
    if item_col is None or config.COL_RESP not in df.columns or config.COL_ID not in df.columns:
        return {"category": "Deep Learning", "name": "DKT (minimal)", "why": why, "error": "Required columns missing (item/item_index, response, ID)"}

    # Build train/test student sets
    train_students = set(df.iloc[train_idx][config.COL_ID].astype(str).unique().tolist())
    test_students = set(df.iloc[test_idx][config.COL_ID].astype(str).unique().tolist())

    # Vocabulary from training items only (avoid test leakage)
    items_train = df.iloc[train_idx][item_col].astype(str).unique().tolist()
    it2ix = {it: i + 1 for i, it in enumerate(items_train)}  # 0 = PAD
    pad_idx = 0
    vocab_size = len(it2ix) + 1

    # Build sequences per student from a filtered, normalized view of df
    df_seq = df[[config.COL_ID, item_col, config.COL_ORDER, config.COL_RESP]].copy()
    df_seq = df_seq.dropna(subset=[config.COL_RESP])
    df_seq[config.COL_ID] = df_seq[config.COL_ID].astype(str)
    df_seq = df_seq.sort_values([config.COL_ID, config.COL_ORDER])

    def build_sequences(student_ids: List[str]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        seqs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for sid in student_ids:
            sub = df_seq.loc[df_seq[config.COL_ID] == sid]
            if sub.empty:
                continue
            items_all = sub[item_col].astype(str).map(lambda x: it2ix.get(x, 0)).values
            y_series = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
            mask = y_series.isin([0, 1]).values
            if mask.sum() == 0:
                continue
            items = items_all[mask]
            y = y_series[mask].astype(int).values
            rows = sub.index.values[mask]
            # Keep sequences with at least 1 labeled step (DKT can handle length-1)
            if len(y) >= 1:  # be permissive to avoid empty splits
                seqs.append((items, y, rows))
        return seqs

    train_seqs = build_sequences(sorted(train_students))
    test_seqs = build_sequences(sorted(test_students))
    # Diagnostics
    diag = {
        "train_student_count": len(train_students),
        "test_student_count": len(test_students),
        "train_seq_count": len(train_seqs),
        "test_seq_count": len(test_seqs),
        "fallback_used": False,
    }
    # Fallback: if either split has zero sequences, re-split among students who actually have sequences
    if not train_seqs or not test_seqs:
        all_students = sorted(df_seq[config.COL_ID].astype(str).unique())

        def student_has_seq(sid: str) -> bool:
            sub = df_seq.loc[df_seq[config.COL_ID] == sid]
            if sub.empty:
                return False
            y_series = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
            return y_series.isin([0, 1]).sum() >= 1

        students_with_seq = [sid for sid in all_students if student_has_seq(sid)]
        if len(students_with_seq) >= 2:
            tr_fb, te_fb = train_test_split(
                students_with_seq,
                test_size=float(getattr(config, "TEST_SIZE", 0.2)),
                random_state=int(getattr(config, "RANDOM_STATE", 42)),
                shuffle=True,
            )
            train_seqs = build_sequences(sorted(tr_fb))
            test_seqs = build_sequences(sorted(te_fb))
            diag.update({
                "fallback_used": True,
                "train_student_count_fallback": len(tr_fb),
                "test_student_count_fallback": len(te_fb),
                "train_seq_count_fallback": len(train_seqs),
                "test_seq_count_fallback": len(test_seqs),
            })

    if not train_seqs or not test_seqs:
        return {
            "category": "Deep Learning",
            "name": "DKT (minimal)",
            "why": why,
            "error": "No valid train/test sequences",
            **diag,
        }

    # Pad and batch via DataLoader
    class SeqDataset(torch.utils.data.Dataset):
        def __init__(self, seqs):
            self.seqs = seqs
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return self.seqs[idx]

    def collate(batch):
        # batch: list of (items, y, rows) where rows may be absent in legacy tuples
        lengths = [len(x[0]) for x in batch]
        max_len = max(lengths)
        it_pad = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
        y_pad = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 to mask loss
        r_pad = torch.full((len(batch), max_len), -1, dtype=torch.long)
        for i, elem in enumerate(batch):
            if len(elem) == 3:
                it, y, rows = elem
            else:
                it, y = elem
                rows = None
            L = len(it)
            it_pad[i, :L] = torch.tensor(it, dtype=torch.long)
            y_pad[i, :L] = torch.tensor(y, dtype=torch.long)
            if rows is not None:
                r_pad[i, :L] = torch.tensor(rows, dtype=torch.long)
        return it_pad, y_pad, r_pad, torch.tensor(lengths, dtype=torch.long)

    test_loader = DataLoader(SeqDataset(test_seqs), batch_size=getattr(config, "DKT_BATCH", 32), shuffle=False, collate_fn=collate)

    class DKT(nn.Module):
        def __init__(self, vocab: int, emb: int = 32, hid: int = 64):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=pad_idx)
            self.lstm = nn.LSTM(emb, hid, batch_first=True)
            self.drop = nn.Dropout(p=float(getattr(config, "DKT_DROPOUT", 0.0)))
            self.out = nn.Linear(hid, 1)
        def forward(self, it, lengths):
            x = self.emb(it)
            # Pack padded for efficiency
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            h, _ = self.lstm(packed)
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
            h = self.drop(h)
            logits = self.out(h).squeeze(-1)
            return logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKT(vocab=vocab_size, emb=getattr(config, "DKT_EMB_DIM", 32), hid=getattr(config, "DKT_HID_DIM", 64)).to(device)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(
        model.parameters(), lr=float(getattr(config, "DKT_LR", 1e-3)), weight_decay=float(getattr(config, "DKT_WEIGHT_DECAY", 0.0))
    )

    # Prepare inner validation split over training students (for early stopping)
    best_val = float("inf")
    best_epoch = -1
    patience = int(getattr(config, "DKT_PATIENCE", 0))
    wait = 0

    students_with_seq = sorted({sid for sid, _ in [(s, None) for s in train_students]})
    # Filter to students that actually have sequences
    def has_seq(sid: str) -> bool:
        sub = df_seq.loc[df_seq[config.COL_ID] == sid]
        if sub.empty:
            return False
        y_series = pd.to_numeric(sub[config.COL_RESP], errors="coerce")
        return y_series.isin([0, 1]).sum() >= 1
    students_with_seq = [s for s in students_with_seq if has_seq(s)]
    if len(students_with_seq) >= 3:
        s_tr_in, s_va_in = train_test_split(students_with_seq, test_size=0.2, random_state=config.RANDOM_STATE, shuffle=True)
        train_seqs_in = build_sequences(sorted(s_tr_in))
        val_seqs = build_sequences(sorted(s_va_in))
    else:
        train_seqs_in = train_seqs
        val_seqs = []

    inner_train_loader = DataLoader(SeqDataset(train_seqs_in), batch_size=getattr(config, "DKT_BATCH", 32), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(SeqDataset(val_seqs), batch_size=getattr(config, "DKT_BATCH", 32), shuffle=False, collate_fn=collate) if val_seqs else None
    full_train_loader = DataLoader(SeqDataset(train_seqs), batch_size=getattr(config, "DKT_BATCH", 32), shuffle=True, collate_fn=collate)
    test_loader = DataLoader(SeqDataset(test_seqs), batch_size=getattr(config, "DKT_BATCH", 32), shuffle=False, collate_fn=collate)

    # Phase 1: Train on inner split with early stopping to find best epoch count
    def _train_one_epoch(mdl, loader, optimizer, criterion, dev):
        mdl.train()
        for it_pad, y_pad, r_pad, lengths in loader:
            it_pad = it_pad.to(dev)
            y_pad = y_pad.to(dev)
            lengths = lengths.to(dev)
            logits = mdl(it_pad, lengths)
            y_float = (y_pad == 1).float()
            mask = (y_pad != -1).float()
            if logits.size(1) > 1:
                logits_n = logits[:, :-1]
                y_n = y_float[:, 1:]
                m_n = mask[:, 1:]
                loss_raw = criterion(logits_n, y_n)
                loss = (loss_raw * m_n).sum() / (m_n.sum() + 1e-6)
            else:
                loss = (logits * 0).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _val_loss(mdl, loader, criterion, dev):
        mdl.eval()
        val_losses = []
        with torch.no_grad():
            for it_pad, y_pad, r_pad, lengths in loader:
                it_pad = it_pad.to(dev)
                y_pad = y_pad.to(dev)
                lengths = lengths.to(dev)
                logits = mdl(it_pad, lengths)
                y_float = (y_pad == 1).float()
                mask = (y_pad != -1).float()
                if logits.size(1) > 1:
                    logits_n = logits[:, :-1]
                    y_n = y_float[:, 1:]
                    m_n = mask[:, 1:]
                    l_raw = criterion(logits_n, y_n)
                    l = (l_raw * m_n).sum() / (m_n.sum() + 1e-6)
                    val_losses.append(l.item())
        return float(np.mean(val_losses)) if val_losses else float("inf")

    model.train()
    start_time = time.perf_counter()
    budget = getattr(config, "TRAIN_TIME_BUDGET_S", None)
    for epoch in range(max(1, config.EPOCHS_DKT)):
        if budget is not None and (time.perf_counter() - start_time) > float(budget):
            break
        _train_one_epoch(model, inner_train_loader, opt, crit, device)
        # Validation
        if val_loader is not None:
            val_loss = _val_loss(model, val_loader, crit, device)
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if patience > 0 and wait >= patience:
                    break

    # Phase 2: Retrain from scratch on ALL training data for best_epoch+1 epochs
    # ONLY if we have time budget remaining.
    elapsed = time.perf_counter() - start_time
    if budget is None or elapsed < float(budget):
        # Save Phase 1 weights in case Phase 2 yields nothing (e.g. 0 epochs)
        phase1_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        chosen_epochs = (best_epoch + 1) if best_epoch >= 0 else max(1, config.EPOCHS_DKT)
        model = DKT(vocab=vocab_size, emb=getattr(config, "DKT_EMB_DIM", 32), hid=getattr(config, "DKT_HID_DIM", 64)).to(device)
        opt = torch.optim.Adam(
            model.parameters(), lr=float(getattr(config, "DKT_LR", 1e-3)), weight_decay=float(getattr(config, "DKT_WEIGHT_DECAY", 0.0))
        )
        model.train()

        epochs_run = 0
        for epoch in range(chosen_epochs):
            if budget is not None and (time.perf_counter() - start_time) > float(budget):
                break
            _train_one_epoch(model, full_train_loader, opt, crit, device)
            epochs_run += 1

        # If we didn't complete even 1 epoch of Phase 2, revert to Phase 1 model
        if epochs_run == 0:
            model.load_state_dict(phase1_state)
            diag["phase2_aborted"] = True
    else:
        diag["phase2_skipped"] = True

    # Evaluate: collect all per-step predictions with valid labels
    model.eval()
    y_true_all: List[int] = []
    y_prob_all: List[float] = []
    row_idx_all: List[int] = []
    with torch.no_grad():
        for it_pad, y_pad, r_pad, lengths in test_loader:
            it_pad = it_pad.to(device)
            y_pad = y_pad.to(device)
            r_pad = r_pad.to(device)
            lengths = lengths.to(device)
            logits = model(it_pad, lengths)
            probs = torch.sigmoid(logits)
            # Next-item eval: use positions 1..L-1 for y_true and 0..L-2 for probabilities
            if logits.size(1) > 1:
                y_slice = y_pad[:, 1:]
                p_slice = probs[:, :-1]
                r_slice = r_pad[:, 1:]
                mask = (y_slice != -1)
                y_true = (y_slice[mask]).long().cpu().numpy()
                y_prob = (p_slice[mask]).float().cpu().numpy()
                r_vals = (r_slice[mask]).long().cpu().numpy()
                if y_true.size:
                    y_true_all.extend(y_true.tolist())
                    y_prob_all.extend(y_prob.tolist())
                    row_idx_all.extend(r_vals.tolist())

    if not y_true_all:
        return {
            "category": "Deep Learning",
            "name": "DKT (minimal)",
            "why": why,
            "error": "No eval points",
            **diag,
        }

    return {
        "category": "Deep Learning",
        "name": "DKT (minimal)",
        "why": why,
        "y_true": np.array(y_true_all),
        "y_prob": np.array(y_prob_all),
        "test_rows": np.array(row_idx_all, dtype=int),
        **diag,
        "best_epoch": int(best_epoch) if best_epoch >= 0 else None,
    }
