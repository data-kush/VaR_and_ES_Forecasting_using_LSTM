"""
lstm_model.py – Stateful LSTM for joint VaR/ES forecasting.

Key design decisions (all motivated by the bug analysis and literature):

  1. STATEFUL processing  (Qiu, Lazar & Nakata 2024)
     The LSTM hidden state carries forward across the ENTIRE time series.
     This lets the model learn cross-window patterns like volatility
     persistence and regime transitions — the single largest improvement
     over the prior (windowed, stateless) implementation.

  2. CORRECT quantile loss
     The violation-amplification multiplier is placed on the VIOLATION
     side (e < 0), NOT the non-violation side.  Placing it on the wrong
     side was the root cause of the model forecasting ~14 % VaR when
     trained for the 5 % level.

  3. FZ0 loss  (Fissler & Ziegel 2016)
     The only strictly consistent (elicitable) scoring function for the
     joint (VaR, ES) pair.  Replaces the ad-hoc λ × ES_coverage_loss.

  4. NO shuffle in training DataLoader
     Financial time series must remain in chronological order so the
     LSTM cell state has meaningful causal temporal context.

  5. Truncated BPTT
     Hidden state values carry forward; gradients are detached at each
     chunk boundary (length TBPTT_LEN) to keep memory bounded on CPU.

  6. Conservative output parameterisation
     Both VaR and ES use −softplus(·) to enforce strict negativity,
     with an ES ≤ VaR constraint applied analytically.

  7. Validation calibration  (conformal-style shift)
     After training, the alpha-quantile of val-set residuals is used
     as a bias correction that is applied to the test predictions.

Architecture (CPU-optimised, i5 / 8 GB RAM):
  Input  : 6 features per time step (lagged returns, EWMA-vol, etc.)
  LSTM   : 1 layer, hidden_size=32
  Heads  : 2 × (Linear(32→16) → ELU → Linear(16→1) → −softplus)
  Params : ~10 K  (trains in 1–3 min per dataset on i5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.utils import clip_grad_norm_
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

N_FEATURES  = 6
TBPTT_LEN   = 64   # Truncated-BPTT chunk length


# ─────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────

def fz0_loss(var_pred: torch.Tensor,
             es_pred:  torch.Tensor,
             target:   torch.Tensor,
             alpha:    float) -> torch.Tensor:
    """
    Fissler-Ziegel FZ0 loss (Theorem 5.2, Fissler & Ziegel 2016).

    L(r, v, e; α) = −[1/(α·e)] · I(r ≤ v)·(v−r) + v/e + log(−e) − 1

    This is the unique strictly-consistent scoring function for the
    joint (VaR_α, ES_α) functional.  Minimising it guarantees that
    the model is simultaneously trained for correct VaR coverage AND
    correct ES magnitude.

    Inputs  : var_pred (T,), es_pred (T,), target (T,)  — all negative
    """
    v = var_pred
    e = torch.clamp(es_pred, max=-1e-4)   # ES must be strictly negative

    indicator = (target <= v).float()

    term1 = -(1.0 / (alpha * e)) * indicator * (v - target)
    term2 = v / e
    term3 = torch.log(-e)

    loss = torch.mean(term1 + term2 + term3 - 1.0)

    # Guard against NaN/Inf during early training
    if not torch.isfinite(loss):
        loss = torch.tensor(1.0, requires_grad=True)
    return loss


def quantile_loss_conservative(var_pred: torch.Tensor,
                                target:   torch.Tensor,
                                alpha:    float,
                                viol_mult: float = 1.5) -> torch.Tensor:
    """
    Quantile (pinball) loss with the amplification multiplier on the
    VIOLATION side (e < 0), not the non-violation side.

    This shifts the effective quantile downward (more conservative):
      effective α' = α / [α + (1−α)·viol_mult]

    At α=0.05, viol_mult=1.5:  α' ≈ 0.033  (gives a cushion below 5 %)

    Used as a secondary regulariser alongside the FZ0 loss.
    """
    e = target - var_pred
    return torch.mean(
        torch.where(e >= 0,
                    alpha * e,                           # non-violation: standard
                    (alpha - 1.0) * viol_mult * e)       # violation: amplified
    )


def combined_loss(var_pred: torch.Tensor,
                  es_pred:  torch.Tensor,
                  target:   torch.Tensor,
                  alpha:    float,
                  fz0_weight:  float = 0.7,
                  q_weight:    float = 0.3) -> torch.Tensor:
    """
    Weighted combination of FZ0 and conservative quantile losses.
    The FZ0 loss drives the primary joint optimisation;
    the quantile loss adds an extra gradient signal for VaR coverage.
    """
    l_fz0 = fz0_loss(var_pred, es_pred, target, alpha)
    l_q   = quantile_loss_conservative(var_pred, target, alpha)
    return fz0_weight * l_fz0 + q_weight * l_q


# ─────────────────────────────────────────────────────────────────
# Feature engineering (look-ahead safe: uses only lagged values)
# ─────────────────────────────────────────────────────────────────

def build_features(returns: np.ndarray,
                   ewma_lambda: float = 0.94) -> np.ndarray:
    """
    Build 6-feature matrix (T × 6).

    All features at position t use information up to t−1 only,
    so there is strictly NO look-ahead bias.

    Features (lagged by 1):
      col 0 : r_{t-1}            lagged return
      col 1 : |r_{t-1}|          lagged absolute return (vol proxy)
      col 2 : ewma_vol_{t-1}     RiskMetrics EWMA conditional std
      col 3 : roll5_std_{t-1}    5-day rolling std
      col 4 : roll22_std_{t-1}   22-day rolling std
      col 5 : r²_{t-1}           lagged squared return
    """
    r = np.asarray(returns, dtype=np.float32)
    T = len(r)

    # EWMA variance (σ²_t = λ·σ²_{t-1} + (1−λ)·r²_{t-1})
    ewma_var = np.full(T, max(np.nanvar(r[:min(30, T)]), 1e-10), dtype=np.float32)
    for t in range(1, T):
        rv = r[t - 1] if np.isfinite(r[t - 1]) else 0.0
        ewma_var[t] = ewma_lambda * ewma_var[t - 1] + (1 - ewma_lambda) * rv ** 2
    ewma_vol = np.sqrt(np.maximum(ewma_var, 1e-12))

    rs     = pd.Series(r)
    roll5  = rs.rolling(5,  min_periods=2).std().values.astype(np.float32)
    roll22 = rs.rolling(22, min_periods=5).std().values.astype(np.float32)

    # Forward-fill NaN from rolling windows
    for arr in (roll5, roll22):
        m = ~np.isfinite(arr)
        if m.any():
            arr[m] = ewma_vol[m]   # fall back to EWMA vol

    # Lag all features by 1 position (shift right)
    r_lag    = np.roll(r,        1).astype(np.float32)
    abs_lag  = np.roll(np.abs(r),1).astype(np.float32)
    evol_lag = np.roll(ewma_vol, 1).astype(np.float32)
    r5_lag   = np.roll(roll5,    1).astype(np.float32)
    r22_lag  = np.roll(roll22,   1).astype(np.float32)
    r2_lag   = np.roll(r ** 2,   1).astype(np.float32)

    # Position 0 would contain the "rolled-around" last value — zero it out
    for arr in (r_lag, abs_lag, evol_lag, r5_lag, r22_lag, r2_lag):
        arr[0] = 0.0

    feats = np.column_stack([r_lag, abs_lag, evol_lag, r5_lag, r22_lag, r2_lag])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


class FeatureNormalizer:
    """Zero-mean / unit-std normalisation.  Fit on training data only."""

    def fit(self, X: np.ndarray) -> 'FeatureNormalizer':
        self.mu  = X.mean(axis=0, keepdims=True)
        self.sig = X.std( axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sig).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Stateful LSTM model
# ─────────────────────────────────────────────────────────────────

class StatefulLSTMVaR(nn.Module):
    """
    Stateful single-layer LSTM for continuous VaR/ES forecasting.

    The LSTM hidden state (h, c) is preserved across consecutive
    time-step chunks during both training (TBPTT) and inference,
    allowing the model to maintain memory of the ENTIRE historical
    trajectory — key for learning volatility clustering.

    Architecture
    ------------
    Input  : (1, chunk_len, N_FEATURES)   — batch always = 1
    LSTM   : hidden_size neurons
    Heads  : 2 × [Linear(hidden→16) → ELU → Linear(16→1)] → −softplus
    Output : var (chunk_len,), es (chunk_len,)  — both negative
    """

    def __init__(self,
                 input_size:  int   = N_FEATURES,
                 hidden_size: int   = 32,
                 dropout:     float = 0.15):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)

        def _head():
            return nn.Sequential(
                nn.Linear(hidden_size, 16),
                nn.ELU(),
                nn.Linear(16, 1),
            )

        self.var_head = _head()
        self.es_head  = _head()
        self._hidden  = None

    # ── hidden-state management ──────────────────────────────────

    def init_hidden(self):
        device = next(self.parameters()).device
        self._hidden = (
            torch.zeros(1, 1, self.hidden_size, device=device),
            torch.zeros(1, 1, self.hidden_size, device=device),
        )

    def detach_hidden(self):
        """Detach hidden state for Truncated-BPTT."""
        if self._hidden is not None:
            h, c = self._hidden
            self._hidden = (h.detach(), c.detach())

    # ── forward pass ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        x : (1, seq_len, N_FEATURES)

        Returns var (seq_len,) and es (seq_len,) — both strictly negative.
        ES is further clamped to be ≤ VaR (more extreme loss).
        """
        out, self._hidden = self.lstm(x, self._hidden)
        out = self.drop(out)
        h   = out.squeeze(0)        # (seq_len, hidden_size)

        # −softplus ensures strict negativity
        var = -F.softplus(self.var_head(h).squeeze(-1)) - 1e-4
        es  = -F.softplus(self.es_head(h).squeeze(-1))  - 1e-4

        # Enforce ES ≤ VaR (ES is the average tail loss, always worse)
        es = torch.min(es, var - 1e-4)

        return var, es


# ─────────────────────────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────────────────────────

def train_stateful_lstm(df:          pd.DataFrame,
                        alpha:       float = 0.05,
                        hidden_size: int   = 32,
                        dropout:     float = 0.15,
                        epochs:      int   = 60,
                        lr:          float = 3e-4,
                        patience:    int   = 10,
                        train_ratio: float = 0.70,
                        val_ratio:   float = 0.15,
                        aug_series:  list  = None,
                        tbptt_len:   int   = TBPTT_LEN) -> dict:
    """
    Train a stateful LSTM for VaR/ES at significance level `alpha`.

    Parameters
    ----------
    df          : pd.DataFrame with 'returns' column (and optionally 'Date')
    alpha       : coverage level  (e.g. 0.05)
    hidden_size : LSTM hidden units  (32 is efficient on CPU)
    dropout     : dropout rate
    epochs      : max training epochs
    lr          : initial Adam learning rate
    patience    : early-stopping patience on validation FZ0 loss
    train_ratio : fraction of data for training
    val_ratio   : fraction of data for validation
    aug_series  : list of augmented return arrays (same length as training)
    tbptt_len   : TBPTT chunk size (how many steps before detaching grad)

    Returns
    -------
    dict with keys: dates, actual_returns, var_forecasts, es_forecasts,
                    train_loss, val_loss, metrics, alpha,
                    train_size, val_size, test_size, epochs_run
    """
    from backtests import evaluate_forecast

    # ── 1. Prepare returns & dates ───────────────────────────────
    valid = df['returns'].notna()
    returns_all = df['returns'][valid].values.astype(np.float32)
    if 'Date' in df.columns:
        dates_all = pd.to_datetime(df['Date'][valid]).dt.strftime('%Y-%m-%d').values
    else:
        dates_all = np.arange(len(returns_all)).astype(str)

    T       = len(returns_all)
    n_train = int(T * train_ratio)
    n_val   = int(T * val_ratio)
    n_test  = T - n_train - n_val

    if n_train < 200:
        raise ValueError(f"Insufficient training data: {n_train} rows")

    r_train = returns_all[:n_train]
    r_val   = returns_all[n_train: n_train + n_val]
    r_test  = returns_all[n_train + n_val:]
    test_dates = dates_all[n_train + n_val:]

    # ── 2. Feature matrix & normalisation ───────────────────────
    feats_all = build_features(returns_all)
    normalizer = FeatureNormalizer().fit(feats_all[:n_train])
    feats_all_n = normalizer.transform(feats_all)

    f_train = feats_all_n[:n_train]
    f_val   = feats_all_n[n_train: n_train + n_val]
    f_test  = feats_all_n[n_train + n_val:]

    # ── 3. Build augmented training corpus ───────────────────────
    # Original training data + synthetic series
    train_corpus_X = [torch.from_numpy(f_train).unsqueeze(0)]   # (1, T_tr, 6)
    train_corpus_y = [torch.from_numpy(r_train)]

    if aug_series:
        for aug_r in aug_series:
            aug_r = np.asarray(aug_r, dtype=np.float32)[:n_train]
            aug_f = build_features(aug_r)
            aug_fn = normalizer.transform(aug_f)
            train_corpus_X.append(torch.from_numpy(aug_fn).unsqueeze(0))
            train_corpus_y.append(torch.from_numpy(aug_r))

    # Validation tensors (single sequence)
    X_val = torch.from_numpy(f_val).unsqueeze(0)   # (1, n_val, 6)
    y_val = torch.from_numpy(r_val)                 # (n_val,)

    # ── 4. Initialise model & optimiser ─────────────────────────
    model     = StatefulLSTMVaR(input_size=N_FEATURES,
                                hidden_size=hidden_size,
                                dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5, min_lr=1e-6)

    # ── 5. Training loop (Truncated BPTT) ────────────────────────
    train_hist, val_hist = [], []
    best_val   = float('inf')
    best_state = None
    wait       = 0

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        ep_steps = 0

        # Iterate over all training series (original + augmented)
        for X_seq, y_seq in zip(train_corpus_X, train_corpus_y):
            model.init_hidden()
            T_seq = X_seq.shape[1]

            for chunk_start in range(0, T_seq, tbptt_len):
                chunk_end = min(chunk_start + tbptt_len, T_seq)
                Xb = X_seq[:, chunk_start:chunk_end, :]  # (1, chunk, 6)
                yb = y_seq[chunk_start:chunk_end]         # (chunk,)

                model.detach_hidden()
                optimizer.zero_grad()

                var_p, es_p = model(Xb)
                loss = combined_loss(var_p, es_p, yb, alpha)

                if torch.isfinite(loss):
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    ep_loss  += loss.item() * len(yb)
                    ep_steps += len(yb)

        ep_loss = ep_loss / max(ep_steps, 1)
        train_hist.append(round(float(ep_loss), 7))

        # ── Validation ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            model.init_hidden()
            v_loss_total = 0.0
            v_steps = 0
            for chunk_start in range(0, n_val, tbptt_len):
                chunk_end = min(chunk_start + tbptt_len, n_val)
                Xb = X_val[:, chunk_start:chunk_end, :]
                yb = y_val[chunk_start:chunk_end]
                model.detach_hidden()
                var_p, es_p = model(Xb)
                v_loss = combined_loss(var_p, es_p, yb, alpha)
                if torch.isfinite(v_loss):
                    v_loss_total += v_loss.item() * len(yb)
                    v_steps      += len(yb)
            v_loss_val = v_loss_total / max(v_steps, 1)
        val_hist.append(round(float(v_loss_val), 7))
        scheduler.step(v_loss_val)

        # ── Early stopping ───────────────────────────────────────
        if v_loss_val < best_val - 1e-7:
            best_val   = v_loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── 6. Calibration on validation set ────────────────────────
    # Compute alpha-quantile of (actual - predicted) residuals on val.
    # Negative cal_offset  → VaR was too loose → shift down (more conservative).
    model.eval()
    with torch.no_grad():
        model.init_hidden()
        val_var_chunks = []
        for chunk_start in range(0, n_val, tbptt_len):
            chunk_end = min(chunk_start + tbptt_len, n_val)
            Xb = X_val[:, chunk_start:chunk_end, :]
            model.detach_hidden()
            var_p, _ = model(Xb)
            val_var_chunks.append(var_p.numpy())
        val_var_np = np.concatenate(val_var_chunks).astype(float)

    cal_offset = float(np.quantile(r_val.astype(float) - val_var_np, alpha))

    # ── 7. Inference on test set ─────────────────────────────────
    # Warm up on train + val to bring hidden state to the right regime
    model.eval()
    with torch.no_grad():
        # Warm-up through training period
        X_warmup = torch.from_numpy(f_train).unsqueeze(0)
        model.init_hidden()
        for cs in range(0, n_train, tbptt_len):
            ce = min(cs + tbptt_len, n_train)
            model.detach_hidden()
            model(X_warmup[:, cs:ce, :])

        # Warm-up through validation period
        for cs in range(0, n_val, tbptt_len):
            ce = min(cs + tbptt_len, n_val)
            model.detach_hidden()
            model(X_val[:, cs:ce, :])

        # Predict on test period
        X_test = torch.from_numpy(f_test).unsqueeze(0)
        test_var_chunks, test_es_chunks = [], []
        for cs in range(0, n_test, tbptt_len):
            ce = min(cs + tbptt_len, n_test)
            model.detach_hidden()
            vp, ep = model(X_test[:, cs:ce, :])
            test_var_chunks.append(vp.numpy())
            test_es_chunks.append(ep.numpy())

    var_f = np.concatenate(test_var_chunks).astype(float) + cal_offset
    es_f  = np.concatenate(test_es_chunks).astype(float) + cal_offset
    es_f  = np.minimum(es_f, var_f - 1e-6)    # enforce ES ≤ VaR

    # ── 8. Evaluate ──────────────────────────────────────────────
    ret_te  = r_test.astype(float)
    metrics = evaluate_forecast(ret_te, var_f, es_f, alpha)

    return {
        'dates':          test_dates.tolist(),
        'actual_returns': ret_te.tolist(),
        'var_forecasts':  var_f.tolist(),
        'es_forecasts':   es_f.tolist(),
        'train_loss':     train_hist,
        'val_loss':       val_hist,
        'metrics':        metrics,
        'alpha':          alpha,
        'train_size':     int(n_train),
        'val_size':       int(n_val),
        'test_size':      int(n_test),
        'epochs_run':     len(train_hist),
    }
