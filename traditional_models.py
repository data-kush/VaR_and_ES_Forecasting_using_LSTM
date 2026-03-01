"""
traditional_models.py – Benchmark VaR/ES forecasting models.

Models implemented:
  1. GARCH(1,1) with skewed-t distribution  (rolling 1-step-ahead)
  2. GAS FZ1F  (Patton, Ziegel & Chen 2019)  – score-driven dynamics
     estimated by minimising the FZ0 loss (same loss as the LSTM)
  3. Historical Simulation (HS)  – rolling 250-day empirical quantile
  4. Filtered Historical Simulation (FHS)  – HS on EWMA-standardised residuals

All models produce out-of-sample (test-period) VaR and ES forecasts,
evaluated with Kupiec and Christoffersen tests.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────

def _load_returns(df: pd.DataFrame,
                  train_ratio: float,
                  val_ratio:   float):
    """Extract returns array and train/val/test split indices."""
    r = df['returns'].dropna().values.astype(float)
    dates = (pd.to_datetime(df['Date'][df['returns'].notna()])
             .dt.strftime('%Y-%m-%d').values
             if 'Date' in df.columns
             else np.arange(len(r)).astype(str))
    T = len(r)
    n_tr = int(T * train_ratio)
    n_va = int(T * val_ratio)
    return r, dates, T, n_tr, n_va


def _ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA conditional standard deviation."""
    r  = np.asarray(returns, dtype=float)
    n  = len(r)
    v2 = np.full(n, max(np.nanvar(r[:30]), 1e-10))
    for t in range(1, n):
        rv    = r[t - 1] if np.isfinite(r[t - 1]) else 0.0
        v2[t] = lam * v2[t - 1] + (1 - lam) * rv ** 2
    return np.sqrt(np.maximum(v2, 1e-12))


def _build_result(r_test, var_f, es_f, test_dates, alpha,
                  n_train, n_val, extra=None):
    """Assemble a standardised result dict (shared across all models)."""
    from backtests import evaluate_forecast
    metrics = evaluate_forecast(r_test, var_f, es_f, alpha)
    base = {
        'dates':          test_dates.tolist() if hasattr(test_dates, 'tolist') else list(test_dates),
        'actual_returns': r_test.tolist(),
        'var_forecasts':  np.where(np.isfinite(var_f), var_f, np.nanmean(var_f)).tolist(),
        'es_forecasts':   np.where(np.isfinite(es_f),  es_f,  np.nanmean(es_f)).tolist(),
        'metrics':        metrics,
        'alpha':          alpha,
        'train_size':     n_train,
        'val_size':       n_val,
        'test_size':      len(r_test),
    }
    if extra:
        base.update(extra)
    return base


# ─────────────────────────────────────────────────────────────────
# 1. Rolling GARCH(1,1)-skewed-t
# ─────────────────────────────────────────────────────────────────

def fit_rolling_garch(df:          pd.DataFrame,
                      alpha:       float = 0.05,
                      train_ratio: float = 0.70,
                      val_ratio:   float = 0.15) -> dict:
    """
    Rolling GARCH(1,1) with skewed-t innovations.

    Fit on training data.  For the test period, propagate conditional
    variance using the fitted GARCH parameters — fast and accurate
    compared to re-fitting at every step.
    """
    from arch import arch_model as _arch

    r, dates, T, n_tr, n_va = _load_returns(df, train_ratio, val_ratio)

    r_train = r[:n_tr] * 100.0   # arch uses % returns

    # Fit on training data
    am  = _arch(r_train, vol='Garch', p=1, q=1, dist='skewt')
    res = am.fit(disp='off', show_warning=False)

    mu_p  = float(res.params.get('mu',      0.0))
    omega = float(res.params.get('omega',   max(1e-6, np.var(r_train) * 0.05)))
    a1    = float(np.clip(res.params.get('alpha[1]', 0.10), 0.01, 0.30))
    b1    = float(np.clip(res.params.get('beta[1]',  0.85), 0.50, 0.97))
    nu    = float(np.clip(res.params.get('nu',       8.0),  4.0,  30.0))
    lam   = float(np.clip(res.params.get('lambda',  -0.1), -0.5,  0.5))

    if a1 + b1 >= 1.0:
        b1 = 0.97 - a1

    mu = mu_p / 100.0   # back to decimal

    # Propagate conditional variance through val → test
    sigma2 = float(res.conditional_volatility[-1] / 100.0) ** 2

    # Propagate through validation period
    for rt in r[n_tr: n_tr + n_va]:
        sigma2 = omega / 1e4 + a1 * rt ** 2 + b1 * sigma2

    # Forecast on test period
    r_test  = r[n_tr + n_va:]
    n_te    = len(r_test)
    var_f   = np.empty(n_te)
    es_f    = np.empty(n_te)

    for i, rt in enumerate(r_test):
        sigma  = float(np.sqrt(max(sigma2, 1e-12)))
        q_t    = float(sp_stats.t.ppf(alpha, df=nu))
        pdf_q  = float(sp_stats.t.pdf(q_t,  df=nu))

        var_f[i] = mu + q_t * sigma
        es_f[i]  = mu - sigma * pdf_q / alpha * (nu + q_t ** 2) / max(nu - 1, 1)
        es_f[i]  = min(es_f[i], var_f[i] - 1e-6)

        sigma2 = omega / 1e4 + a1 * rt ** 2 + b1 * sigma2

    test_dates = dates[n_tr + n_va:]

    extra = {
        'aic': round(res.aic, 2),
        'bic': round(res.bic, 2),
    }
    return _build_result(r_test, var_f, es_f, test_dates, alpha,
                         n_tr, n_va, extra)


# ─────────────────────────────────────────────────────────────────
# 2. GAS / FZ1F model  (Patton, Ziegel & Chen 2019)
# ─────────────────────────────────────────────────────────────────

def fit_gas_fz1f(df:          pd.DataFrame,
                 alpha:       float = 0.05,
                 train_ratio: float = 0.70,
                 val_ratio:   float = 0.15) -> dict:
    """
    Generalised Autoregressive Score FZ1F model.

    Dynamics (Patton, Ziegel & Chen 2019, eq. 3.1):
      v_t = ω_v + A_v · δ_{t-1} + B_v · v_{t-1}
      e_t = ω_e + A_e · ε_{t-1} + B_e · e_{t-1}

    where the score innovations are:
      δ_t = α − I(r_t ≤ v_t)
      ε_t = I(r_t ≤ v_t)/α · r_t/e_t − 1

    Parameters estimated by minimising the FZ0 loss on the training set.
    """
    r, dates, T, n_tr, n_va = _load_returns(df, train_ratio, val_ratio)

    r_train = r[:n_tr]
    r_test  = r[n_tr + n_va:]
    n_te    = len(r_test)

    # ── Helper: simulate the GAS series given parameters ─────────
    def _simulate(params, returns):
        omega_v, A_v, B_v, omega_e, A_e, B_e = params
        n  = len(returns)
        v0 = float(np.quantile(returns[:max(50, int(n * 0.1))], alpha))
        tail = returns[:max(50, int(n * 0.1))]
        tail = tail[tail <= v0]
        e0   = float(np.mean(tail)) if len(tail) > 0 else v0 * 1.3
        e0   = min(e0, v0 - 1e-6)

        vs  = np.empty(n, dtype=float)
        es_ = np.empty(n, dtype=float)
        v, e = v0, e0

        for t, rt in enumerate(returns):
            vs[t]  = v
            es_[t] = e
            delta   = alpha - float(rt <= v)
            epsilon = (float(rt <= v) / alpha * rt / e - 1.0
                       if e < -1e-8 else 0.0)
            v_new = omega_v + A_v * delta   + B_v * v
            e_new = omega_e + A_e * epsilon + B_e * e
            e_new = min(e_new, v_new - 1e-6)
            if e_new >= -1e-8:
                e_new = v_new * 1.2
            v, e = v_new, e_new

        return vs, es_

    # ── Objective: FZ0 loss on training data ─────────────────────
    def _objective(params):
        vs, es_ = _simulate(params, r_train)
        e_safe  = np.minimum(es_, -1e-6)
        ind     = (r_train <= vs).astype(float)
        term1   = -(1.0 / (alpha * e_safe)) * ind * (vs - r_train)
        term2   = vs / e_safe
        term3   = np.log(-e_safe)
        loss    = float(np.mean(term1 + term2 + term3 - 1.0))
        return loss if np.isfinite(loss) else 1e10

    # ── Initial parameters ────────────────────────────────────────
    q0   = float(np.quantile(r_train, alpha))
    x0   = [q0 * 0.01, 0.05, 0.90, q0 * 0.01 * 1.2, 0.05, 0.90]

    result = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5,
                               'adaptive': True})

    # ── Forecast on test: warm-up through train + val ─────────────
    all_r  = r[:n_tr + n_va + n_te]
    vs_all, es_all = _simulate(result.x, all_r)

    var_f  = vs_all[n_tr + n_va:]
    es_f   = es_all[n_tr + n_va:]
    es_f   = np.minimum(es_f, var_f - 1e-6)

    test_dates = dates[n_tr + n_va:]
    return _build_result(r_test, var_f, es_f, test_dates, alpha, n_tr, n_va)


# ─────────────────────────────────────────────────────────────────
# 3. Historical Simulation (HS)
# ─────────────────────────────────────────────────────────────────

def rolling_hs(df:          pd.DataFrame,
               alpha:       float = 0.05,
               window:      int   = 250,
               train_ratio: float = 0.70,
               val_ratio:   float = 0.15) -> dict:
    """
    Plain rolling Historical Simulation.
    VaR_t  = empirical alpha-quantile of r_{t-window..t-1}
    ES_t   = mean of returns below VaR_t
    """
    r, dates, T, n_tr, n_va = _load_returns(df, train_ratio, val_ratio)

    r_test     = r[n_tr + n_va:]
    test_dates = dates[n_tr + n_va:]
    n_te       = len(r_test)

    var_f = np.empty(n_te)
    es_f  = np.empty(n_te)

    for i in range(n_te):
        t_abs = n_tr + n_va + i
        start = max(0, t_abs - window)
        hist  = r[start: t_abs]

        if len(hist) >= 20:
            q         = float(np.quantile(hist, alpha))
            tail      = hist[hist <= q]
            var_f[i]  = q
            es_f[i]   = float(np.mean(tail)) if len(tail) > 0 else q * 1.25
        else:
            fallback  = float(np.quantile(r[:n_tr], alpha))
            var_f[i]  = fallback
            es_f[i]   = fallback * 1.25

        es_f[i] = min(es_f[i], var_f[i] - 1e-6)

    return _build_result(r_test, var_f, es_f, test_dates, alpha, n_tr, n_va)


# ─────────────────────────────────────────────────────────────────
# 4. Filtered Historical Simulation (FHS)
# ─────────────────────────────────────────────────────────────────

def rolling_fhs(df:           pd.DataFrame,
                alpha:        float = 0.05,
                window:       int   = 250,
                train_ratio:  float = 0.70,
                val_ratio:    float = 0.15,
                ewma_lambda:  float = 0.94) -> dict:
    """
    Filtered Historical Simulation (Hull & White 1998).

    1. Estimate time-varying σ_t via EWMA (λ=0.94).
    2. Standardise residuals z_t = r_t / σ_t.
    3. Empirical alpha-quantile of z_{t-window..t-1} → rescale by σ_t.

    FHS passes Kupiec/Christoffersen much more reliably than plain HS
    because EWMA adapts VaR to current volatility regimes.
    """
    r, dates, T, n_tr, n_va = _load_returns(df, train_ratio, val_ratio)

    ewma_vol = _ewma_vol(r, ewma_lambda)
    mu       = float(np.mean(r[:n_tr]))
    z        = (r - mu) / ewma_vol       # standardised residuals

    r_test     = r[n_tr + n_va:]
    test_dates = dates[n_tr + n_va:]
    n_te       = len(r_test)

    var_f = np.empty(n_te)
    es_f  = np.empty(n_te)

    # Fallback Student-t quantile (conservative, ν=4)
    q_fb  = float(sp_stats.t.ppf(alpha, df=4))
    es_fb = float(-sp_stats.t.pdf(q_fb, df=4) / alpha * (4 + q_fb ** 2) / 3)

    for i in range(n_te):
        t_abs  = n_tr + n_va + i
        sigma  = float(ewma_vol[t_abs])
        start  = max(0, t_abs - window)
        z_win  = z[start: t_abs]
        z_clean = z_win[np.isfinite(z_win)]

        if len(z_clean) >= 20:
            q_z        = float(np.quantile(z_clean, alpha))
            var_f[i]   = mu + q_z * sigma
            tail_z     = z_clean[z_clean <= q_z]
            es_z       = float(np.mean(tail_z)) if len(tail_z) > 0 else q_z * 1.25
            es_f[i]    = mu + es_z * sigma
        else:
            var_f[i]   = mu + q_fb  * sigma
            es_f[i]    = mu + es_fb * sigma

        es_f[i] = min(es_f[i], var_f[i] - 1e-6)

    return _build_result(r_test, var_f, es_f, test_dates, alpha, n_tr, n_va)


# ─────────────────────────────────────────────────────────────────
# Convenience: run all traditional models at once
# ─────────────────────────────────────────────────────────────────

def run_all_traditional(df:          pd.DataFrame,
                        alpha:       float = 0.05,
                        train_ratio: float = 0.70,
                        val_ratio:   float = 0.15,
                        verbose:     bool  = True) -> dict:
    """
    Run GARCH, GAS-FZ1F, HS, and FHS on `df` and return a dict
    keyed by model name.
    """
    results = {}
    models  = [
        ('garch', fit_rolling_garch),
        ('gas',   fit_gas_fz1f),
        ('hs',    rolling_hs),
        ('fhs',   rolling_fhs),
    ]

    for name, fn in models:
        if verbose:
            print(f"    Running {name.upper()}…", end=' ', flush=True)
        try:
            res = fn(df, alpha=alpha, train_ratio=train_ratio, val_ratio=val_ratio)
            results[name] = res
            kup  = res['metrics']['kupiec']
            cc   = res['metrics']['christoffersen']
            if verbose:
                print(f"violations={kup['violations']}/{kup['expected_violations']}  "
                      f"Kupiec={'PASS' if kup['pass'] else 'FAIL'}  "
                      f"CC={'PASS' if cc['pass'] else 'FAIL'}")
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            results[name] = None

    return results
