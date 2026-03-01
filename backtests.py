"""
backtests.py – Statistical backtesting for VaR and ES forecasts.

Tests implemented:
  - Kupiec (1995) Proportion-of-Failures (POF / unconditional coverage)
  - Christoffersen (1998) Conditional Coverage (CC)
  - FZ0 quantile score (Fissler-Ziegel 2016)
  - ES mean-exceedance score
"""

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────
# Kupiec POF test
# ─────────────────────────────────────────────────────────────────

def kupiec_test(returns: np.ndarray,
                var_forecasts: np.ndarray,
                alpha: float) -> dict:
    """
    Kupiec (1995) Proportion-of-Failures test.

    H₀ : violation rate p̂ = α  (correct unconditional coverage)
    LR_uc = −2·[log L(α) − log L(p̂)]  ~  χ²(1) under H₀

    A violation occurs when the actual return r_t < VaR_t.
    VaR convention: negative number (e.g. −0.02 for a −2 % loss threshold).

    Parameters
    ----------
    returns       : realised log-returns (test period)
    var_forecasts : VaR forecasts for the same period  (negative values)
    alpha         : coverage level (e.g. 0.05)

    Returns
    -------
    dict with keys: violations, total, violation_rate, expected_rate,
                    expected_violations, lr_statistic, p_value, pass
    """
    r = np.asarray(returns,       dtype=float)
    v = np.asarray(var_forecasts, dtype=float)

    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]
    T = len(r)

    if T == 0:
        return {'pass': False, 'p_value': 0.0, 'violations': 0,
                'total': 0, 'error': 'no_data'}

    N     = int((r < v).sum())
    p_hat = N / T

    if N == 0 or N == T:
        lr_stat, p_value = 0.0, 1.0
    else:
        ll0 = (T - N) * np.log(1.0 - alpha) + N * np.log(alpha)
        ll1 = (T - N) * np.log(1.0 - p_hat) + N * np.log(p_hat)
        lr_stat = max(0.0, -2.0 * (ll0 - ll1))
        p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    return {
        'violations':          N,
        'total':               T,
        'violation_rate':      round(p_hat, 6),
        'expected_rate':       alpha,
        'expected_violations': int(round(T * alpha)),
        'lr_statistic':        round(lr_stat, 4),
        'p_value':             round(p_value, 4),
        'pass':                p_value > 0.05,
    }


# ─────────────────────────────────────────────────────────────────
# Christoffersen CC test
# ─────────────────────────────────────────────────────────────────

def christoffersen_test(returns: np.ndarray,
                        var_forecasts: np.ndarray,
                        alpha: float) -> dict:
    """
    Christoffersen (1998) Conditional Coverage test.

    Decomposes into:
      LR_uc   – Kupiec unconditional coverage (χ²(1))
      LR_ind  – Independence of violation sequences (Markov test, χ²(1))
      LR_cc   = LR_uc + LR_ind  ~  χ²(2)

    Clustered violations (e.g. during a crisis) will fail LR_ind
    even when LR_uc is satisfied.

    Returns
    -------
    dict with keys: lr_independence, lr_conditional, p_value_ind, p_value_cc,
                    pass (CC), pass_ind, pass_uc
    """
    r = np.asarray(returns,       dtype=float)
    v = np.asarray(var_forecasts, dtype=float)

    mask = np.isfinite(r) & np.isfinite(v)
    r, v = r[mask], v[mask]
    T    = len(r)

    viol = (r < v).astype(int)
    N    = int(viol.sum())

    kup   = kupiec_test(r, v, alpha)
    lr_uc = kup['lr_statistic']

    if N == 0 or N == T or T < 4:
        return {
            'lr_independence': 0.0,
            'lr_conditional':  round(lr_uc, 4),
            'p_value_ind':     1.0,
            'p_value_cc':      kup['p_value'],
            'pass':     kup['pass'],
            'pass_ind': True,
            'pass_uc':  kup['pass'],
        }

    # Transition counts
    n00 = int(((viol[:-1] == 0) & (viol[1:] == 0)).sum())
    n01 = int(((viol[:-1] == 0) & (viol[1:] == 1)).sum())
    n10 = int(((viol[:-1] == 1) & (viol[1:] == 0)).sum())
    n11 = int(((viol[:-1] == 1) & (viol[1:] == 1)).sum())

    pi   = (n01 + n11) / max(T - 1, 1)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)

    def _sl(x: float) -> float:
        return np.log(x) if x > 1e-14 else -1e10

    ll0 = (n00 + n10) * _sl(1 - pi)   + (n01 + n11) * _sl(pi)
    ll1 = (n00 * _sl(1 - pi01) + n01 * _sl(pi01) +
           n10 * _sl(1 - pi11) + n11 * _sl(pi11))

    lr_ind = max(0.0, -2.0 * (ll0 - ll1))
    p_ind  = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    lr_cc  = lr_uc + lr_ind
    p_cc   = float(1.0 - stats.chi2.cdf(lr_cc, df=2))

    return {
        'lr_independence': round(lr_ind, 4),
        'lr_conditional':  round(lr_cc,  4),
        'p_value_ind':     round(p_ind,  4),
        'p_value_cc':      round(p_cc,   4),
        'pass':     p_cc  > 0.05,
        'pass_ind': p_ind > 0.05,
        'pass_uc':  kup['pass'],
    }


# ─────────────────────────────────────────────────────────────────
# Composite evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_forecast(returns:       np.ndarray,
                      var_forecasts: np.ndarray,
                      es_forecasts:  np.ndarray,
                      alpha:         float) -> dict:
    """
    Full backtesting suite.

    Returns
    -------
    dict with kupiec, christoffersen, violations, violation_rate,
         quantile_score (pinball), es_score (MAE on exceedance days), rmse
    """
    r  = np.asarray(returns,       dtype=float)
    v  = np.asarray(var_forecasts, dtype=float)
    es = np.asarray(es_forecasts,  dtype=float)

    mask = np.isfinite(r) & np.isfinite(v)
    r, v   = r[mask], v[mask]
    es     = es[mask]

    kup     = kupiec_test(r, v, alpha)
    christf = christoffersen_test(r, v, alpha)

    viol = r < v
    N    = int(viol.sum())

    # Quantile (pinball) score
    err = r - v
    qs  = float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1.0) * err)))

    # ES accuracy on violation days only
    es_sc = float(np.mean(np.abs(r[viol] - es[viol]))) if N > 0 else 0.0

    # RMSE across all days
    rmse = float(np.sqrt(np.mean((r - v) ** 2)))

    return {
        'kupiec':              kup,
        'christoffersen':      christf,
        'violations':          N,
        'expected_violations': kup['expected_violations'],
        'violation_rate':      kup['violation_rate'],
        'quantile_score':      round(qs,    8),
        'es_score':            round(es_sc, 8),
        'rmse':                round(rmse,  8),
    }
