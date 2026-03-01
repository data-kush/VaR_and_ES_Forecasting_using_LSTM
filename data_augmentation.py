"""
data_augmentation.py – Synthetic data generation for financial return series.

Two strategies (as a quantitative analyst would apply):

  1. GARCH Simulation
     Fit GARCH(1,1) with skewed-t innovations to the training returns.
     Then simulate N statistically-equivalent paths of the same length.
     Preserves: volatility clustering, fat tails, negative skewness.

  2. Stationary Block Bootstrap  (Politis & Romano 1994)
     Resample variable-length overlapping blocks from the training data.
     Preserves: autocorrelation structure and local distributional properties.

Both methods generate return series that look like real financial data,
effectively multiplying the training dataset without look-ahead bias.
"""

import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────
# GARCH-based simulation
# ─────────────────────────────────────────────────────────────────

def garch_simulate(returns_train: np.ndarray,
                   n_paths: int = 8,
                   seed: int = 42) -> list:
    """
    Fit GARCH(1,1)-skewed-t to training returns and simulate n_paths
    synthetic return series of the same length.

    Parameters
    ----------
    returns_train : 1-D array of training log-returns
    n_paths       : number of synthetic paths to generate
    seed          : random seed for reproducibility

    Returns
    -------
    list of np.ndarray  (each has the same length as returns_train)
    """
    rng = np.random.default_rng(seed)
    n   = len(returns_train)

    # Scale to % for arch stability
    r_pct = returns_train * 100.0

    try:
        am  = arch_model(r_pct, vol='Garch', p=1, q=1, dist='skewt')
        res = am.fit(disp='off', show_warning=False)

        mu    = float(res.params.get('mu',      0.0))
        omega = float(res.params.get('omega',   max(1e-6, np.var(r_pct) * 0.05)))
        a1    = float(res.params.get('alpha[1]', 0.10))
        b1    = float(res.params.get('beta[1]',  0.85))
        nu    = float(np.clip(res.params.get('nu', 8.0), 4.0, 30.0))
        lam   = float(np.clip(res.params.get('lambda', -0.1), -0.5, 0.5))

        # Clamp GARCH parameters to valid region
        a1 = np.clip(a1, 0.01, 0.30)
        b1 = np.clip(b1, 0.50, 0.97)
        if a1 + b1 >= 1.0:
            b1 = 0.97 - a1

        sigma2_init = float(res.conditional_volatility[-1]) ** 2

    except Exception:
        # Fallback: simple GARCH(1,1) with normal innovations
        mu, omega, a1, b1 = 0.0, np.var(r_pct) * 0.05, 0.10, 0.85
        nu, lam = 8.0, -0.1
        sigma2_init = np.var(r_pct)

    from scipy import stats as sp_stats

    paths = []
    for i in range(n_paths):
        sigma2 = sigma2_init
        simulated = np.empty(n, dtype=float)

        for t in range(n):
            sigma  = np.sqrt(max(sigma2, 1e-8))
            # Draw innovation from skewed-t
            z      = float(sp_stats.t.rvs(df=nu, random_state=rng)) * (1 + lam * np.sign(rng.standard_normal()))
            r_t    = (mu + sigma * z) / 100.0  # back to decimal
            simulated[t] = r_t
            sigma2 = omega + a1 * (sigma * z) ** 2 + b1 * sigma2

        paths.append(simulated.astype(np.float32))

    return paths


# ─────────────────────────────────────────────────────────────────
# Stationary block bootstrap
# ─────────────────────────────────────────────────────────────────

def block_bootstrap(returns_train: np.ndarray,
                    n_series: int = 8,
                    mean_block_size: int = 25,
                    seed: int = 123) -> list:
    """
    Stationary Block Bootstrap (Politis & Romano, 1994).

    Block lengths are geometrically distributed with mean = mean_block_size,
    which makes the resulting series approximately stationary.
    This preserves local autocorrelation structure (volatility clustering).

    Parameters
    ----------
    returns_train   : 1-D training return array
    n_series        : number of synthetic series to generate
    mean_block_size : average block length (geometric distribution)
    seed            : random seed

    Returns
    -------
    list of np.ndarray  (each has the same length as returns_train)
    """
    rng = np.random.default_rng(seed)
    n   = len(returns_train)
    p   = 1.0 / mean_block_size   # geometric distribution parameter

    series_list = []
    for _ in range(n_series):
        resampled = np.empty(n, dtype=np.float32)
        idx = 0
        while idx < n:
            # Random starting position
            start      = int(rng.integers(0, n))
            block_len  = int(np.clip(rng.geometric(p), 1, min(mean_block_size * 3, n)))
            for j in range(block_len):
                if idx >= n:
                    break
                resampled[idx] = returns_train[(start + j) % n]
                idx += 1
        series_list.append(resampled)

    return series_list


# ─────────────────────────────────────────────────────────────────
# Unified augmentation pipeline
# ─────────────────────────────────────────────────────────────────

def augment_training_returns(returns_train: np.ndarray,
                             n_garch: int = 6,
                             n_bootstrap: int = 6,
                             seed: int = 0) -> list:
    """
    Generate augmented training return series using both GARCH simulation
    and block bootstrap. Returns a combined list of synthetic series.

    Parameters
    ----------
    returns_train : 1-D array of ORIGINAL training log-returns
    n_garch       : number of GARCH-simulated paths
    n_bootstrap   : number of block-bootstrapped paths
    seed          : base random seed

    Returns
    -------
    list of np.ndarray  (length = n_garch + n_bootstrap)
    """
    aug = []

    if n_garch > 0:
        try:
            aug += garch_simulate(returns_train, n_paths=n_garch, seed=seed)
        except Exception as e:
            print(f"  [WARN] GARCH simulation failed ({e}); skipping.")

    if n_bootstrap > 0:
        try:
            aug += block_bootstrap(returns_train, n_series=n_bootstrap,
                                   seed=seed + 999)
        except Exception as e:
            print(f"  [WARN] Block bootstrap failed ({e}); skipping.")

    return aug
