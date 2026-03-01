"""
train.py – Main training script for LSTM-VaR/ES project.

Usage (VS Code terminal):
    python train.py                    # Train all datasets at α=5%
    python train.py --alpha 0.01       # Train at α=1%
    python train.py --alpha all        # Train at α=1%, 2.5%, 5%, 10%
    python train.py --asset SP500      # Train one specific dataset
    python train.py --skip-aug         # Disable data augmentation (faster)

Results are written to  results/<ASSET>.json
Model checkpoints are  NOT saved (re-train to reproduce).

System requirements: i5 CPU, 8 GB RAM — no GPU needed.
Expected runtime:     ~5–8 min per (asset × alpha) on a modern i5.
"""

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── Project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from lstm_model       import train_stateful_lstm
from traditional_models import run_all_traditional
from data_augmentation  import augment_training_returns

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent.parent          # CSVs live one level up
RESULTS_DIR  = Path(__file__).parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

ASSETS       = ['DJIA', 'FTSE', 'Gold', 'Oil', 'SP500', 'USDJPY']
ALPHA_DEFAULT = [0.05]
ALPHA_ALL     = [0.01, 0.025, 0.05, 0.10]

# LSTM hyper-parameters (CPU-optimised)
LSTM_CONFIG = dict(
    hidden_size  = 32,
    dropout      = 0.15,
    epochs       = 60,
    lr           = 3e-4,
    patience     = 10,
    train_ratio  = 0.70,
    val_ratio    = 0.15,
    tbptt_len    = 64,
)

# Data augmentation config
AUG_CONFIG = dict(
    n_garch     = 6,
    n_bootstrap = 6,
)


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_asset(asset_name: str) -> pd.DataFrame:
    """Load and standardise a single asset CSV."""
    path = DATA_DIR / f'{asset_name}.csv'
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    col_lower  = {c.lower(): c for c in df.columns}

    # Rename to standard columns
    for std, variants in [
        ('Date',    ['date']),
        ('price',   ['price', 'close', 'adj close']),
        ('returns', ['returns', 'return']),
    ]:
        for v in variants:
            if v in col_lower and col_lower[v] != std:
                df = df.rename(columns={col_lower[v]: std})

    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        df['Date'] = df['Date'].dt.tz_localize(None)
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    # Compute log returns if missing
    if 'returns' not in df.columns:
        if 'price' in df.columns:
            df['price']   = pd.to_numeric(df['price'], errors='coerce')
            df['returns'] = np.log(df['price'] / df['price'].shift(1))
        else:
            raise ValueError(f"{asset_name}: no returns or price column")

    df = df.dropna(subset=['returns']).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# Per-asset training
# ─────────────────────────────────────────────────────────────────

def train_asset(asset_name: str,
                alpha_levels: list,
                use_aug:      bool = True) -> dict:
    """
    Full training pipeline for one asset:
    1. Load data
    2. Augment training returns
    3. Train LSTM at each alpha level
    4. Train all traditional models at each alpha level
    5. Return result dict
    """
    print(f"\n{'='*60}")
    print(f"  ASSET: {asset_name}")
    print(f"{'='*60}")

    df = load_asset(asset_name)
    print(f"  Rows: {len(df)}  |  "
          f"Period: {df['Date'].iloc[0].strftime('%Y-%m-%d') if 'Date' in df.columns else '?'}"
          f" → {df['Date'].iloc[-1].strftime('%Y-%m-%d') if 'Date' in df.columns else '?'}")

    r_all   = df['returns'].dropna().values.astype(np.float32)
    T       = len(r_all)
    n_train = int(T * 0.70)
    r_train = r_all[:n_train]

    # ── Data augmentation ────────────────────────────────────────
    aug_series = []
    if use_aug:
        print(f"\n  [1/3] Augmenting training data…", end=' ', flush=True)
        t0 = time.time()
        aug_series = augment_training_returns(
            r_train,
            n_garch     = AUG_CONFIG['n_garch'],
            n_bootstrap = AUG_CONFIG['n_bootstrap'],
        )
        print(f"{len(aug_series)} synthetic series generated "
              f"({time.time() - t0:.1f}s)")
    else:
        print(f"\n  [1/3] Augmentation skipped.")

    asset_result = {
        'asset':          asset_name,
        'generated_at':   datetime.now().isoformat(),
        'total_rows':     T,
        'train_size':     n_train,
        'val_size':       int(T * 0.15),
        'test_size':      T - n_train - int(T * 0.15),
        'augmentation':   {
            'enabled':       use_aug,
            'n_synthetic':   len(aug_series),
            'method':        'garch_simulation + block_bootstrap',
        },
        'results':        {},   # keyed by alpha string
    }

    # ── Train at each alpha level ─────────────────────────────────
    for alpha in alpha_levels:
        alpha_key = str(alpha)
        print(f"\n  ── Alpha = {alpha*100:.1f}% ────────────────────────")
        alpha_result = {}

        # ── LSTM ─────────────────────────────────────────────────
        print(f"  [2/3] Training Stateful LSTM…", end=' ', flush=True)
        t0 = time.time()
        try:
            lstm_res = train_stateful_lstm(
                df,
                alpha       = alpha,
                aug_series  = aug_series,
                **LSTM_CONFIG,
            )
            elapsed = time.time() - t0
            kup = lstm_res['metrics']['kupiec']
            cc  = lstm_res['metrics']['christoffersen']
            print(f"done ({elapsed:.0f}s)  |  "
                  f"epochs={lstm_res['epochs_run']}  |  "
                  f"violations={kup['violations']}/{kup['expected_violations']}  |  "
                  f"Kupiec={'PASS' if kup['pass'] else 'FAIL'}  |  "
                  f"CC={'PASS' if cc['pass'] else 'FAIL'}")
            alpha_result['lstm'] = _flatten_result(lstm_res)
        except Exception as e:
            print(f"FAILED: {e}")
            alpha_result['lstm'] = None

        # ── Traditional models ────────────────────────────────────
        print(f"  [3/3] Traditional models…")
        try:
            trad_results = run_all_traditional(
                df, alpha=alpha,
                train_ratio=0.70, val_ratio=0.15, verbose=True
            )
            for m_name, m_res in trad_results.items():
                if m_res is not None:
                    alpha_result[m_name] = _flatten_result(m_res)
                else:
                    alpha_result[m_name] = None
        except Exception as e:
            print(f"  Traditional models ERROR: {e}")

        asset_result['results'][alpha_key] = alpha_result

    return asset_result


# ─────────────────────────────────────────────────────────────────
# Result flattening (merge nested metrics into top-level keys)
# ─────────────────────────────────────────────────────────────────

def _flatten_result(res: dict) -> dict:
    """
    Flatten nested metrics so the dashboard can read them directly.
    Converts numpy arrays to Python lists for JSON serialisation.
    """
    if res is None:
        return None

    def _to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return x

    flat = {k: _to_list(v) for k, v in res.items() if k != 'metrics'}

    m = res.get('metrics', {})
    flat['kupiec']          = m.get('kupiec', {})
    flat['christoffersen']  = m.get('christoffersen', {})
    flat['violations']      = m.get('violations', 0)
    flat['expected_violations'] = m.get('expected_violations', 0)
    flat['violation_rate']  = m.get('violation_rate', 0.0)
    flat['quantile_score']  = m.get('quantile_score', 0.0)
    flat['es_score']        = m.get('es_score', 0.0)
    flat['rmse']            = m.get('rmse', 0.0)

    return flat


# ─────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────

def print_summary(all_results: dict, alpha_levels: list):
    print(f"\n\n{'='*80}")
    print(f"  TRAINING COMPLETE — SUMMARY")
    print(f"{'='*80}")

    for alpha in alpha_levels:
        alpha_key = str(alpha)
        print(f"\n  α = {alpha*100:.1f}%")
        header = f"  {'Asset':<8} {'Model':<8} {'Viol':>6} {'Exp':>5} "
        header += f"{'Rate':>7} {'Kupiec':>8} {'CC':>8} {'QScore':>10}"
        print(header)
        print(f"  {'-'*72}")

        for asset, asset_res in all_results.items():
            results_at_alpha = asset_res.get('results', {}).get(alpha_key, {})
            for model_name in ['lstm', 'garch', 'gas', 'hs', 'fhs']:
                m = results_at_alpha.get(model_name)
                if m is None:
                    continue
                kup = m.get('kupiec', {})
                cc  = m.get('christoffersen', {})
                row = (f"  {asset:<8} {model_name.upper():<8} "
                       f"{kup.get('violations',0):>6} "
                       f"{kup.get('expected_violations',0):>5} "
                       f"{kup.get('violation_rate',0)*100:>6.2f}% "
                       f"{'PASS' if kup.get('pass') else 'FAIL':>8} "
                       f"{'PASS' if cc.get('pass') else 'FAIL':>8} "
                       f"{m.get('quantile_score',0):>10.6f}")
                print(row)


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM-VaR/ES and benchmark models on 6 financial datasets.')
    parser.add_argument('--alpha', default='0.05',
                        help='Coverage level: 0.01 | 0.025 | 0.05 | 0.10 | all  (default: 0.05)')
    parser.add_argument('--asset', default='all',
                        help='Dataset to train: DJIA | FTSE | Gold | Oil | SP500 | USDJPY | all')
    parser.add_argument('--skip-aug', action='store_true',
                        help='Disable data augmentation (faster, lower accuracy)')
    parser.add_argument('--force', action='store_true',
                        help='Re-train even if result file already exists')
    args = parser.parse_args()

    # Parse alpha
    if args.alpha == 'all':
        alpha_levels = ALPHA_ALL
    else:
        try:
            alpha_levels = [float(args.alpha)]
        except ValueError:
            print(f"Invalid alpha: {args.alpha}")
            sys.exit(1)

    # Parse asset list
    if args.asset == 'all':
        assets = ASSETS
    elif args.asset in ASSETS:
        assets = [args.asset]
    else:
        print(f"Unknown asset '{args.asset}'. Choose from: {ASSETS}")
        sys.exit(1)

    use_aug = not args.skip_aug

    print(f"\nLSTM-VaR/ES Training Pipeline")
    print(f"  Assets       : {assets}")
    print(f"  Alpha levels : {alpha_levels}")
    print(f"  Augmentation : {'enabled' if use_aug else 'disabled'}")
    print(f"  Results dir  : {RESULTS_DIR}")
    print(f"  Started at   : {datetime.now().strftime('%H:%M:%S')}")

    global_start = time.time()
    all_results  = {}

    for asset in assets:
        out_path = RESULTS_DIR / f'{asset}.json'

        if out_path.exists() and not args.force:
            print(f"\n  Skipping {asset} — result file already exists.")
            print(f"  (Use --force to re-train.)")
            with open(out_path) as f:
                all_results[asset] = json.load(f)
            continue

        try:
            asset_result = train_asset(asset, alpha_levels, use_aug=use_aug)
            all_results[asset] = asset_result

            # Save immediately after each asset (crash-safe)
            with open(out_path, 'w') as f:
                json.dump(asset_result, f, indent=2, default=_json_safe)
            print(f"\n  ✓ Results saved → {out_path}")

        except Exception as e:
            print(f"\n  ✗ {asset} FAILED: {e}")
            import traceback; traceback.print_exc()

    # Print summary table
    if all_results:
        print_summary(all_results, alpha_levels)

    elapsed = time.time() - global_start
    print(f"\n\nTotal training time: {elapsed/60:.1f} minutes")
    print(f"Dashboard: run  python app.py  then open http://localhost:5000")


def _json_safe(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


if __name__ == '__main__':
    main()
