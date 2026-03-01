"""
app.py – Flask professional risk-assessment dashboard.

Routes
------
GET  /                          → main dashboard (no file-upload)
GET  /api/assets                → list of trained assets + available alphas
GET  /api/results/<asset>       → full result dict for one asset
GET  /api/summary               → condensed pass/fail table (all assets)
GET  /api/model_info            → static model architecture description

Run:  python app.py
Then open  http://localhost:5000
"""

from flask import Flask, jsonify, render_template, abort
from pathlib import Path
import json
import numpy as np

app = Flask(__name__)

RESULTS_DIR = Path(__file__).parent / 'results'
ASSETS      = ['DJIA', 'FTSE', 'Gold', 'Oil', 'SP500', 'USDJPY']
ASSET_INFO  = {
    'DJIA':   {'name': 'Dow Jones Industrial Average', 'class': 'Equity Index',   'currency': 'USD'},
    'FTSE':   {'name': 'FTSE 100 Index',               'class': 'Equity Index',   'currency': 'GBP'},
    'Gold':   {'name': 'Gold Spot Price',               'class': 'Commodity',      'currency': 'USD'},
    'Oil':    {'name': 'Crude Oil (WTI) Spot',          'class': 'Commodity',      'currency': 'USD'},
    'SP500':  {'name': 'S&P 500 Index',                 'class': 'Equity Index',   'currency': 'USD'},
    'USDJPY': {'name': 'USD/JPY Exchange Rate',         'class': 'FX',             'currency': 'JPY'},
}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _load_result(asset: str) -> dict | None:
    p = RESULTS_DIR / f'{asset}.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _safe_float(x):
    """Convert value to JSON-safe float."""
    if x is None:
        return None
    try:
        v = float(x)
        return None if (v != v) else v   # NaN check
    except (TypeError, ValueError):
        return None


def _clean_dict(d):
    """Recursively make a dict JSON-safe (replace NaN/Inf with None)."""
    if isinstance(d, dict):
        return {k: _clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean_dict(v) for v in d]
    if isinstance(d, float):
        return None if (d != d or abs(d) == float('inf')) else d
    if isinstance(d, (np.floating, np.integer)):
        v = float(d)
        return None if (v != v) else v
    return d


# ─────────────────────────────────────────────────────────────────
# Routes — pages
# ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ─────────────────────────────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────────────────────────────

@app.route('/api/assets')
def api_assets():
    """Return list of assets that have result files, with available alpha levels."""
    available = []
    for asset in ASSETS:
        res = _load_result(asset)
        if res is not None:
            alphas = list(res.get('results', {}).keys())
            info   = ASSET_INFO.get(asset, {})
            available.append({
                'id':           asset,
                'name':         info.get('name', asset),
                'asset_class':  info.get('class', ''),
                'currency':     info.get('currency', ''),
                'alpha_levels': sorted(alphas, key=float),
                'total_rows':   res.get('total_rows', 0),
                'test_size':    res.get('test_size', 0),
                'train_size':   res.get('train_size', 0),
            })
    return jsonify({'assets': available, 'count': len(available)})


@app.route('/api/results/<asset>')
def api_results(asset: str):
    """Return full result dict for one asset."""
    if asset not in ASSETS:
        abort(404, description=f"Unknown asset: {asset}")
    res = _load_result(asset)
    if res is None:
        abort(404, description=f"No results found for {asset}. Run train.py first.")
    return jsonify(_clean_dict(res))


@app.route('/api/summary')
def api_summary():
    """
    Condensed summary of all trained assets.
    Returns a flat list of rows for the comparison table.
    """
    rows = []
    for asset in ASSETS:
        res = _load_result(asset)
        if res is None:
            continue
        for alpha_key, alpha_res in res.get('results', {}).items():
            for model in ['lstm', 'garch', 'gas', 'hs', 'fhs']:
                m = alpha_res.get(model)
                if m is None:
                    continue
                kup = m.get('kupiec', {})
                cc  = m.get('christoffersen', {})
                rows.append({
                    'asset':              asset,
                    'alpha':              float(alpha_key),
                    'model':              model.upper(),
                    'violations':         kup.get('violations', 0),
                    'expected_violations':kup.get('expected_violations', 0),
                    'violation_rate':     _safe_float(kup.get('violation_rate', 0)),
                    'kupiec_pass':        bool(kup.get('pass', False)),
                    'kupiec_pval':        _safe_float(kup.get('p_value', 0)),
                    'cc_pass':            bool(cc.get('pass', False)),
                    'cc_pval':            _safe_float(cc.get('p_value_cc', 0)),
                    'ind_pass':           bool(cc.get('pass_ind', False)),
                    'quantile_score':     _safe_float(m.get('quantile_score', 0)),
                    'es_score':           _safe_float(m.get('es_score', 0)),
                    'rmse':               _safe_float(m.get('rmse', 0)),
                })
    return jsonify({'rows': rows})


@app.route('/api/model_info')
def api_model_info():
    """Static description of the LSTM architecture and methodology."""
    return jsonify({
        'model_name': 'Stateful LSTM-VaR/ES',
        'architecture': {
            'type':         'Stateful Single-Layer LSTM',
            'input':        '6 lagged features per time step',
            'hidden_size':  32,
            'output_heads': 2,
            'output':       'VaR and ES (strictly negative)',
            'parameters':   '~10,000',
            'constraint':   'ES ≤ VaR enforced analytically',
        },
        'loss_function': {
            'primary':   'Fissler-Ziegel FZ0 (weight 0.7)',
            'secondary': 'Conservative quantile loss (weight 0.3)',
            'reference': 'Fissler & Ziegel (2016), Patton et al. (2019)',
        },
        'training': {
            'method':       'Truncated BPTT (chunk_size=64)',
            'optimizer':    'Adam (lr=3e-4)',
            'scheduler':    'ReduceLROnPlateau',
            'early_stop':   'patience=10 on val FZ0 loss',
            'shuffle':      False,
            'stateful':     True,
        },
        'data_augmentation': {
            'garch_paths':       6,
            'bootstrap_paths':   6,
            'method':            'GARCH(1,1)-skewed-t simulation + Stationary Block Bootstrap',
        },
        'calibration': {
            'method':    'Alpha-quantile residual shift on validation set',
            'direction': 'Shifts VaR downward if actual violation rate > alpha',
        },
        'improvements_over_baseline': [
            'Stateful LSTM: hidden state persists across entire time series',
            'Correct quantile loss: violation-side multiplier (not non-violation)',
            'FZ0 loss: only consistent joint scoring function for (VaR, ES)',
            'No shuffle: temporal order preserved',
            'No BatchNorm bypass: consistent train/inference normalisation',
        ],
        'benchmark_models': ['GARCH(1,1)-skewed-t', 'GAS FZ1F', 'Historical Simulation', 'Filtered HS'],
        'backtests': ['Kupiec POF (χ²(1))', 'Christoffersen CC (χ²(2))'],
        'reference_papers': [
            'Qiu, Lazar & Nakata (2024) – VaR and ES forecasting via RNN-based stateful models',
            'Fissler & Ziegel (2016) – Higher order elicitability',
            'Patton, Ziegel & Chen (2019) – Dynamic semiparametric models for ES and VaR',
            'Kupiec (1995) – Techniques for verifying the accuracy of risk measurement models',
            'Christoffersen (1998) – Evaluating interval forecasts',
        ],
    })


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\nRisk Assessment Dashboard")
    print("  Open  http://localhost:5000  in your browser\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
