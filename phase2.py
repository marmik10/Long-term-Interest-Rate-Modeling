"""PROJECT 2 — PHASE 2: Nelson–Siegel and Svensson term-structure calibration.

Run from VS Code / terminal (Python 3.9+):

    python3 -m pip install numpy pandas scipy matplotlib
    python3 project2_phase2_ns_svensson.py

Place this script beside the ``project2_phase1_outputs`` folder created in Phase 1.
By default it reads the 201 completed monthly 1Y–30Y Treasury ZERO yields.
You can supply an alternate path with --input.

Research design:
* Fit Nelson–Siegel (NS) and Nelson–Siegel–Svensson (NSS) to 30 fitted
  continuously compounded GSW zero yields per date, measured in percent.
* Compare in-sample curve reconstruction at all 30 maturities.
* Fit on odd maturities and evaluate even maturities (cross-maturity holdout).
  This is not a chronological out-of-sample interest-rate forecast.
* Estimate cross-sectional curves date-by-date independently, with no lookahead
  in any individual curve's fit. Time-series models begin in Phase 3.
* Keep model-fit errors distinct from validation on actual market prices:
  the Fed's GSW zero yields are themselves computed using a Svensson curve.
  An NSS near-perfect reconstruction is therefore NOT independent evidence
  of superior bond-pricing or yield-forecasting performance.

Outputs written to ``project2_phase2_outputs`` next to this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parent
MATURITIES = np.arange(1, 31, dtype=float)
COLUMNS = [f'{i}Y' for i in range(1, 31)]
ODD = MATURITIES.astype(int) % 2 == 1
EVEN = ~ODD
EPS = 1e-12


def ns_loadings(t: np.ndarray, lam: float) -> np.ndarray:
    """Exponential-decay loadings, with t measured in YEARS and lambda 1/YEAR."""
    t = np.asarray(t, dtype=float)
    x = lam * t
    slope = -np.expm1(-x) / x  # stable (1-exp(-x))/x
    curvature = slope - np.exp(-x)
    return np.column_stack((np.ones_like(t), slope, curvature))


def nss_loadings(t: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    out = ns_loadings(t, lam1)
    x2 = lam2 * np.asarray(t, dtype=float)
    curvature2 = -np.expm1(-x2) / x2 - np.exp(-x2)
    return np.column_stack((out, curvature2))


def solve_linear(t: np.ndarray, y: np.ndarray, lambdas: tuple[float, ...], model: str):
    X = ns_loadings(t, lambdas[0]) if model == 'NS' else nss_loadings(t, *lambdas)
    # Reject severe singularity instead of reporting giant, meaningless beta estimates.
    if np.linalg.cond(X) > 1e8:
        return float('inf'), np.full(X.shape[1], np.nan)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    sse = float(np.dot(pred - y, pred - y))
    return sse, beta


def fit_ns(t: np.ndarray, y: np.ndarray):
    bounds = [(0.03, 0.22), (0.22, 0.85), (0.85, 2.5)]
    solutions = []
    for lo, hi in bounds:
        res = minimize_scalar(
            lambda lam: solve_linear(t, y, (lam,), 'NS')[0],
            bounds=(lo, hi), method='bounded',
            options={'xatol': 1e-6, 'maxiter': 120}
        )
        solutions.append((float(res.fun), float(res.x)))
    best_sse, lam = min(solutions)
    _, beta = solve_linear(t, y, (lam,), 'NS')
    if not np.isfinite(best_sse):
        raise ArithmeticError('NS optimization found no valid parameter set')
    return beta, (lam,)


def fit_nss(t: np.ndarray, y: np.ndarray, ns_lam: float):
    """Full two-decay nonlinear optimization using variable projection for beta."""
    lo, hi = np.log(0.025), np.log(3.0)
    starts = [(ns_lam, 0.055), (ns_lam, 0.3), (ns_lam, 1.2), (0.12, 0.75)]

    def objective(z):
        a, b = np.exp(z)
        sse, beta = solve_linear(t, y, (a, b), 'NSS')
        if not np.isfinite(sse):
            return 1e12
        # Small regularizer suppresses unstable betas when the two humps
        # approximate one another; calibrated predictions remain the priority.
        return sse + 1e-9 * float(np.dot(beta, beta))

    solutions = []
    for a, b in starts:
        initial = np.log(np.clip([a, b], np.exp(lo), np.exp(hi)))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in subtract', category=RuntimeWarning)
            res = minimize(objective, initial, method='L-BFGS-B',
                           bounds=[(lo, hi), (lo, hi)],
                           options={'maxiter': 250, 'ftol': 1e-12, 'maxls': 30})
        aa, bb = np.exp(res.x)
        sse, beta = solve_linear(t, y, (float(aa), float(bb)), 'NSS')
        if np.isfinite(sse):
            solutions.append((sse, beta, (float(aa), float(bb))))
    if not solutions:
        raise ArithmeticError('NSS optimization found no well-conditioned parameter set')
    _, beta, lambdas = min(solutions, key=lambda item: item[0])
    return beta, lambdas


def predict(t: np.ndarray, beta: np.ndarray, lambdas: tuple[float, ...], model: str) -> np.ndarray:
    X = ns_loadings(t, lambdas[0]) if model == 'NS' else nss_loadings(t, *lambdas)
    return X @ beta


def rmse_bps(actual: np.ndarray, fitted: np.ndarray) -> float:
    return float(100 * np.sqrt(np.mean((actual - fitted) ** 2)))


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing Phase 1 data: {path}\n'
            'Put this script beside project2_phase1_outputs, or pass '\
            '--input /absolute/path/to/monthly_zero_yields_pct.csv'
        )
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    if list(df.columns) != COLUMNS:
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f'Phase 1 file missing 1Y–30Y columns: {sorted(missing)}')
        df = df[COLUMNS]
    df = df.apply(pd.to_numeric, errors='raise').sort_index()
    if df.index.has_duplicates or df.isna().any().any():
        raise ValueError('Duplicate monthly dates or missing yields: validate Phase 1 data')
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite yields in Phase 1 CSV')
    if len(df) < 12:
        raise ValueError('At least 12 observations required')
    return df


def plot_all(data: pd.DataFrame, metric: pd.DataFrame, latest: pd.DataFrame,
             ns_params: pd.DataFrame, out: Path):
    plt.figure(figsize=(11, 5))
    t = latest['maturity_years'].to_numpy()
    for key, legend, fmt in [
        ('observed_zero_yield_pct', 'GSW fitted zero yields', 'o'),
        ('ns_all_pct', 'Nelson–Siegel', '-'),
        ('nss_all_pct', 'Svensson', '--')]:
        plt.plot(t, latest[key], fmt, label=legend, linewidth=2, markersize=4)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Continuously compounded zero yield (%)')
    plt.title(f'Curve calibration — {data.index[-1].date()}')
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(out/'01_latest_curve_fits.png', dpi=170); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].plot(t, 100*(latest['ns_all_pct'] - latest['observed_zero_yield_pct']), label='NS')
    axes[0].plot(t, 100*(latest['nss_all_pct'] - latest['observed_zero_yield_pct']), label='NSS')
    axes[0].axhline(0, color='gray', linewidth=1)
    axes[0].set(title='All-maturity fitted residuals', xlabel='Maturity (years)', ylabel='Model − Fed yield (bps)')
    axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].plot(t, 100*(latest['ns_odd_fit_pct'] - latest['observed_zero_yield_pct']), label='NS')
    axes[1].plot(t, 100*(latest['nss_odd_fit_pct'] - latest['observed_zero_yield_pct']), label='NSS')
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set(title='Fit using only odd-year maturities', xlabel='Maturity (years)', ylabel='Model − Fed yield (bps)')
    axes[1].legend(); axes[1].grid(alpha=0.2)
    plt.tight_layout(); plt.savefig(out/'02_latest_curve_residuals.png', dpi=170); plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for model in ('ns', 'nss'):
        axes[0].plot(metric.index, metric[f'{model}_full_rmse_bps'], label=model.upper())
        axes[1].plot(metric.index, metric[f'{model}_holdout_even_rmse_bps'], label=model.upper())
    axes[0].set(ylabel='RMSE (bps)', title='All-maturity in-sample reconstruction')
    axes[1].set(ylabel='RMSE (bps)', title='Odd-year fit → even-year maturity holdout')
    for ax in axes:
        ax.grid(alpha=0.2); ax.legend()
    plt.tight_layout(); plt.savefig(out/'03_fit_errors_over_time.png', dpi=170); plt.close()

    plt.figure(figsize=(11, 5))
    for key, label in [('beta0_pct','Level (beta0)'), ('beta1_pct','Slope coefficient (beta1)'),
                       ('beta2_pct','Curvature coefficient (beta2)')]:
        plt.plot(ns_params.index, ns_params[key], label=label)
    plt.title('Nelson–Siegel cross-sectional parameters (NOT forecasts)')
    plt.ylabel('Yield coefficient (percentage points)')
    plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(out/'04_ns_parameter_history.png', dpi=170); plt.close()


def run(args):
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(inp)
    if args.max_months:
        data = data.tail(args.max_months)

    metrics, params_ns, params_nss = [], [], []
    latest = None
    for k, (date, row) in enumerate(data.iterrows(), 1):
        y = row.to_numpy(dtype=float)
        beta_ns, lambda_ns = fit_ns(MATURITIES, y)
        beta_nss, lambda_nss = fit_nss(MATURITIES, y, lambda_ns[0])
        ns_all = predict(MATURITIES, beta_ns, lambda_ns, 'NS')
        nss_all = predict(MATURITIES, beta_nss, lambda_nss, 'NSS')

        beta_ns_o, lambda_ns_o = fit_ns(MATURITIES[ODD], y[ODD])
        beta_nss_o, lambda_nss_o = fit_nss(MATURITIES[ODD], y[ODD], lambda_ns_o[0])
        ns_odd_predict = predict(MATURITIES, beta_ns_o, lambda_ns_o, 'NS')
        nss_odd_predict = predict(MATURITIES, beta_nss_o, lambda_nss_o, 'NSS')

        metrics.append({
            'Date': date,
            'ns_full_rmse_bps': rmse_bps(y, ns_all),
            'nss_full_rmse_bps': rmse_bps(y, nss_all),
            'ns_train_odd_rmse_bps': rmse_bps(y[ODD], ns_odd_predict[ODD]),
            'nss_train_odd_rmse_bps': rmse_bps(y[ODD], nss_odd_predict[ODD]),
            'ns_holdout_even_rmse_bps': rmse_bps(y[EVEN], ns_odd_predict[EVEN]),
            'nss_holdout_even_rmse_bps': rmse_bps(y[EVEN], nss_odd_predict[EVEN]),
        })
        params_ns.append(dict(Date=date, beta0_pct=beta_ns[0], beta1_pct=beta_ns[1],
                              beta2_pct=beta_ns[2], lambda1_per_year=lambda_ns[0]))
        params_nss.append(dict(Date=date, beta0_pct=beta_nss[0], beta1_pct=beta_nss[1],
                               beta2_pct=beta_nss[2], beta3_pct=beta_nss[3],
                               lambda1_per_year=lambda_nss[0], lambda2_per_year=lambda_nss[1]))
        if k == len(data):
            latest = pd.DataFrame({
                'maturity_years': MATURITIES,
                'observed_zero_yield_pct': y,
                'ns_all_pct': ns_all,
                'nss_all_pct': nss_all,
                'ns_odd_fit_pct': ns_odd_predict,
                'nss_odd_fit_pct': nss_odd_predict,
                'is_holdout_even_maturity': EVEN,
            })
        if k % 25 == 0 or k == len(data):
            print(f'Calibrated {k}/{len(data)} monthly curves')

    assert latest is not None
    metrics_df = pd.DataFrame(metrics).set_index('Date')
    ns_df = pd.DataFrame(params_ns).set_index('Date')
    nss_df = pd.DataFrame(params_nss).set_index('Date')
    metrics_df.to_csv(out/'phase2_fit_metrics_monthly.csv', float_format='%.8f')
    ns_df.to_csv(out/'phase2_ns_parameters_monthly.csv', float_format='%.9f')
    nss_df.to_csv(out/'phase2_nss_parameters_monthly.csv', float_format='%.9f')
    latest.to_csv(out/'phase2_latest_curve_comparison.csv', index=False, float_format='%.9f')
    plot_all(data, metrics_df, latest, ns_df, out)

    summary = []
    summary.append(f'SOURCE: {inp}')
    summary.append(f'Sample: {data.index.min().date()} to {data.index.max().date()} ({len(data)} monthly curves)')
    summary.append('Units: Treasury zero yields/NS betas/NSS betas in PERCENT, RMSE in basis points')
    summary.append('Maturities: 1Y to 30Y, continuously compounded GSW zero yields')
    summary.append('Odd-year maturities (15) used to calibrate holdout test; even-year maturities (15) held out')
    summary.append('')
    for model in ('ns', 'nss'):
        label = 'Nelson–Siegel' if model == 'ns' else 'Svensson'
        for metric, tag in [('full_rmse_bps','all-maturity fit'),
                            ('holdout_even_rmse_bps','even-year maturity holdout')]:
            s = metrics_df[f'{model}_{metric}']
            summary.append(f'{label} {tag}: mean={s.mean():.4f} bp, median={s.median():.4f} bp, 95th-pct={s.quantile(0.95):.4f} bp')
    summary += [
        '',
        'METHODOLOGY / LIMITATIONS:',
        '1. The GSW published yield curve is itself estimated with the Svensson family.',
        '   Small NSS fitting RMSE is largely reconstruction, not independent market validation.',
        '2. Even-year maturity holdout tests interpolation/extrapolation within the SAME',
        '   date and source; it is NOT out-of-time forecasting or a pricing backtest.',
        '3. NSS beta and decay parameters may not be statistically identifiable even when',
        '   the yield curve fits closely. Interpret NSS factors cautiously.',
        '4. Optimizations are conducted independently for each date; parameter jumps can',
        '   reflect nonidentifiability. Dynamic model fitting is a separate Phase 3 task.',
        '5. Spot yields here are continuously compounded PERCENT values; do not mix',
        '   these with par yields, swap rates, or percent-to-decimal conversions.',
        '6. Strict forecast validation needs an expanding/rolling training window',
        '   with prediction-time information only (Phase 3).',
        '7. Treasury fitted zero yields are not SOFR discount curves and alone cannot',
        '   establish derivatives-implied G2++ or HJM pricing-volatility parameters.',
    ]
    (out/'phase2_methodology_and_results.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8')
    print('\n'+'='*72+'\nPHASE 2 — NELSON–SIEGEL / SVENSSON RESULTS\n'+'='*72)
    for line in summary[:12]:
        print(line)
    print(f'\nSaved 4 CSVs, 4 figures, and methodology report to:\n{out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=str(ROOT/'project2_phase1_outputs'/'monthly_zero_yields_pct.csv'),
                        help='Phase 1 monthly zero-yield CSV (columns 1Y ... 30Y)')
    parser.add_argument('--output', default=str(ROOT/'project2_phase2_outputs'))
    parser.add_argument('--max-months', type=int, default=0, help='Optional: only calibrate the most recent N months for a quick test')
    args = parser.parse_args()
    if args.max_months < 0:
        parser.error('--max-months must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
