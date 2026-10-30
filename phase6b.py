#!/usr/bin/env python3
"""Project 2 / Phase 6B: Empirical HJM forward-curve volatility validation.

REAL INPUT: Phase 1 Federal Reserve GSW fitted instantaneous forward rates, 1Y..30Y.
Compares one-factor and two-factor exponentially decaying Gaussian HJM forward
shock covariance against historical daily forward-rate changes. Evaluates on a
chronological HOLDOUT with frozen parameters. Also reports historical forward-
change PCA and checks the corresponding HJM Q-measure no-arbitrage drift identity.

CAUTION: Daily changes estimate P-measure volatility *proxies*, not market-implied
Q pricing parameters; this model is NOT calibrated to SOFR swaption quotes.
The 1Y..30Y GSW forward rates are Svensson-fitted estimates, not trade prices.
Daily forward rates are at CONSTANT time-to-maturity; HJM is in fixed-date T.
Small-time-tenor covariance matching neglects roll down/dt effects.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
TENORS = np.arange(1, 31, dtype=float)
DT = 1.0 / 252.0


def source_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Cannot locate {path}; first run Phase 1.')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    dates = pd.to_datetime(df.index, errors='coerce')
    if dates.isna().any():
        raise ValueError('Invalid dates in the forward data.')
    df.index = dates
    cols = [f'{i}Y' for i in range(1, 31)]
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise ValueError(f'Missing GSW forward-rate tenor columns: {absent}')
    out = df.loc[:, cols].apply(pd.to_numeric, errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan).sort_index()
    if out.index.has_duplicates:
        raise ValueError('Duplicate observation dates in input.')
    return out / 100.0  # reported percent -> decimal annualized rates


def sample_changes(data: pd.DataFrame) -> pd.DataFrame:
    # Important: no differencing across unobserved yield curves; missing rows are
    # preserved until AFTER differences are computed.
    d = data.diff().iloc[1:].dropna(how='any')
    if len(d) < 150:
        raise ValueError('Need >= 150 complete daily forward-rate changes.')
    if not np.isfinite(d.to_numpy()).all():
        raise ValueError('Non-finite daily changes remain.')
    if float(np.max(np.abs(d.to_numpy()))) > 0.20:
        raise ValueError('Unreasonably large daily change, inspect input data and units.')
    return d


def annual_cov(x: np.ndarray) -> np.ndarray:
    cov = np.cov(np.asarray(x, float), rowvar=False, ddof=1) / DT
    return 0.5 * (cov + cov.T)


def exponential_vol(tau, a, sigma):
    return sigma * np.exp(-a * np.asarray(tau))


def model_cov1(p: np.ndarray) -> np.ndarray:
    a, sigma = p
    v = exponential_vol(TENORS, a, sigma)
    return np.outer(v, v)


def model_cov2(p: np.ndarray) -> np.ndarray:
    a, b, sigma, eta, rho = p
    v1 = exponential_vol(TENORS, a, sigma)
    v2 = exponential_vol(TENORS, b, eta)
    return (np.outer(v1, v1) + np.outer(v2, v2)
            + rho * (np.outer(v1, v2) + np.outer(v2, v1)))


def relative_frobenius(model: np.ndarray, empirical: np.ndarray) -> float:
    return float(100 * np.linalg.norm(model - empirical, ord='fro') /
                 max(np.linalg.norm(empirical, ord='fro'), 1e-20))


def fit_cov(cov: np.ndarray, factors: int):
    triu = np.triu_indices(30)
    scale = max(np.linalg.norm(cov[triu]), 1e-15)
    if factors == 1:
        starts = [[.01, .02], [.08, .02], [.5, .05], [1.0, .1]]
        lower = [.001, .00001]
        upper = [3.0, .30]
        predict = model_cov1
    elif factors == 2:
        starts = [
            [.016, .78, .012, .019, -.90],
            [.08, .65, .011, .016, -.55],
            [.04, .30, .025, .025, -.3],
            [.20, 1.0, .025, .04, .5],
            [.45, .08, .02, .02, -.6],
            [.02, 1.6, .03, .05, .0],
        ]
        lower = [.001, .001, .00001, .00001, -.99]
        upper = [3.0, 4.0, .30, .30, .99]
        predict = model_cov2
    else:
        raise ValueError('factors must be 1 or 2')
    best = None
    for seed in starts:
        result = least_squares(lambda p: (predict(p)[triu] - cov[triu]) / scale,
                               x0=seed, bounds=(lower, upper), method='trf',
                               max_nfev=900, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        if best is None or 2*result.cost < 2*best.cost:
            best = result
    if best is None or not best.success:
        raise RuntimeError(f'{factors}-factor optimizer did not converge: {best.message if best else "unknown"}')
    p = best.x.copy()
    if factors == 2 and p[0] > p[1]:
        p = np.array([p[1], p[0], p[3], p[2], p[4]])
    return p, predict(p), float(np.sqrt(2 * best.cost)), int(best.nfev)


def h_j_m_drift(u: np.ndarray, p: np.ndarray):
    a, b, sigma, eta, rho = p
    ea = np.exp(-a * u)
    eb = np.exp(-b * u)
    Ia = -np.expm1(-a * u) / a
    Ib = -np.expm1(-b * u) / b
    return sigma * ea * (sigma * Ia + rho * eta * Ib) + eta * eb * (eta * Ib + rho * sigma * Ia)


def drift_check(p: np.ndarray):
    a, b, sigma, eta, rho = p
    u = np.linspace(0, 30, 3001)
    B1 = -np.expm1(-a * u) / a
    B2 = -np.expm1(-b * u) / b
    var = sigma**2*B1**2 + eta**2*B2**2 + 2*rho*sigma*eta*B1*B2
    derivative = (sigma**2*B1*np.exp(-a*u) +
                  eta**2*B2*np.exp(-b*u) +
                  rho*sigma*eta*(B1*np.exp(-b*u) + B2*np.exp(-a*u)))
    target = h_j_m_drift(u, p)
    # Independently calculate 0.5 numerical derivative of integrated forward variance;
    # centered finite differences exclude endpoints and have a nonzero truncation error.
    numerical = 0.5 * np.gradient(var, u)
    max_error = float(np.max(np.abs(numerical[1:-1] - target[1:-1])))
    algebra_error = float(np.max(np.abs(target - derivative)))
    return max_error, algebra_error


def pca_explained(cov: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvalues / max(eigenvalues.sum(), 1e-20)


def summary_plot_lines(train_cov, hold_cov, cov1, cov2, output):
    observed = np.sqrt(np.maximum(np.diag(hold_cov), 0)) * 1e4
    pred1 = np.sqrt(np.maximum(np.diag(cov1), 0)) * 1e4
    pred2 = np.sqrt(np.maximum(np.diag(cov2), 0)) * 1e4
    fit_obs = np.sqrt(np.maximum(np.diag(train_cov), 0)) * 1e4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(TENORS, fit_obs, label='Observed training forward volatility')
    ax.plot(TENORS, observed, label='Observed holdout forward volatility')
    ax.plot(TENORS, pred1, label='One-factor fit from training')
    ax.plot(TENORS, pred2, label='Two-factor fit from training')
    ax.set(title='Annualized forward-rate volatility by tenor', xlabel='Years', ylabel='bps / sqrt(year)')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(output/'01_forward_volatility_by_tenor.png', dpi=160); plt.close(fig)

    # Plot signed covariance residuals on a common scale to expose model error.
    resid1 = cov1 - hold_cov
    resid2 = cov2 - hold_cov
    vmax = max(abs(resid1).max(), abs(resid2).max(), 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, residual, title in zip(axes,[resid1, resid2], ['One-factor holdout residual','Two-factor holdout residual']):
        im=axis.imshow(residual * 1e8, cmap='coolwarm', vmin=-vmax*1e8, vmax=vmax*1e8, origin='lower')
        axis.set(title=title, xlabel='Tenor index: 1Y to 30Y', ylabel='Tenor index')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Annualized covariance residual (bps²/year)')
    fig.subplots_adjust(right=.86, wspace=.35)
    fig.savefig(output/'02_holdout_covariance_residuals.png', dpi=160); plt.close(fig)


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--forwards', default=str(HERE/'project2_phase1_outputs/daily_instantaneous_forwards_pct.csv'))
    arg.add_argument('--train-end', default='2017-12-31')
    arg.add_argument('--output', default=str(HERE/'project2_phase6b_outputs'))
    args = arg.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = source_data(Path(args.forwards))
    changes = sample_changes(data)
    train = changes.loc[changes.index <= pd.Timestamp(args.train_end)]
    hold = changes.loc[changes.index > pd.Timestamp(args.train_end)]
    if len(train) < 150 or len(hold) < 150:
        raise ValueError(f'Insufficient TRAIN or HOLDOUT daily observations: {len(train)}, {len(hold)}')

    c_train, c_hold = annual_cov(train.values), annual_cov(hold.values)
    param1, c1, obj1, nev1 = fit_cov(c_train, 1)
    param2, c2, obj2, nev2 = fit_cov(c_train, 2)
    pca_train, pca_hold = pca_explained(c_train), pca_explained(c_hold)
    drift_fd, drift_alg = drift_check(param2)
    if np.linalg.eigvalsh(c2).min() < -1e-10:
        raise ValueError('Fitted HJM covariance not positive semidefinite.')
    if not np.isfinite(c2).all() or not np.isfinite(drift_fd):
        raise ValueError('Numerical failure: non-finite covariance or drift test.')

    rows = []
    for model, fitted in [('1-factor exponential HJM',c1),('2-factor exponential HJM',c2)]:
        rows.append({'model':model,
                     'train_covariance_rel_error_pct':relative_frobenius(fitted,c_train),
                     'holdout_covariance_rel_error_pct':relative_frobenius(fitted,c_hold)})
    errors = pd.DataFrame(rows)
    errors.to_csv(out/'phase6b_covariance_errors.csv',index=False)
    pd.DataFrame({'tenor_years':TENORS,
        'train_vol_bps_per_sqrt_year':1e4*np.sqrt(np.maximum(np.diag(c_train),0)),
        'holdout_vol_bps_per_sqrt_year':1e4*np.sqrt(np.maximum(np.diag(c_hold),0)),
        'onefactor_vol_bps_per_sqrt_year':1e4*np.sqrt(np.maximum(np.diag(c1),0)),
        'twofactor_vol_bps_per_sqrt_year':1e4*np.sqrt(np.maximum(np.diag(c2),0))}
        ).to_csv(out/'phase6b_forward_volatility.csv',index=False)
    pd.DataFrame({'component':np.arange(1,31),'train_variance_share':pca_train,
                  'holdout_variance_share':pca_hold}).to_csv(out/'phase6b_forward_pca.csv',index=False)
    for nm, mat in [('train_empirical',c_train),('holdout_empirical',c_hold),('onefactor_trainfit',c1),('twofactor_trainfit',c2)]:
        pd.DataFrame(mat,index=[f'{i}Y' for i in range(1,31)],columns=[f'{i}Y' for i in range(1,31)]).to_csv(out/f'phase6b_covariance_{nm}.csv')
    summary_plot_lines(c_train,c_hold,c1,c2,out)

    p = {'one_factor':{'a':float(param1[0]),'sigma':float(param1[1])},
         'two_factor':dict(zip(['a','b','sigma','eta','rho'],map(float,param2))),
         'data_source':str(Path(args.forwards).resolve()),
         'input_unit':'Percent, converted to decimals',
         'data_limitation':'Historical daily fixed-tenor GSW forward-change covariance; not derivative-implied Q calibration'}
    (out/'phase6b_parameters.json').write_text(json.dumps(p,indent=2))
    report = (
        '='*76+'\nPHASE 6B — REAL HISTORICAL TREASURY FORWARD COVARIANCE / HJM\n'+'='*76+'\n'
        f'Source: {Path(args.forwards).resolve()}\n'
        f'Daily forward changes TRAIN: {train.index.min().date()} to {train.index.max().date()} ({len(train)})\n'
        f'Daily forward changes HOLDOUT: {hold.index.min().date()} to {hold.index.max().date()} ({len(hold)})\n'
        'Annualized covariance assumes 252 trading days per year. Forward rates are decimal annualized rates.\n'
        f'One-factor parameters: a={param1[0]:.6f}/yr, sigma={param1[1]:.6f}/sqrt(yr)\n'
        f'Two-factor parameters: a={param2[0]:.6f}/yr, b={param2[1]:.6f}/yr, '
        f'sigma={param2[2]:.6f}/sqrt(yr), eta={param2[3]:.6f}/sqrt(yr), rho={param2[4]:.6f}\n'
        f'One-factor covariance fit error TRAIN={relative_frobenius(c1,c_train):.3f}%, HOLDOUT={relative_frobenius(c1,c_hold):.3f}%\n'
        f'Two-factor covariance fit error TRAIN={relative_frobenius(c2,c_train):.3f}%, HOLDOUT={relative_frobenius(c2,c_hold):.3f}%\n'
        f'Forward daily-change PCA TRAIN: PC1={100*pca_train[0]:.3f}%, PC2={100*pca_train[1]:.3f}%, PC3={100*pca_train[2]:.3f}%\n'
        f'Forward daily-change PCA HOLDOUT: PC1={100*pca_hold[0]:.3f}%, PC2={100*pca_hold[1]:.3f}%, PC3={100*pca_hold[2]:.3f}%\n'
        f'HJM risk-neutral drift algebra identity max error: {drift_alg:.3e}\n'
        f'HJM risk-neutral drift finite-difference max error: {drift_fd:.3e}\n'
        f'Optimizer evaluations (best starts): onefactor={nev1}, twofactor={nev2}\n'
        'LIMITATIONS: Parameters fit HISTORICAL P-measure forward covariance, not Q-measure option prices.\n'
        'Forward differences use fixed time-to-maturity daily series as short-step proxy, not fixed T forwards.\n'
        'No SOFR OIS discounting / live swaption calibration / empirical option-pricing claim.\n'
        f'Output folder: {out.resolve()}\n'
    )
    print(report)
    (out/'phase6b_report.txt').write_text(report)


if __name__ == '__main__':
    main()
