#!/usr/bin/env python3
"""Project 2 / Phase 6C: REAL forward-curve covariance stability diagnostics.

Requires Phase 1 forward rates and Phase 6B covariance matrices. Uses NO
synthetic market data. All covariance candidates are frozen at 2017-12-31.
A train-PCA 3-factor covariance is a statistical benchmark, NOT a 3-factor HJM
market calibration. Oracle scaled holdout results are diagnostics ONLY.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TENORS = np.arange(1, 31)
DT = 1 / 252
REGIMES = [
    ('2018-2019', '2018-01-01', '2019-12-31'),
    ('2020-2021', '2020-01-01', '2021-12-31'),
    ('2022-2023', '2022-01-01', '2023-12-31'),
    ('2024-2026', '2024-01-01', '2026-12-31'),
]


def load_rates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing Phase 1 daily forward-rate file: {path}')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    if df.index.isna().any() or df.index.has_duplicates:
        raise ValueError('Invalid or duplicate data dates.')
    columns = [f'{t}Y' for t in TENORS]
    if not set(columns).issubset(df.columns):
        raise ValueError('Expect forward-rate columns 1Y through 30Y.')
    df = df[columns].apply(pd.to_numeric, errors='coerce')
    return df.replace([np.inf, -np.inf], np.nan) / 100


def changes_from_rates(rates: pd.DataFrame) -> pd.DataFrame:
    # Preserves missing source dates while differencing; same convention as Phase 6B.
    changes = rates.diff().iloc[1:].dropna(how='any')
    if len(changes) < 500 or not np.isfinite(changes.to_numpy()).all():
        raise ValueError('Insufficient valid forward-rate changes.')
    if np.abs(changes.to_numpy()).max() > .2:
        raise ValueError('Unexpected daily yield changes; verify percent vs decimal.')
    return changes


def annual_cov(arr: np.ndarray) -> np.ndarray:
    return np.cov(arr, rowvar=False, ddof=1) / DT


def fro_error(candidate: np.ndarray, actual: np.ndarray) -> float:
    return 100 * np.linalg.norm(candidate - actual, 'fro') / max(np.linalg.norm(actual, 'fro'), 1e-20)


def shape_error(candidate: np.ndarray, actual: np.ndarray) -> float:
    """Frobenius error after removing each matrix's absolute covariance scale."""
    a, b = np.linalg.norm(candidate, 'fro'), np.linalg.norm(actual, 'fro')
    if min(a, b) < 1e-20:
        raise ValueError('Zero covariance matrix unexpectedly encountered.')
    return 100 * np.linalg.norm(candidate/a - actual/b, 'fro')


def oracle_scalar(candidate: np.ndarray, actual: np.ndarray):
    """Use HOLDOUT actual covariance to choose scale -- diagnostic, NOT predictive."""
    scale = np.sum(candidate * actual) / max(np.sum(candidate * candidate), 1e-30)
    scale = max(float(scale), 0)
    return scale, fro_error(scale * candidate, actual)


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Missing Phase 6B output: {path}. Run Phase 6B first.')
    mat = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
    if mat.shape != (30,30) or not np.isfinite(mat).all():
        raise ValueError(f'Invalid saved covariance matrix: {path}')
    if not np.allclose(mat, mat.T, atol=1e-12):
        raise ValueError(f'Covariance is not symmetric: {path}')
    return .5 * (mat + mat.T)


def sorted_pca(cov: np.ndarray):
    vals, vecs = np.linalg.eigh(cov)
    ix = np.argsort(vals)[::-1]
    vals, vecs = np.maximum(vals[ix], 0), vecs[:,ix]
    return vals, vecs


def train_rank_cov(eigvals, eigvecs, rank: int):
    """Stable rank-k PSD reconstruction without BLAS matrix multiplication.

    Some macOS NumPy/Accelerate configurations report divide-by-zero or
    overflow inside the tiny 30x30 @ operation, even for finite inputs.
    A sum of scaled outer products avoids that operation and makes the
    intended symmetric PSD covariance reconstruction explicit.
    """
    vals = np.asarray(eigvals, dtype=np.float64)
    vecs = np.asarray(eigvecs, dtype=np.float64)
    if rank < 1 or rank > len(vals) or vecs.shape != (len(vals), len(vals)):
        raise ValueError('Invalid PCA rank or eigenvector dimensions.')
    if not np.isfinite(vals).all() or not np.isfinite(vecs).all():
        raise ValueError('Nonfinite PCA eigenvalues/eigenvectors.')
    if np.min(vals) < -1e-12:
        raise ValueError('Unexpected materially negative covariance eigenvalues.')

    result = np.zeros((len(vals), len(vals)), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for i in range(rank):
            result += max(float(vals[i]), 0.0) * np.outer(vecs[:, i], vecs[:, i])
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite reconstructed covariance.')
    # The trace of an orthonormal PCA rank-k reconstruction equals sum of
    # its retained eigenvalues; check this independent numerical invariant.
    expected_trace = float(np.sum(np.maximum(vals[:rank], 0)))
    if not np.isclose(np.trace(result), expected_trace, rtol=1e-9, atol=1e-14):
        raise FloatingPointError('PCA rank covariance failed trace validation.')
    return 0.5 * (result + result.T)


def save_plots(out, train_cov, hold_cov, eig_train, eig_hold, models, comparison, changes):
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(8,4.5))
    idx = np.arange(1,7)
    ax.plot(idx,100*eig_train[:6]/eig_train.sum(), marker='o', label='2010-2017 train')
    ax.plot(idx,100*eig_hold[:6]/eig_hold.sum(), marker='o', label='2018-2026 holdout')
    ax.set(xlabel='Principal component', ylabel='Explained variance (%)', title='Forward-rate PCA regime shift')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(out/'01_forward_pca_comparison.png',dpi=150);plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(TENORS,1e4*np.sqrt(np.maximum(np.diag(train_cov),0)), label='Train realized volatility')
    ax.plot(TENORS,1e4*np.sqrt(np.maximum(np.diag(hold_cov),0)), label='Holdout realized volatility')
    ax.set(xlabel='Forward tenor (years)', ylabel='Annualized bps / sqrt(year)',title='Observed forward-rate volatility shift')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out/'02_volatility_term_structure_shift.png',dpi=150);plt.close(fig)

    plot = comparison.set_index('model')
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(11,5))
    ax.bar(x-.18, plot['train_error_pct'], width=.36, label='Train')
    ax.bar(x+.18, plot['holdout_error_pct'], width=.36, label='Frozen model on holdout')
    ax.set_xticks(x,plot.index,rotation=17,ha='right')
    ax.set(ylabel='Covariance relative Frobenius error (%)', title='Predictive covariance benchmark comparison')
    ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout()
    fig.savefig(out/'03_frozen_covariance_model_errors.png',dpi=150);plt.close(fig)

    annual = (changes[[f'{t}Y' for t in (2,10,30)]]
              .rolling(252,min_periods=126).std() * np.sqrt(252) * 1e4)
    fig, ax = plt.subplots(figsize=(11,4.5))
    for col in annual.columns:
        ax.plot(annual.index,annual[col],label=col)
    ax.axvline(pd.Timestamp('2017-12-31'),linestyle='--',color='gray',label='Train/holdout split')
    ax.set(ylabel='Rolling annualized bps / sqrt(year)',title='252-session forward-rate volatility (historical)')
    ax.grid(alpha=.25);ax.legend(ncol=4);fig.tight_layout()
    fig.savefig(out/'04_rolling_forward_rate_volatility.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forwards',default=str(ROOT/'project2_phase1_outputs/daily_instantaneous_forwards_pct.csv'))
    parser.add_argument('--phase6b',default=str(ROOT/'project2_phase6b_outputs'))
    parser.add_argument('--train-end',default='2017-12-31')
    parser.add_argument('--output',default=str(ROOT/'project2_phase6c_outputs'))
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rates=load_rates(Path(args.forwards))
    delta=changes_from_rates(rates)
    split=pd.Timestamp(args.train_end)
    train,hold=delta.loc[:split],delta.loc[split+pd.Timedelta(days=1):]
    if len(train)<150 or len(hold)<150:
        raise ValueError('Training and holdout samples each require 150+ daily changes.')
    ctrain,chold=annual_cov(train.to_numpy()),annual_cov(hold.to_numpy())
    phase6b=Path(args.phase6b)
    saved_train=load_matrix(phase6b/'phase6b_covariance_train_empirical.csv')
    saved_hold=load_matrix(phase6b/'phase6b_covariance_holdout_empirical.csv')
    for label,a,b in [('training',ctrain,saved_train),('holdout',chold,saved_hold)]:
        err=fro_error(a,b)
        if err>1e-5:
            raise ValueError(f'Phase 6B {label} covariance differs ({err:.5g}%); ensure same input dates and train end.')
    c1=load_matrix(phase6b/'phase6b_covariance_onefactor_trainfit.csv')
    c2=load_matrix(phase6b/'phase6b_covariance_twofactor_trainfit.csv')
    ev,v=sorted_pca(ctrain);evh,vh=sorted_pca(chold)
    models={
        '1-factor HJM':c1,
        '2-factor HJM':c2,
        'Train-PCA rank 1':train_rank_cov(ev,v,1),
        'Train-PCA rank 2':train_rank_cov(ev,v,2),
        'Train-PCA rank 3':train_rank_cov(ev,v,3),
        'Full train covariance':ctrain,
    }
    comparison=[]
    for name, pred in models.items():
        oracle,oracle_err=oracle_scalar(pred,chold)
        comparison.append({'model':name,
           'train_error_pct':fro_error(pred,ctrain),
           'holdout_error_pct':fro_error(pred,chold),
           'holdout_shape_only_error_pct':shape_error(pred,chold),
           'holdout_oracle_scalar':oracle,
           'holdout_posthoc_scaled_error_pct_NOT_OOS':oracle_err})
    df=pd.DataFrame(comparison)
    df.to_csv(out/'phase6c_frozen_covariance_model_comparison.csv',index=False)

    factor_df=pd.DataFrame({'PC':np.arange(1,31),
        'train_var_share_pct':100*ev/ev.sum(),
        'holdout_var_share_pct':100*evh/evh.sum()})
    factor_df.to_csv(out/'phase6c_pca_variance_shift.csv',index=False)

    # Fixed training PC directions projected into holdout: NO retraining.
    align=pd.DataFrame({'PC':np.arange(1,4),
        'absolute_train_holdout_eigenvector_dot':[abs(np.dot(v[:,i],vh[:,i])) for i in range(3)]})
    align.to_csv(out/'phase6c_pca_vector_alignment.csv',index=False)
    regime=[]
    for label,from_date,to_date in REGIMES:
        subset=hold.loc[from_date:to_date]
        if len(subset)<60:continue
        cref=annual_cov(subset.to_numpy())
        row={'regime':label,'n_daily_changes':len(subset),
             'realized_trace_vol_ratio_vs_training':np.trace(cref)/np.trace(ctrain)}
        for name,pred in models.items():
            key=name.lower().replace('-','').replace(' ','_')
            row[f'{key}_error_pct']=fro_error(pred,cref)
            row[f'{key}_shape_error_pct']=shape_error(pred,cref)
        regime.append(row)
    pd.DataFrame(regime).to_csv(out/'phase6c_regime_errors.csv',index=False)
    pd.DataFrame({'tenor_years':TENORS,
        'train_vol_bps_sqrt_year':1e4*np.sqrt(np.maximum(np.diag(ctrain),0)),
        'holdout_vol_bps_sqrt_year':1e4*np.sqrt(np.maximum(np.diag(chold),0))}
        ).to_csv(out/'phase6c_term_structure_volatility.csv',index=False)
    save_plots(out,ctrain,chold,ev,evh,models,df,delta)
    print('='*76)
    print('PHASE 6C — HJM FORWARD VOLATILITY REGIME AND FACTOR DIAGNOSTICS')
    print('='*76)
    print('REAL INPUT:',Path(args.forwards).resolve())
    print(f'Train {train.index.min().date()} to {train.index.max().date()} ({len(train)} changes)')
    print(f'Holdout {hold.index.min().date()} to {hold.index.max().date()} ({len(hold)} changes)')
    print('Phase 6B covariance matrices verified against the input.')
    print('PCA low-rank reconstruction: stable outer-product summation with trace checks.')
    print(f'Holdout / training total covariance variance ratio: {np.trace(chold)/np.trace(ctrain):.4f}')
    print('Cumulative top-3 explained variance: train=%.2f%% holdout=%.2f%%'%(100*ev[:3].sum()/ev.sum(),100*evh[:3].sum()/evh.sum()))
    print('\nFROZEN TRAIN COVARIANCE MODELS (percent errors; lower is better):')
    print(df[['model','train_error_pct','holdout_error_pct','holdout_shape_only_error_pct']].round(3).to_string(index=False))
    print('\nHOLDOUT POST-HOC SCALE DIAGNOSTIC (NOT A LIVE FORECAST):')
    print(df[['model','holdout_oracle_scalar','holdout_posthoc_scaled_error_pct_NOT_OOS']].round(3).to_string(index=False))
    print('\nCAUTIONS: GSW fitted forwards, constant-time-to-maturity daily differences, no traded option vol.')
    print('Train-PCA rank-3 is a statistical benchmark, NOT a calibrated 3-factor HJM.')
    print('Any scale chosen from holdout data is post-hoc and NOT a predictive improvement.')
    print('The factor comparisons are not derivative pricing validations.')
    print('Saved outputs to:',out.resolve())
    (out/'phase6c_report.txt').write_text('PHASE 6C: all quoted results are historical P-measure covariance diagnostics.\n'+df.to_string(index=False)+'\n')

if __name__=='__main__':main()
