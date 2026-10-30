"""PROJECT 2 — PHASE 3: Dynamic Nelson–Siegel (DNS) yield curve forecasting.

Run in VS Code (Python 3.9+):
    python3 -m pip install numpy pandas matplotlib
    python3 project2_phase3_dns_forecasting.py

Defaults to Phase 1 file next to this script:
    project2_phase1_outputs/monthly_zero_yields_pct.csv

Design:
- Observe continuously compounded Fed fitted Treasury ZERO yields (1Y to 30Y, PERCENT).
- Re-estimate fixed-loading NS factors each month. Phase 2's optimized lambda varies
  by month, so its coefficients are not a constant-basis factor time series.
- Fixed lambda=0.25 (1/years) is declared before the forecast experiment.
- Expanding-window forecasting: only data up to forecast origin are used.
- Direct multi-step horizon achieved by recursion of 1-step factor regressions.
- Compare yield random walk (observed yield held constant), factor persistence,
  independent factor AR(1), and 3-factor VAR(1).
- Evaluate horizons 1/3/6 monthly steps at 2Y/5Y/10Y/20Y/30Y and ALL 30 tenors.
- Report RMSE/MAE/Bias in bps; positive improvement over RW is better.

Limitations:
- Source data are fitted Fed curves (Svensson-family), not actual raw bond prices.
- Forecast residuals at horizons 3/6 overlap; RMSEs aren't independent observations.
- No volatility term-premium or arbitrage-free modeling yet: AFNS/G2++/HJM later.
- Choice of lambda and hyperparameters MUST NOT be optimized on evaluation period.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MATURITIES = np.arange(1,31, dtype=int)
COLUMNS = [f'{m}Y' for m in MATURITIES]
TENORS = [2, 5, 10, 20, 30]
HORIZONS = (1, 3, 6)
MODELS = ('Yield-RW', 'NS-Persistence', 'DNS-AR1', 'DNS-VAR1')


def ns_design(t: np.ndarray, lam: float) -> np.ndarray:
    """Design: y(tau)=beta0+beta1*L1(tau)+beta2*L2(tau)."""
    t = np.asarray(t, dtype=float)
    if not (lam > 0 and np.isfinite(lam)):
        raise ValueError('Fixed lambda must be positive, finite, and expressed per year')
    x = lam * t
    s = -np.expm1(-x)/x
    c = s - np.exp(-x)
    return np.column_stack((np.ones_like(x), s, c))


def read_yields(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing Phase 1 monthly CSV: {path}\n'
          'Save this script beside project2_phase1_outputs or use --input <path>')
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df = df[COLUMNS].apply(pd.to_numeric,errors='raise').sort_index()
    if df.index.isna().any() or df.index.has_duplicates or df.isna().any().any():
        raise ValueError('Dates are invalid or duplicated, or yield values are missing')
    arr = df.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError('Dataset contains infinite or NaN yields')
    period = df.index.to_period('M')
    if period.has_duplicates:
        raise ValueError('More than one observation per month')
    expected = pd.period_range(period.min(),period.max(),freq='M')
    if not period.equals(expected):
        raise ValueError('Missing calendar months, correct data alignment first')
    if (np.abs(arr)>40).any():
        raise ValueError('Implausibly large yields >40%: verify data units')
    return df


def extract_factors(y: np.ndarray, design: np.ndarray):
    """Cross-sectional OLS; x (N,3), fitted (N,30); each date independent."""
    coefs, *_ = np.linalg.lstsq(design, y.T, rcond=None)
    factors = coefs.T
    fitted = factors @ design.T
    return factors, fitted


def ar1_coeff(factors_train: np.ndarray):
    """OLS AR(1), intercept per factor. Shape (2, 3)."""
    x = np.column_stack((np.ones(len(factors_train)-1), factors_train[:-1]))
    # Independent AR(1) for each factor.
    beta = np.zeros((2, 3))
    for j in range(3):
        beta[:,j], *_ = np.linalg.lstsq(x[:,[0,j+1]], factors_train[1:,j], rcond=None)
    return beta


def var1_coeff(factors_train: np.ndarray):
    """OLS VAR(1), coefficient matrix with intercept (4,3)."""
    x = np.column_stack((np.ones(len(factors_train)-1), factors_train[:-1]))
    beta, *_ = np.linalg.lstsq(x, factors_train[1:], rcond=None)
    return beta


def forecast_factors(f0: np.ndarray, beta: np.ndarray, h: int, model: str) -> np.ndarray:
    pred = f0.copy()
    for _ in range(h):
        if model == 'DNS-AR1':
            pred = beta[0] + beta[1] * pred
        elif model == 'DNS-VAR1':
            pred = beta[0] + pred @ beta[1:]
        else:
            raise ValueError(model)
    return pred


def walk_forward(df: pd.DataFrame, lam: float, min_train: int, horizons=HORIZONS):
    y = df.to_numpy(dtype=float)
    design = ns_design(MATURITIES, lam)
    factors, fitted = extract_factors(y, design)
    n = len(df)
    if min_train < 24:
        raise ValueError('--min-train must be at least 24 months')
    if n < min_train + max(horizons) + 1:
        raise ValueError('Not enough monthly observations to evaluate each horizon')
    rows = []
    allerr = []
    for h in horizons:
        for origin in range(min_train - 1, n-h):
            # Fitting uses only dates 0,...,origin (no future data).
            ftrain = factors[:origin+1]
            ar = ar1_coeff(ftrain)
            var = var1_coeff(ftrain)
            preds = {
                'Yield-RW': y[origin].copy(),
                'NS-Persistence': fitted[origin].copy(),
                'DNS-AR1': forecast_factors(factors[origin],ar,h,'DNS-AR1') @ design.T,
                'DNS-VAR1': forecast_factors(factors[origin],var,h,'DNS-VAR1') @ design.T,
            }
            actual = y[origin+h]
            for model, pred in preds.items():
                if not np.isfinite(pred).all():
                    raise ArithmeticError(f'Forecast is nonfinite: {model}, origin {origin}')
                diff_bps = (pred-actual)*100.0
                allerr.append({
                    'horizon_months':h, 'model':model, 'origin_month':df.index[origin],
                    'target_month':df.index[origin+h], 'maturity':'ALL_30',
                    'squared_error_mean_bps2': float(np.mean(diff_bps**2)),
                    'absolute_error_mean_bps': float(np.mean(np.abs(diff_bps))),
                    'bias_mean_bps':float(np.mean(diff_bps)),
                })
                for t in TENORS:
                    i = t - 1
                    rows.append({
                        'origin_month':df.index[origin], 'target_month':df.index[origin+h],
                        'horizon_months':h, 'model':model, 'tenor':f'{t}Y',
                        'actual_yield_pct':float(actual[i]),
                        'forecast_yield_pct':float(pred[i]),
                        'error_bps':float(diff_bps[i]),
                    })
    return (pd.DataFrame(rows), pd.DataFrame(allerr),
            pd.DataFrame(factors, index=df.index, columns=['level_beta0_pct','slope_beta1_pct','curvature_beta2_pct']),
            pd.Series(np.sqrt(np.mean(((fitted-y)*100)**2,axis=1)), index=df.index,name='ns_cross_sectional_rmse_bps'))


def evaluate(forecast: pd.DataFrame, allerr: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (h, model, t), sub in forecast.groupby(['horizon_months','model','tenor'],sort=True):
        errors = sub['error_bps'].to_numpy()
        records.append(dict(horizon_months=h, model=model, tenor=t,
          n_origins=len(errors), rmse_bps=float(np.sqrt(np.mean(errors**2))),
          mae_bps=float(np.mean(np.abs(errors))), bias_bps=float(np.mean(errors))))
    for (h, model), sub in allerr.groupby(['horizon_months','model'],sort=True):
        records.append(dict(horizon_months=h, model=model, tenor='ALL_30',
          n_origins=len(sub), rmse_bps=float(np.sqrt(sub['squared_error_mean_bps2'].mean())),
          mae_bps=float(sub['absolute_error_mean_bps'].mean()),
          bias_bps=float(sub['bias_mean_bps'].mean())))
    metrics = pd.DataFrame(records).sort_values(['horizon_months','tenor','model']).reset_index(drop=True)
    rw = metrics[metrics.model=='Yield-RW'].set_index(['horizon_months','tenor'])['rmse_bps']
    metrics['rmse_improvement_vs_rw_pct'] = [
        100*(1-r.rmse_bps / rw.loc[(r.horizon_months,r.tenor)])
        if rw.loc[(r.horizon_months,r.tenor)] > 0 else np.nan
        for r in metrics.itertuples(index=False)]
    return metrics


def charts(df, factor_df, crossfit, forecast, metrics, out):
    fig, axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,col, label in zip(axs, factor_df.columns, ['Level beta0','Slope beta1','Curvature beta2']):
        ax.plot(factor_df.index,factor_df[col],lw=1.6)
        ax.set_ylabel('percent')
        ax.set_title(label)
        ax.grid(alpha=0.23)
    fig.suptitle('Fixed-loading Nelson–Siegel factors (descriptive, not forecasts)')
    fig.tight_layout()
    fig.savefig(out/'01_dns_factor_history.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(crossfit.index,crossfit.values,lw=1.6)
    ax.set(title='Monthly NS fixed-loading curve-fitting RMSE',ylabel='RMSE (bps)',xlabel='Month')
    ax.grid(alpha=0.25);fig.tight_layout()
    fig.savefig(out/'02_fixed_ns_curve_rmse.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    for ax,h in zip(axs,[1,6]):
        sub=forecast[(forecast.tenor=='10Y')&(forecast.horizon_months==h)]
        series=sub[sub.model=='Yield-RW'].set_index('target_month').sort_index()
        ax.plot(series.index,series.actual_yield_pct,label='Observed 10Y',color='black',lw=1.8)
        for model in ['Yield-RW','DNS-AR1','DNS-VAR1']:
            pred=sub[sub.model==model].set_index('target_month').sort_index()
            ax.plot(pred.index,pred.forecast_yield_pct,label=model,alpha=0.75,lw=1.1)
        ax.set(title=f'{h}-month ahead forecast (10Y)',xlabel='Target month',ylabel='Yield (%)')
        ax.grid(alpha=0.2)
    axs[0].legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(out/'03_10year_out_of_sample_forecasts.png',dpi=170);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    focus=['2Y','5Y','10Y','20Y','30Y']
    for ax,h in zip(axs,HORIZONS):
        ss=metrics[(metrics.horizon_months==h)&(metrics.tenor.isin(focus))]
        x=np.arange(len(focus))
        for k,model in enumerate(MODELS):
            p=ss[ss.model==model].set_index('tenor').reindex(focus)
            ax.bar(x + (k-1.5)*0.19,p.rmse_bps.values,width=0.19,label=model)
        ax.set_xticks(x,focus)
        ax.set(title=f'{h}-month forecasts',xlabel='Treasury maturity')
        ax.grid(axis='y',alpha=0.2)
    axs[0].set_ylabel('Out-of-sample RMSE (bps)')
    axs[-1].legend(fontsize=8,loc='upper left')
    fig.suptitle('No-change yield forecast vs dynamic Nelson–Siegel')
    fig.tight_layout();fig.savefig(out/'04_forecast_rmse_comparison.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,default=HERE/'project2_phase1_outputs'/'monthly_zero_yields_pct.csv')
    parser.add_argument('--output',type=Path,default=HERE/'project2_phase3_outputs')
    parser.add_argument('--lambda-per-year',type=float,default=0.25)
    parser.add_argument('--min-train',type=int,default=96,help='Months in first expanding window, default 96')
    args=parser.parse_args()
    df=read_yields(args.input.expanduser())
    if not np.isfinite(args.lambda_per_year) or args.lambda_per_year <= 0:
        parser.error('--lambda-per-year must be positive and finite')
    if args.min_train<24 or len(df)<args.min_train+max(HORIZONS)+1:
        parser.error('--min-train must be >=24 with enough history remaining for the horizons')
    out=args.output.expanduser();out.mkdir(parents=True,exist_ok=True)
    forecast, allerr, factors, fit_rmse=walk_forward(df,args.lambda_per_year,args.min_train)
    metrics=evaluate(forecast,allerr)
    forecast.to_csv(out/'phase3_forecasts_selected_tenors.csv',index=False,float_format='%.9f')
    metrics.to_csv(out/'phase3_forecast_metrics.csv',index=False,float_format='%.7f')
    factors.to_csv(out/'phase3_dns_factors_monthly.csv',float_format='%.9f')
    fit_rmse.to_csv(out/'phase3_ns_fixed_lambda_fit_rmse.csv',header=True,float_format='%.8f')
    charts(df,factors,fit_rmse,forecast,metrics,out)
    rows=[
        'PHASE 3 — DNS FORECASTING / EXPANDING-WINDOW BACKTEST',
        f'Sample: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} months)',
        f'Fixed lambda (per year): {args.lambda_per_year:.5f}',
        f'Initial training window: {args.min_train} months',
        f'Average monthly fixed-NS cross-sectional RMSE: {fit_rmse.mean():.4f} bps',
        'All RMSEs are genuine later-month yield forecasts, NOT maturity holdouts.',
        'Forecast horizon measured in calendar months; targets align by observation index.',
        'Units: continuous zero yields in percent; forecast errors reported in bps.',
        'Evaluation forecasts for horizons >1 overlap; no significance test is implied.',
        'GSW source curves are Svensson-fitted estimates and can be revised.',
        '',
    ]
    for h in HORIZONS:
        for tenor in ['10Y','30Y','ALL_30']:
            rows.append(f'HORIZON {h} MONTHS / MATURITY {tenor}')
            for model in MODELS:
                m=metrics[(metrics.horizon_months==h)&(metrics.tenor==tenor)&(metrics.model==model)].iloc[0]
                rows.append(f" {model:16s} RMSE={m.rmse_bps:8.3f} bp  MAE={m.mae_bps:8.3f} bp  RW improvement={m.rmse_improvement_vs_rw_pct:+7.2f}%  origins={int(m.n_origins)}")
            rows.append('')
    (out/'phase3_methodology_and_results.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('\n'+'='*80+'\n'+'\n'.join(rows)+'\n'+'='*80)
    print('Saved 4 CSVs, 4 charts, and methodology report to:',out.resolve())

if __name__=='__main__':
    main()
