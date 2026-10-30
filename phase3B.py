"""PROJECT 2 — PHASE 3B: DNS decay sensitivity and paired forecast evaluation.

Requirements: Python 3.9+, numpy, pandas, scipy, matplotlib.
Run from the same directory as your Phase 1 output:
    python3 project2_phase3b_dns_robustness.py

The two decay parameters are fixed before this run: 0.25/year (Phase 3) and
0.7308/year (Diebold-Li conventional 0.0609/month with maturities in years).
The 0.7308 setting is a *post-results sensitivity analysis* rather than a new
independent holdout. Do NOT choose the best of the two based on the test window
and report the selected result as out-of-sample model selection.

HAC/DM-style tests are indicative only (model comparisons, many tenors/horizons,
multiple testing, estimated factors, overlapping multi-horizon forecasts).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MATURITIES = np.arange(1,31,dtype=float)
COLUMNS = [f'{i}Y' for i in range(1,31)]
TENORS = ['2Y','5Y','10Y','20Y','30Y','ALL_30']
HORIZONS = (1,3,6)
LAMBDAS = (("baseline_0.2500",.25), ("diebold_li_0.7308",.7308))
MODELS = ('Yield-RW','NS-Persistence','DNS-AR1','DNS-VAR1')


def load_yields(path: Path):
    if not path.is_file():
        raise FileNotFoundError('Expected Phase 1 data at '+str(path))
    df = pd.read_csv(path,index_col=0,parse_dates=[0])
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError('Expected 1Y to 30Y zero-yield columns in Phase 1 CSV')
    df = df[COLUMNS].sort_index().astype(float)
    if df.index.has_duplicates or df.index.isna().any() or df.isna().any().any():
        raise ValueError('Invalid/missing/duplicated monthly data')
    p = df.index.to_period('M')
    if not p.equals(pd.period_range(p.min(),p.max(),freq='M')):
        raise ValueError('Missing calendar months in monthly data')
    if not np.isfinite(df.to_numpy()).all() or (np.abs(df.to_numpy())>40).any():
        raise ValueError('Nonfinite or implausible yields: check units')
    return df


def loadings(lam):
    x=lam*MATURITIES
    slope = -np.expm1(-x)/x
    curvature = slope - np.exp(-x)
    return np.column_stack([np.ones_like(x),slope,curvature])


def fit_factors(y, load):
    coefficients, *_=np.linalg.lstsq(load,y.T,rcond=None)
    factors=coefficients.T
    fits=factors@load.T
    return factors,fits


def coefficients_ols(train):
    X = np.column_stack([np.ones(len(train)-1), train[:-1]])
    ar=np.empty((2,3))
    for j in range(3):
        ar[:,j], *_=np.linalg.lstsq(X[:,[0,j+1]],train[1:,j],rcond=None)
    var, *_=np.linalg.lstsq(X,train[1:],rcond=None)
    return ar,var


def predict_factors(f0,coef,h,mode):
    x=f0.copy()
    for _ in range(h):
        if mode=='DNS-AR1': x=coef[0]+coef[1]*x
        else: x=coef[0]+x@coef[1:]
    return x


def run_backtest(df,min_train):
    n=len(df)
    if min_train<24 or min_train+max(HORIZONS)>=n:
        raise ValueError('Need at least 24 training months and additional test months')
    y=df.to_numpy(float)
    rows=[]
    fit_rows=[]
    for spec,lam in LAMBDAS:
        L=loadings(lam)
        factors,fitted=fit_factors(y,L)
        fit_err=np.sqrt(np.mean(((y-fitted)*100)**2,axis=1))
        fit_rows.append({'specification':spec,'lambda_per_year':lam,
                         'mean_curve_fit_rmse_bps':float(np.mean(fit_err)),
                         'median_curve_fit_rmse_bps':float(np.median(fit_err))})
        for h in HORIZONS:
            for origin in range(min_train-1,n-h):
                ar,var=coefficients_ols(factors[:origin+1])
                preds={'Yield-RW':y[origin], 'NS-Persistence':fitted[origin],
                    'DNS-AR1':predict_factors(factors[origin],ar,h,'DNS-AR1')@L.T,
                    'DNS-VAR1':predict_factors(factors[origin],var,h,'DNS-VAR1')@L.T}
                actual=y[origin+h]
                for model,pred in preds.items():
                    err=(pred-actual)*100
                    if not np.isfinite(err).all():
                        raise FloatingPointError('Nonfinite forecast: '+model)
                    for tenor in TENORS:
                        ix=int(tenor[:-1])-1 if tenor!='ALL_30' else None
                        vals=err if ix is None else np.array([err[ix]])
                        rows.append({'specification':spec,'lambda_per_year':lam,
                            'horizon_months':h,'origin_month':df.index[origin],
                            'target_month':df.index[origin+h], 'model':model,'tenor':tenor,
                            'loss_bps2':float(np.mean(vals**2)),
                            'abs_error_bps':float(np.mean(np.abs(vals))),
                            'mean_error_bps':float(np.mean(vals))})
    return pd.DataFrame(rows),pd.DataFrame(fit_rows)


def metrics_and_dm(obs):
    cols=['specification','horizon_months','model','tenor']
    m=obs.groupby(cols,sort=True).agg(
        n_origins=('loss_bps2','size'),
        mean_loss_bps2=('loss_bps2','mean'),
        mae_bps=('abs_error_bps','mean'),
        bias_bps=('mean_error_bps','mean')).reset_index()
    m['rmse_bps']=np.sqrt(m.mean_loss_bps2)
    baseline=(m[m.model=='Yield-RW'].set_index(['specification','horizon_months','tenor'])['rmse_bps'])
    m['improvement_vs_rw_pct']=[100*(1-r.rmse_bps/baseline.loc[(r.specification,r.horizon_months,r.tenor)])
                                 for r in m.itertuples(index=False)]
    tests=[]
    for (spec,h,tenor), g in obs.groupby(['specification','horizon_months','tenor'],sort=True):
        rw=g[g.model=='Yield-RW'].set_index('origin_month')['loss_bps2']
        for model in ('NS-Persistence','DNS-AR1','DNS-VAR1'):
            sub=g[g.model==model].set_index('origin_month')['loss_bps2']
            d=(rw-sub).sort_index().to_numpy(dtype=float)  # >0 means model beats RW
            n=len(d)
            # Bartlett-kernel HAC variance for mean of squared-loss differential.
            lag=min(n-1,max(h-1,int(np.ceil(n**(1/3)))))
            dc=d-d.mean()
            long_run_var=np.dot(dc,dc)/n
            for k in range(1,lag+1):
                gk=np.dot(dc[k:],dc[:-k])/n
                long_run_var+=2*(1-k/(lag+1))*gk
            if long_run_var>0:
                stat=float(d.mean()/np.sqrt(long_run_var/n))
                p=float(2*NormalDist().cdf(-abs(stat)))
            else:
                stat=np.nan;p=np.nan
            tests.append({'specification':spec,'horizon_months':h,'tenor':tenor,'model':model,
                          'n_origins':n,'hac_lags':lag,
                          'mean_rw_minus_model_loss_bps2':float(d.mean()),
                          'dm_style_z':stat,'approx_two_sided_p':p})
    return m,pd.DataFrame(tests)


def plot_results(m,out):
    for ten in ['10Y','30Y','ALL_30']:
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
        for ax,h in zip(axes,HORIZONS):
            sub=m[(m.tenor==ten)&(m.horizon_months==h)&(m.model!='Yield-RW')]
            rw=m[(m.tenor==ten)&(m.horizon_months==h)&(m.model=='Yield-RW')].rmse_bps.iloc[0]
            labels=[]; vals=[]
            for s in [k for k,v in LAMBDAS]:
                for mod in ['NS-Persistence','DNS-AR1','DNS-VAR1']:
                    row=sub[(sub.specification==s)&(sub.model==mod)]
                    labels.append(('0.25' if s.startswith('baseline') else '0.73')+' '+mod.replace('DNS-',''))
                    vals.append(row.rmse_bps.iloc[0])
            ax.barh(labels,vals)
            ax.axvline(rw,color='black',linestyle='--',label=f'RW {rw:.1f} bp')
            ax.set(title=f'{h}-month ahead',xlabel='RMSE (bps)')
            ax.legend(fontsize=8); ax.grid(axis='x',alpha=.2)
            if ax!=axes[0]: ax.tick_params(axis='y',labelleft=False)
        fig.suptitle(f'Phase 3B: {ten} forecast sensitivity to DNS decay')
        fig.tight_layout()
        fig.savefig(out/f'phase3b_rmse_{ten}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for spec,lam in LAMBDAS:
        L=loadings(lam)
        ax.plot(MATURITIES,L[:,2],lw=1.8,label=f'Curvature loading: lambda {lam:g}/year')
    ax.set(xlabel='Maturity (years)',ylabel='Curvature loading',
           title='Nelson-Siegel curvature loadings for two pre-specified decay choices')
    ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'phase3b_decay_loading_comparison.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'project2_phase1_outputs'/'monthly_zero_yields_pct.csv')
    parser.add_argument('--output',type=Path,default=HERE/'project2_phase3b_outputs')
    parser.add_argument('--min-train',type=int,default=96)
    args=parser.parse_args()
    df=load_yields(args.input.expanduser())
    out=args.output.expanduser()
    out.mkdir(parents=True,exist_ok=True)
    obs, fit=run_backtest(df,args.min_train)
    m, tests=metrics_and_dm(obs)
    fit.to_csv(out/'phase3b_cross_sectional_fit.csv',index=False)
    m.to_csv(out/'phase3b_forecast_metrics.csv',index=False)
    tests.to_csv(out/'phase3b_hac_comparison.csv',index=False)
    plot_results(m,out)
    report=[
        'PHASE 3B — SENSITIVITY AND PAIRED FORECAST COMPARISON',
        f'Sample: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} months',
        f'First training window: {args.min_train} monthly observations',
        'All yield and NS parameters in percent; RMSEs in basis points.',
        'Decay rates: 0.25 and 0.7308 per year (pre-declared for this rerun).',
        'IMPORTANT: sensitivity after seeing initial OOS results is exploratory, not an untouched test.',
        'HAC comparisons: approximate two-sided tests vs random walk, unadjusted for multiple comparisons.',
        'Positive mean RW-minus-model loss means model has lower average squared forecast error.',
        'GSW yields are fitted zero-coupon curves; not raw Treasury trades.',
        '\nCROSS-SECTIONAL CURVE FIT:',
        fit.to_string(index=False,float_format=lambda x:f'{x:.3f}'),
        '\n10Y / 30Y / ALL_30 RMSE BY HORIZON:',
    ]
    for h in HORIZONS:
        report.append(f'\nHorizon {h} months:')
        s=m[(m.horizon_months==h)&m.tenor.isin(['10Y','30Y','ALL_30'])]
        report.append(s[['tenor','specification','model','rmse_bps','improvement_vs_rw_pct']]
          .to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    report.extend(['\nSaved 3 CSVs, 4 figures, methodology report to:',str(out)])
    (out/'phase3b_methodology_and_results.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report))

if __name__=='__main__':main()
