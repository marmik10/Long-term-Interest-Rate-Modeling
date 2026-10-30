#!/usr/bin/env python3
"""Phase 5A — G2++: historical covariance diagnostics and Q pricing identities.

Input: Phase 1 'project2_phase1_outputs/daily_zero_yields_pct.csv'.

IMPORTANT: Historical realized yield covariance is NOT a calibration of
risk-neutral G2++ dynamics to swaption prices. The estimated parameters are
historical moment-matched PROXIES that are used as an illustrative set of Q
parameters only to demonstrate correct mathematical pricing identities.
Never label resulting derivatives as market-calibrated option prices.

We fit mean reversion and diffusion parameters to the instantaneous covariance
approximation for changes in (2, 5, 10, 20, 30)-year yields on the TRAIN sample
(2010 through 2017), then evaluate covariance reproduction in a later sample.

G2++ under Q:
    r(t) = phi(t) + x(t) + z(t)
    dx = -a*x dt + sigma dW1, dz = -b*z dt + eta dW2,
    d<W1,W2> = rho dt.
    V(T) = Var_Q[ integral_0^T (x(u)+z(u)) du ].
    phi(t) = f_market(0,t) + 1/2 V'(t).
    P(t,T) = P(0,T)/P(0,t) *
             exp[-B_a(T-t)x_t - B_b(T-t)z_t
                 + 1/2 (V(T-t)-V(T)+V(t))].

For t=0, x=z=0, P(0,T) is reproduced exactly for ANY valid G2++ params.

Requires: numpy, pandas, scipy, matplotlib.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TENORS = np.arange(1,31,dtype=float)
SELECTED = np.array([2,5,10,20,30],dtype=float)
SELCOLS = [f'{int(t)}Y' for t in SELECTED]


def B(k, t):
    t = np.asarray(t,dtype=float)
    return -np.expm1(-k*t)/k


def j(k1,k2,T):
    """Integral from 0 to T of B(k1,s)*B(k2,s) ds.

    Numerical quadrature uses stable closed-form primitive in regimes here.
    """
    T=np.asarray(T,dtype=float)
    return (T + np.expm1(-k1*T)/k1 + np.expm1(-k2*T)/k2
            - np.expm1(-(k1+k2)*T)/(k1+k2))/(k1*k2)


def var_integral(T,a,b,sigma,eta,rho):
    T=np.asarray(T,dtype=float)
    return (sigma**2*j(a,a,T) + eta**2*j(b,b,T) +
            2*rho*sigma*eta*j(a,b,T))


def implied_cov(tenors,p):
    a,b,sigma,eta,rho=p
    u=np.column_stack((B(a,tenors)/tenors, B(b,tenors)/tenors))
    diffusion=np.array([[sigma**2,rho*sigma*eta],[rho*sigma*eta,eta**2]])
    return u @ diffusion @ u.T


def rank1_cov(tenors,p):
    a,sigma=p
    u=B(a,tenors)/tenors
    return sigma**2*np.outer(u,u)


def calibrated_cov(moment, tenors, two=True):
    iu=np.triu_indices(len(tenors))
    # Scale errors by observed variance to keep units manageable (rates decimal).
    scale=max(float(np.mean(np.diag(moment))),1e-12)
    if two:
        f=lambda p: implied_cov(tenors,p)
        lo=[0.012,0.085,0.0001,0.0001,-0.975]
        hi=[1.2,3.0,0.12,0.12,0.975]
        guesses=[(0.09,0.7,.012,.009,-.65),(.04,1.1,.018,.006,-.4),
                 (.14,.5,.02,.008,.0),(.035,.25,.010,.018,-.85),
                 (.18,1.8,.02,.012,.3)]
        # Soft penalty on b<a to keep slow and fast factors recognizable.
        def resid(p):
            r=(f(p)-moment)[iu]/scale
            return np.r_[r, 2*max(p[0]+0.03-p[1],0)]
    else:
        f=lambda p: rank1_cov(tenors,p)
        lo=[0.012,.0001]; hi=[3.,.12]
        guesses=[(.09,.016),(.25,.02),(.7,.03),(1.8,.04)]
        resid=lambda p: (f(p)-moment)[iu]/scale
    best=None
    for g in guesses:
        opt=least_squares(resid,g,bounds=(lo,hi),max_nfev=3000,
                          ftol=1e-11,xtol=1e-11,gtol=1e-11)
        if best is None or np.linalg.norm(resid(opt.x))<np.linalg.norm(resid(best.x)):
            best=opt
    if not best.success:
        raise RuntimeError(f'Covariance optimizer did not converge: {best.message}')
    return np.asarray(best.x), float(np.linalg.norm(resid(best.x))), bool(best.success)


def load_daily(path):
    df=pd.read_csv(path,index_col=0,parse_dates=True)
    if not set([f'{t}Y' for t in range(1,31)]).issubset(df.columns):
        raise ValueError('Expected all columns 1Y to 30Y from Phase 1.')
    if df.index.has_duplicates or not df.index.is_monotonic_increasing:
        df=df.sort_index(); df=df[~df.index.duplicated(keep='last')]
    df=df[[f'{t}Y' for t in range(1,31)]].apply(pd.to_numeric,errors='coerce')
    df=df.replace([np.inf,-np.inf],np.nan).dropna(how='any')
    if len(df)<600:
        raise ValueError('Too few complete daily yield curves for the G2++ experiment.')
    if (df.abs().to_numpy()>25).any():
        raise ValueError('Implausible rates over 25% detected; check percent units.')
    return df/100.0


def day_differences(df):
    # Avoid month-long/holiday data gaps: only changes between consecutive
    # observations at most 5 calendar days apart are treated as daily changes.
    y=df[SELCOLS]
    gap=y.index.to_series().diff().dt.days
    moves=y.diff()[gap.between(1,5)].dropna()
    return moves


def empirical_cov(moves):
    return moves.cov().to_numpy(dtype=float)*252.0


def fit_error(emp,model):
    return float(np.linalg.norm(model-emp,'fro')/np.linalg.norm(emp,'fro'))


def zcb_price(t,T,x,z,zero_curve,p):
    """Exact G2++ exponential-affine discount bond formula.

    zero_curve maps maturity (years from initial date) to P(0,T).
    x,z are factors at time t, in decimal short-rate units.
    """
    a,b,sigma,eta,rho=p
    if T<=t or t<0 or T>30:
        raise ValueError('Require 0 <= t < T <= 30 years.')
    v=lambda s: var_integral(s,a,b,sigma,eta,rho)
    delta=T-t
    return float(zero_curve(T)/zero_curve(t)*np.exp(
        -B(a,delta)*x - B(b,delta)*z +
        0.5*(v(delta)-v(T)+v(t))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default=str(ROOT/'project2_phase1_outputs'/'daily_zero_yields_pct.csv'))
    ap.add_argument('--output',default=str(ROOT/'project2_phase5a_outputs'))
    ap.add_argument('--train-end',default='2017-12-31')
    ap.add_argument('--mc-paths',type=int,default=150000)
    args=ap.parse_args()
    if args.mc_paths<5000: raise ValueError('MC paths should be >=5000')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    daily=load_daily(args.input)
    moves=day_differences(daily)
    train=moves.loc[:args.train_end]
    test=moves.loc[args.train_end:]
    test=test[test.index>pd.Timestamp(args.train_end)]
    if len(train)<400 or len(test)<200:
        raise ValueError('Need >=400 training daily moves and >=200 test daily moves.')
    c_train=empirical_cov(train);c_test=empirical_cov(test)
    q2, loss2, ok2=calibrated_cov(c_train,SELECTED,True)
    q1, loss1, ok1=calibrated_cov(c_train,SELECTED,False)
    mod2=implied_cov(SELECTED,q2);mod1=rank1_cov(SELECTED,q1)
    train_2=fit_error(c_train,mod2);train_1=fit_error(c_train,mod1)
    test_2=fit_error(c_test,mod2);test_1=fit_error(c_test,mod1)

    today=daily.iloc[-1]
    # Initial curve discount prices at integer maturities, log-linear in
    # maturity for fractional intermediate years. P(0,0)=1 exactly.
    xgrid=np.r_[0.,TENORS]
    logp=np.r_[0.,-TENORS*today.to_numpy(dtype=float)]
    def P0(T):
        if not 0<=T<=30: raise ValueError('Time beyond supplied curve.')
        return float(np.exp(np.interp(T,xgrid,logp)))

    reconstruction=np.array([zcb_price(0,T,0,0,P0,q2) for T in TENORS])
    market=np.array([P0(T) for T in TENORS])
    max_fit_error=float(np.max(np.abs(reconstruction-market)))
    if max_fit_error>2e-13:
        raise AssertionError('Exact initial curve identity failed.')
    # Monte Carlo integrated Gaussian distribution: E[exp(-int_0^T r)]=P0(T).
    T=10.; a,b,sig,eta,rho=q2
    vv=float(var_integral(T,*q2))
    if not np.isfinite(vv) or vv<=0:
        raise ValueError('Invalid Gaussian integrated-rate variance.')
    rng=np.random.default_rng(20261027)
    z=rng.standard_normal(args.mc_paths)
    price_paths=P0(T)*np.exp(-vv/2-np.sqrt(vv)*z)
    mc=float(np.mean(price_paths)); se=float(np.std(price_paths,ddof=1)/np.sqrt(len(z)))
    ana=P0(T); mc_z=(mc-ana)/se
    if abs(mc_z)>5: raise AssertionError('Monte Carlo pricing discrepancy >5 SE.')

    # Bond prices at one year under illustrative factor shocks.
    t=1.
    sdx=np.sqrt(sig**2*(-np.expm1(-2*a*t))/(2*a))
    sdz=np.sqrt(eta**2*(-np.expm1(-2*b*t))/(2*b))
    scenarios=[('Base: x=z=0',0.,0.),('Slow factor +1 SD',sdx,0.),
               ('Fast factor +1 SD',0.,sdz),('Twist: slow +1SD, fast -1SD',sdx,-sdz)]
    rows=[]
    for name,x,z in scenarios:
        for maturity in [3.,6.,11.,21.,30.]:
            remaining=maturity-t
            price=zcb_price(t,maturity,x,z,P0,q2)
            yieldpct=-100*np.log(price)/remaining
            rows.append(dict(scenario=name,t_yr=t,maturity_from_today_yr=maturity,
                             remaining_tenor_yr=remaining,x_factor_pct=x*100,
                             z_factor_pct=z*100,bond_price_per_100=100*price,
                             implied_zero_yield_pct=yieldpct))
    scenarios_df=pd.DataFrame(rows)
    scenarios_df.to_csv(out/'phase5a_future_bond_scenarios.csv',index=False)
    parameter_df=pd.DataFrame([{'model':'G2++','a_per_year':a,'b_per_year':b,'sigma_decimal_sqrt_year':sig,
           'eta_decimal_sqrt_year':eta,'rho':rho,'train_cov_error_fro':train_2,
           'test_cov_error_fro':test_2},
          {'model':'1-factor','a_per_year':q1[0],'b_per_year':np.nan,
           'sigma_decimal_sqrt_year':q1[1], 'eta_decimal_sqrt_year':np.nan,
           'rho':np.nan,'train_cov_error_fro':train_1,'test_cov_error_fro':test_1}])
    parameter_df.to_csv(out/'phase5a_covariance_fit_summary.csv',index=False)
    covrows=[]
    for i,tenori in enumerate(SELECTED):
        for k,tenork in enumerate(SELECTED):
            covrows.append({'tenor_i_years':tenori,'tenor_j_years':tenork,
                'train_empirical_ann_cov':c_train[i,k],
                'holdout_empirical_ann_cov':c_test[i,k],
                'g2pp_ann_cov':mod2[i,k], 'one_factor_ann_cov':mod1[i,k]})
    pd.DataFrame(covrows).to_csv(out/'phase5a_covariance_matrices.csv',index=False)
    prices=pd.DataFrame({'maturity_years':TENORS,'market_yield_pct':today.to_numpy()*100,
            'market_discount_factor':market,'g2pp_discount_factor':reconstruction,
            'abs_discount_error':np.abs(reconstruction-market)})
    prices.to_csv(out/'phase5a_initial_curve_reconstruction.csv',index=False)
    
    plt.figure(figsize=(9,5)); plt.plot(TENORS,today.to_numpy()*100,linewidth=2,label='Published zero curve')
    plt.scatter(TENORS,-np.log(reconstruction)/TENORS*100,s=14,label='G2++ at t=0')
    plt.xlabel('Maturity (years)');plt.ylabel('Continuously compounded zero yield (%)')
    plt.title('Exact initial curve reconstruction (by deterministic shift)')
    plt.legend();plt.tight_layout();plt.savefig(out/'01_exact_initial_curve_fit.png',dpi=160);plt.close()

    plt.figure(figsize=(8,5));plt.imshow(np.corrcoef(train.to_numpy().T),cmap='coolwarm',vmin=-1,vmax=1)
    plt.xticks(np.arange(len(SELECTED)),[f'{int(v)}Y' for v in SELECTED]);
    plt.yticks(np.arange(len(SELECTED)),[f'{int(v)}Y' for v in SELECTED])
    plt.title('Historical correlations of daily yield changes (training)')
    plt.colorbar();plt.tight_layout();plt.savefig(out/'02_training_yield_change_correlation.png',dpi=160);plt.close()

    fig,ax=plt.subplots(figsize=(8,5));
    labels=['1-factor','G2++']; xx=np.arange(2)
    ax.bar(xx-.18,[train_1,train_2],width=.36,label='2010–2017 train')
    ax.bar(xx+.18,[test_1,test_2],width=.36,label='2018+ holdout')
    ax.set_xticks(xx,labels);ax.set_ylabel('Relative Frobenius covariance error')
    ax.set_title('Realized covariance reproduction (NOT option calibration)')
    ax.legend();fig.tight_layout();fig.savefig(out/'03_covariance_model_comparison.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5));
    for s in scenarios_df.scenario.unique():
        d=scenarios_df[scenarios_df.scenario==s]
        ax.plot(d.remaining_tenor_yr,d.implied_zero_yield_pct,marker='o',label=s)
    ax.set_xlabel('Remaining time to maturity at t=1 (years)')
    ax.set_ylabel('Scenario zero yield (%)');ax.set_title('G2++ two-factor term-structure scenarios at t=1')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'04_future_factor_scenarios.png',dpi=160);plt.close(fig)

    report=(f'PHASE 5A — G2++ TWO-FACTOR COVARIANCE & PRICING IDENTITIES\n'
        f'Dataset: {daily.index[0].date()} to {daily.index[-1].date()} | {len(daily)} daily curves\n'
        f'Covariance calibration TRAIN: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} daily changes)\n'
        f'Holdout: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} changes)\n'
        f'Historical proxy estimates: a={a:.5f}/yr, b={b:.5f}/yr, sigma={sig:.6f}/sqrt(yr), eta={eta:.6f}/sqrt(yr), rho={rho:.5f}\n'
        f'Train relative covariance error: 1-factor={train_1:.4%}, G2++={train_2:.4%}\n'
        f'Holdout relative covariance error: 1-factor={test_1:.4%}, G2++={test_2:.4%}\n'
        f'Initial Treasury curve pricing max |P_model-P_market|: {max_fit_error:.3e} per $1\n'
        f'10Y analytic P(0,10): {ana:.8f}; MC: {mc:.8f}; MC SE: {se:.8f}; z={mc_z:+.3f}\n'
        f'10Y integrated factor variance: {vv:.8e} (rate integral squared)\n'
        f'CRITICAL LIMITATION: G2++ diffusion parameters estimated from PHYSICAL historical yield covariance.\n'
        f'Risk-neutral calibration to swaption/cap volatilities has NOT been performed.\n'
        f'Exact initial-curve reproduction is enforced by phi(t) and is NOT independent proof of pricing accuracy.\n'
        f'Daily yield moves use 252 trading days/year and an instantaneous covariance approximation.\n'
        f'Sample Treasury curves are fitted GSW zero yields; no measurement error model in this phase.\n'
        f'No option pricing nor hedging conclusions should be attributed to this phase.\n'
        f'Saved outputs to {out.resolve()}\n')
    (out/'phase5a_report.txt').write_text(report)
    (out/'phase5a_parameters.json').write_text(json.dumps({
        'train_end':args.train_end,'g2pp_parameters':[float(v) for v in q2],
        'onefactor_parameters':[float(v) for v in q1],
        'note':'Historical covariance proxy, NOT Q calibration'},indent=2))
    print('\n'+'='*75+'\n'+report+'='*75)

if __name__=='__main__':main()
