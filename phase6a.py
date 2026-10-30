#!/usr/bin/env python3
"""Project 2 — Phase 6A: two-factor Gaussian Heath–Jarrow–Morton foundations.

Uses Phase 1's latest continuously-compounded Treasury zero curve and forward
curve plus Phase 5A's *historical* two-factor covariance parameters by default.
This is a RISK-NEUTRAL PRICING IDENTITY DEMONSTRATION, NOT market calibration:
Treasury physical-measure covariance parameters are merely plugged into a Q
model. It is NOT a new independent fit if those parameters came from G2++.

Under Q and with two correlated OU shocks:
    df(t,T) = alpha(T-t) dt + sigma exp[-a(T-t)] dW1
                             + eta exp[-b(T-t)] dW2
    alpha(u) = .5 d/du S(u)
    S(u) = sigma^2 B_a(u)^2 + eta^2 B_b(u)^2
            + 2 rho sigma eta B_a(u) B_b(u)
    B_a(u) = (1-exp[-a u])/a.

The HJM no-arbitrage drift implies
    f(t,T) = f0(T) + 0.5[S(T)-S(T-t)]
                    + exp[-a(T-t)] x(t) + exp[-b(T-t)] z(t).
    r(t) = f0(t) + .5 S(t) + x(t)+z(t).

Under this model for all zero-bond maturities T:
    E_Q[exp(-integral_0^T r(s) ds)] = P(0,T),
which is verified in both exact integrated-Gaussian and independently
simulated OU-path Monte Carlo experiments.

IMPORTANT: This does not calibrate derivative-implied volatility, risk premia,
or prove the model fits traded options. One-curve Treasury demonstration only.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT = {'a':0.08, 'b':0.65, 'sigma':0.011, 'eta':0.016, 'rho':-0.55}


def bfun(a, t):
    t=np.asarray(t,dtype=float)
    return -np.expm1(-a*t)/a


def variance_forward_integral(t,p):
    u=np.asarray(t,dtype=float)
    ba=bfun(p['a'],u); bb=bfun(p['b'],u)
    return p['sigma']**2*ba**2+p['eta']**2*bb**2+2*p['rho']*p['sigma']*p['eta']*ba*bb


def no_arbitrage_alpha(t,p):
    u=np.asarray(t,dtype=float)
    ba=bfun(p['a'],u); bb=bfun(p['b'],u)
    ea=np.exp(-p['a']*u); eb=np.exp(-p['b']*u)
    return (p['sigma']**2*ba*ea + p['eta']**2*bb*eb
            + p['rho']*p['sigma']*p['eta']*(ba*eb + bb*ea))


def load_source(path):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find input: {path}')
    df=pd.read_csv(path,index_col=0,parse_dates=True)
    df.index=pd.to_datetime(df.index,errors='coerce')
    if df.index.isna().any():
        raise ValueError('Bad dates in Treasury file')
    df=df.sort_index()
    for i in range(1,31):
        if f'{i}Y' not in df.columns:
            raise ValueError(f'Missing required maturity {i}Y in {path.name}')
    selected=df[[f'{i}Y' for i in range(1,31)]].apply(pd.to_numeric,errors='coerce')
    return selected.replace([np.inf,-np.inf],np.nan).dropna(how='any')


def load_params(param_json, param_csv):
    if Path(param_json).is_file():
        src=Path(param_json)
        raw=json.loads(src.read_text())
        # Phase 5A's report stores params in a nested block; support both layouts.
        print(f'Found Phase 5A parameter file: {src}')
        if isinstance(raw.get('g2pp_parameters'),list) and len(raw['g2pp_parameters'])==5:
            return dict(zip(['a','b','sigma','eta','rho'],map(float,raw['g2pp_parameters']))), f'Phase 5A historical proxy ({src.name})'
        if isinstance(raw.get('two_factor'),dict): raw=raw['two_factor']
        if isinstance(raw.get('params'),dict): raw=raw['params']
        if isinstance(raw.get('g2pp'),dict): raw=raw['g2pp']
        names={'a':['a','a_per_year'], 'b':['b','b_per_year'],
               'sigma':['sigma','sigma_decimal_sqrt_year'],
               'eta':['eta','eta_decimal_sqrt_year'], 'rho':['rho']}
        p={}
        for k, opts in names.items():
            for option in opts:
                if option in raw:
                    p[k]=float(raw[option]);break
        if len(p)==5:
            return p,f'Phase 5A historical proxy ({src.name})'
        warnings.warn('Phase 5A JSON format not recognized; checking summary CSV.')
    if Path(param_csv).is_file():
        df=pd.read_csv(param_csv)
        row=df.loc[df['model'].eq('G2++')].iloc[0]
        p={'a':float(row.a_per_year),'b':float(row.b_per_year),
           'sigma':float(row.sigma_decimal_sqrt_year),
           'eta':float(row.eta_decimal_sqrt_year),'rho':float(row.rho)}
        return p,f'Phase 5A historical proxy ({Path(param_csv).name})'
    warnings.warn('No Phase 5A parameter files found: using synthetic illustration constants.')
    return DEFAULT.copy(),'SYNTHETIC illustrative constants (NOT market estimates)'


def check_params(p):
    if not all(np.isfinite(list(p.values()))):raise ValueError('Non-finite model parameter')
    if any(p[k]<=0 for k in ['a','b','sigma','eta']):raise ValueError('a,b,sigma,eta must be positive')
    if not -1 < p['rho'] < 1:raise ValueError('rho must be strictly between -1 and 1')


def create_initial_curve(zrow):
    mats=np.arange(1,31,dtype=float)
    yields=zrow.to_numpy(dtype=float)/100.
    if not np.all(np.isfinite(yields)) or np.max(np.abs(yields))>0.5:
        raise ValueError('Unexpected zero-curve yield units or extreme values')
    tenors=np.concatenate([[0.0],mats])
    logp=np.concatenate([[0.0],-mats*yields])
    interp=PchipInterpolator(tenors,logp,extrapolate=False)
    def logP(t):return interp(t)
    def f0(t):return -interp.derivative()(t)
    return mats,yields,logP,f0


def integrate_gaussian_variance(T,p):
    """Integral from 0 to T of S(u) du, 150-point Gauss-Legendre quadrature."""
    x,w=np.polynomial.legendre.leggauss(150)
    u=0.5*T*(x+1)
    return 0.5*T*np.dot(w,variance_forward_integral(u,p))


def monte_carlo_paths(T,logP,f0,p,n_paths=30000,steps_per_year=24,seed=12345,include_drift=True):
    """Monte Carlo integrates simulated OU short rates independently of exact-discount normal identity."""
    n_steps=int(math.ceil(T*steps_per_year));dt=T/n_steps
    ea=np.exp(-p['a']*dt);eb=np.exp(-p['b']*dt)
    varx=p['sigma']**2*(-np.expm1(-2*p['a']*dt))/(2*p['a'])
    varz=p['eta']**2*(-np.expm1(-2*p['b']*dt))/(2*p['b'])
    cov=p['rho']*p['sigma']*p['eta']*(-np.expm1(-(p['a']+p['b'])*dt))/(p['a']+p['b'])
    root=np.linalg.cholesky([[varx,cov],[cov,varz]])
    rng=np.random.default_rng(seed)
    x=np.zeros(n_paths);z=np.zeros(n_paths)
    deterministic=lambda s:f0(s)+(0.5*variance_forward_integral(s,p) if include_drift else 0.0)
    prev_r=float(deterministic(0))
    accumulator=np.zeros(n_paths)
    for j in range(n_steps):
        uu=rng.standard_normal((2,n_paths))
        next_x=ea*x+root[0,0]*uu[0]
        next_z=eb*z+root[1,0]*uu[0]+root[1,1]*uu[1]
        next_r=deterministic((j+1)*dt)+next_x+next_z
        accumulator +=0.5*dt*(prev_r+next_r)
        x,z=next_x,next_z
        prev_r=next_r
    values=np.exp(-accumulator)
    return float(values.mean()),float(values.std(ddof=1)/math.sqrt(n_paths)),n_steps


def main():
    parser=argparse.ArgumentParser(description='Two-factor Gaussian HJM pricing and martingale tests')
    parser.add_argument('--zero',default=str(HERE/'project2_phase1_outputs/daily_zero_yields_pct.csv'))
    parser.add_argument('--forward',default=str(HERE/'project2_phase1_outputs/daily_instantaneous_forwards_pct.csv'))
    parser.add_argument('--params-json',default=str(HERE/'project2_phase5a_outputs/phase5a_parameters.json'))
    parser.add_argument('--params-csv',default=str(HERE/'project2_phase5a_outputs/phase5a_covariance_fit_summary.csv'))
    parser.add_argument('--output',default=str(HERE/'project2_phase6a_outputs'))
    parser.add_argument('--paths',type=int,default=30000)
    parser.add_argument('--steps-per-year',type=int,default=24)
    parser.add_argument('--horizon',type=float,default=10.0)
    parser.add_argument('--forward-time',type=float,default=1.0)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if args.paths<1000 or args.steps_per_year<6:raise ValueError('Use >=1000 paths and >=6 steps per year')
    if not 0 < args.forward_time < 30:raise ValueError('forward-time must be in (0,30)')
    if not 0 < args.horizon<=30:raise ValueError('horizon must be in (0,30]')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    zero=load_source(args.zero);forward=load_source(args.forward)
    common=zero.index.intersection(forward.index)
    if len(common)==0:raise ValueError('No dates shared between zero and forward inputs')
    date=common.max();zrow=zero.loc[date];frow=forward.loc[date]
    mats,yields,logP,f0=create_initial_curve(zrow)
    p,provenance=load_params(args.params_json,args.params_csv);check_params(p)
    T=float(args.horizon)

    # HJM drift = .5 * S'(tau) is also g(tau)^T integral_0^tau g(v) dv.
    u=np.linspace(0.02,30,401)
    s_a=bfun(p['a'],u);s_b=bfun(p['b'],u)
    lhs=no_arbitrage_alpha(u,p)
    rhs=(p['sigma']*np.exp(-p['a']*u)*(p['sigma']*s_a+p['rho']*p['eta']*s_b)
        +p['eta']*np.sqrt(1-p['rho']**2)*np.exp(-p['b']*u)
              *p['eta']*np.sqrt(1-p['rho']**2)*s_b
        +p['rho']*p['eta']*np.exp(-p['b']*u)*(p['sigma']*s_a+p['rho']*p['eta']*s_b))
    # RHS above is g_1 * G_1 + g_2 * G_2 with two independent Brownian shocks.
    alpha_err=float(np.max(np.abs(lhs-rhs)))
    if alpha_err>1e-10:raise ArithmeticError(f'HJM drift restriction violated: {alpha_err}')

    price0=float(np.exp(logP(T)))
    V=integrate_gaussian_variance(T,p)
    # Integrated Gaussian Monte Carlo, independent seed from pathwise experiment.
    rng=np.random.default_rng(args.seed)
    samples=np.exp(logP(T)-0.5*V-rng.normal(0,np.sqrt(V),size=args.paths))
    exact_mc=float(samples.mean());exact_se=float(samples.std(ddof=1)/math.sqrt(args.paths))
    path_mc,path_se,nsteps=monte_carlo_paths(T,logP,f0,p,args.paths,args.steps_per_year,args.seed+1,True)
    no_drift_mc,no_drift_se,_=monte_carlo_paths(T,logP,f0,p,args.paths,args.steps_per_year,args.seed+2,False)
    wrong_theory=price0*math.exp(0.5*V)
    z_exact=(exact_mc-price0)/exact_se
    z_path=(path_mc-price0)/path_se
    z_no=(no_drift_mc-wrong_theory)/no_drift_se

    # Zero-curve knots reconstructed exactly (up to floating-point precision).
    kn_prices=np.exp(logP(mats));kn_market=np.exp(-mats*yields)
    init_err=float(np.max(np.abs(kn_prices-kn_market)))
    spline_forward=np.asarray(f0(mats))*100
    target_fwd=frow.to_numpy(dtype=float)
    fwd_rmse=float(np.sqrt(np.mean((spline_forward-target_fwd)**2))*100)
    # This is an interpolation diagnostic, not yield-curve market model fit.

    maturity_grid=np.arange(1,31,dtype=float)
    tstar=float(args.forward_time)
    valid_T=maturity_grid[maturity_grid>tstar+0.000001]
    # Gaussian forward curve at future date tstar. Simulate correlated OU factor pair.
    eaa=-np.expm1(-2*p['a']*tstar)/(2*p['a'])
    ebb=-np.expm1(-2*p['b']*tstar)/(2*p['b'])
    eab=-np.expm1(-(p['a']+p['b'])*tstar)/(p['a']+p['b'])
    cv=np.array([[p['sigma']**2*eaa,p['rho']*p['sigma']*p['eta']*eab],
                 [p['rho']*p['sigma']*p['eta']*eab,p['eta']**2*ebb]])
    state_rng=np.random.default_rng(args.seed+3)
    future_states=state_rng.multivariate_normal([0,0],cv,size=6000)
    future_mean=f0(valid_T)+0.5*(variance_forward_integral(valid_T,p)-variance_forward_integral(valid_T-tstar,p))
    future_samples=future_mean[None,:] +np.exp(-p['a']*(valid_T-tstar))[None,:]*future_states[:,0,None] +np.exp(-p['b']*(valid_T-tstar))[None,:]*future_states[:,1,None]
    p5=np.percentile(future_samples,5,axis=0);p95=np.percentile(future_samples,95,axis=0)
    p05=np.percentile(future_samples,50,axis=0)

    init_table=pd.DataFrame({'maturity_years':mats,'observed_zero_yield_pct':100*yields,
                 'P0T':kn_market, 'observed_instant_forward_pct':target_fwd,
                 'spline_implied_instant_forward_pct':spline_forward,
                 'forward_difference_bps':100*(spline_forward-target_fwd)})
    init_table.to_csv(out/'phase6a_initial_curve.csv',index=False)
    future_table=pd.DataFrame({'absolute_maturity_years':valid_T,
                 'initial_spline_forward_pct':100*f0(valid_T),
                 'hjm_mean_forward_at_future_time_pct':100*future_mean,
                 'hjm_forward_5pct':100*p5,'hjm_forward_median':100*p05,
                 'hjm_forward_95pct':100*p95})
    future_table.to_csv(out/'phase6a_future_forward_scenarios.csv',index=False)
    table=pd.DataFrame([
        {'method':'Analytical from initial discount curve','price_per_1':price0,'mc_standard_error':np.nan,'z_vs_analytic':0},
        {'method':'Exact integrated-Gaussian MC','price_per_1':exact_mc,'mc_standard_error':exact_se,'z_vs_analytic':z_exact},
        {'method':'Independent OU pathwise MC','price_per_1':path_mc,'mc_standard_error':path_se,'z_vs_analytic':z_path},
        {'method':'WRONG: MC without HJM drift','price_per_1':no_drift_mc,'mc_standard_error':no_drift_se,'z_vs_analytic':(no_drift_mc-price0)/no_drift_se},
        {'method':'WRONG: no-drift theoretical price','price_per_1':wrong_theory,'mc_standard_error':np.nan,'z_vs_analytic':np.nan}
    ])
    table.to_csv(out/'phase6a_bond_pricing_checks.csv',index=False)

    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(mats,100*yields,label='Latest Treasury zero yield',lw=2)
    ax.plot(mats,spline_forward,label='Forward from fitted zero-discount spline',lw=2)
    ax.plot(mats,target_fwd,label='Published GSW instantaneous forward',lw=1.5,linestyle='--')
    ax.set_title('Latest Treasury zero and instantaneous forward curves')
    ax.set_xlabel('Maturity (years)');ax.set_ylabel('Percent');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'01_initial_forward_and_zero_curves.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(u,1e4*lhs,label='HJM risk-neutral drift alpha(tau)',lw=2)
    ax.plot(u,1e4*rhs,ls='--',label='Volatility × integrated volatility',lw=1.5)
    ax.set_xlabel('Time to maturity (years)');ax.set_ylabel('Drift rate × 10,000')
    ax.set_title('HJM no-arbitrage drift identity (independent algebraic forms)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'02_hjm_no_arbitrage_drift.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(valid_T,100*f0(valid_T),lw=2,label='Current instantaneous forward curve')
    ax.plot(valid_T,100*future_mean,lw=2,label=f'Mean Q forward at t={tstar:g} years')
    ax.fill_between(valid_T,100*p5,100*p95,alpha=.23,label='Q 5–95% scenarios')
    ax.set_title(f'Two-factor HJM forward-rate scenarios at t={tstar:g} years')
    ax.set_xlabel('Absolute maturity T (years)');ax.set_ylabel('Forward rate (%)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'03_hjm_forward_curve_scenarios.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5))
    labels=['Initial curve','Gaussian MC','OU-path MC','No drift (WRONG)']
    vals=[price0,exact_mc,path_mc,no_drift_mc]
    ses=[0,exact_se,path_se,no_drift_se]
    ax.bar(labels,vals,yerr=ses,capsize=5)
    ax.axhline(price0,lw=1.5,ls='--',color='black',label='Analytic initial discount price')
    ax.set_title(f'{T:g}Y zero-bond Monte Carlo HJM drift test')
    ax.set_ylabel('Discount factor (price per $1 face)');ax.tick_params(axis='x',rotation=12)
    ax.legend();fig.tight_layout();fig.savefig(out/'04_discounted_bond_martingale_test.png',dpi=160);plt.close(fig)

    report=(
        '='*76+'\nPHASE 6A — TWO-FACTOR GAUSSIAN HJM NO-ARBITRAGE VALIDATION\n'+'='*76+'\n'
        f'Snapshot: {date.date()} | Historical nominal Treasury fitted yields, 1Y-30Y\n'
        f'Model parameter source: {provenance}\n'
        f'a={p["a"]:.6f}/yr, b={p["b"]:.6f}/yr, sigma={p["sigma"]:.6f}/sqrt(yr), '
        f'eta={p["eta"]:.6f}/sqrt(yr), rho={p["rho"]:.6f}\n'
        f'Initial discount factor knot max error: {init_err:.3e}\n'
        f'Published forwards vs discount-spline implied forwards RMSE: {fwd_rmse:.3f} bp (interpolation diagnostic)\n'
        f'HJM risk-neutral drift identity max absolute error: {alpha_err:.3e}\n'
        f'{T:g}Y initial curve P(0,T) = {price0:.8f}\n'
        f'{T:g}Y integrated-rate variance V(T) = {V:.8f}\n'
        f'{T:g}Y exact integrated-rate MC = {exact_mc:.8f}, SE={exact_se:.8f}, z={z_exact:+.3f}\n'
        f'{T:g}Y independent OU path MC = {path_mc:.8f}, SE={path_se:.8f}, z={z_path:+.3f}\n'
        f'{T:g}Y NO-DRIFT analytical WRONG price = {wrong_theory:.8f}, '
        f'no-drift OU MC = {no_drift_mc:.8f}, SE={no_drift_se:.8f}\n'
        f'No-drift MC vs its analytic prediction z = {z_no:+.3f}\n'
        f'Path integration: {nsteps} time steps, {args.paths} simulated paths\n'
        'LIMITATIONS: Prior Phase 5A HISTORICAL covariance is a proxy, NOT derivative-implied Q calibration.\n'
        'This exponential-volatility HJM is equivalent to a two-factor Gaussian G2++ representation.\n'
        'Its pricing tests are INTERNAL consistency tests, not validation against market option quotes.\n'
        'Interpolating 1Y-30Y zero yields to t=0 creates a short-end assumption.\n'
        'GSW forward and zero curves are published fitted estimates; spline forwards need not match exactly.\n'
        'Q future scenarios are risk-neutral distributions, NOT physical forecasts.\n'
        f'Output: {out.resolve()}\n'
    )
    print(report)
    (out/'phase6a_report.txt').write_text(report)
    (out/'phase6a_parameters.json').write_text(json.dumps({
        'parameters':p,'source':provenance,'snapshot':str(date.date()),
        'hjm_drift_identity_max_error':alpha_err,'initial_P_error':init_err,
        'exact_mc_z':z_exact,'path_mc_z':z_path,'no_drift_mc_z_vs_no_drift_analytic':z_no,
        'note':'G2++ parameter transfer is NOT independent calibration or option-implied Q estimation.'},indent=2))

if __name__=='__main__':main()
