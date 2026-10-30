#!/usr/bin/env python3
"""Project 2 / Phase 4A — Gaussian Arbitrage-Free Nelson–Siegel foundations.

This is an educational, reproducible TWO-STEP AFNS implementation, NOT a joint
maximum-likelihood / Kalman-filter fit, and NOT a derivative-volatility calibration.

Under the pricing measure Q:
    dX = -K_Q X dt + Sigma dW_Q
    K_Q = [[0, 0, 0], [0, lambda, -lambda], [0, 0, lambda]]
    r = X_level + X_slope,   Omega = Sigma Sigma.T
    P(t,T) = exp(-B(T-tau).T X_t + 0.5*Var_Q(integrated short rate))
where B(tau) = tau * (1, NS_slope(tau), NS_curvature(tau)) and
    Var_Q(integrated short rate) = integral_0^tau B(s).T Omega B(s) ds.

Yields in the CSV are in PERCENT, whereas all model math is in DECIMAL rates.
For a first exploratory estimate Omega is estimated from residuals of a monthly
historical P-measure VAR(1) fitted to yield-implied factors, assuming common
constant diffusion under P and Q. We iterate between the Q yield adjustment
and this P-derived covariance estimate. This is a restricted, two-step plug-in
estimate—not joint AFNS latent-state maximum likelihood. In particular, the
published GSW zero yields are themselves fitted Svensson estimates.

Run in the Phase 1 project directory:
    python3 project2_phase4a_afns_foundations.py

Dependencies: numpy, pandas, scipy, matplotlib.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import lstsq as scipy_lstsq
from scipy.integrate import trapezoid
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
TENORS=np.arange(1,31,dtype=float)
NAMES=[f'{k}Y' for k in range(1,31)]
DT=1/12


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing Phase 1 data file: {path}')
    z=pd.read_csv(path,index_col=0,parse_dates=[0])
    if not set(NAMES).issubset(z.columns):
        raise ValueError('Need zero-coupon yields 1Y to 30Y from Phase 1')
    z=z[NAMES].sort_index().astype(float)
    values=z.to_numpy(dtype=float)
    if z.index.has_duplicates or z.index.hasnans or not np.isfinite(values).all():
        raise ValueError('Missing/duplicate dates or non-finite values in yields')
    months=z.index.to_period('M')
    if not months.equals(pd.period_range(months.min(),months.max(),freq='M')):
        raise ValueError('Calendar months must be contiguous')
    if len(z)<48 or np.max(np.abs(values))>40:
        raise ValueError('Unexpected yield sample/units; input must be in PERCENT')
    return z,values/100  # math always uses decimal yields


def ns_loadings(tau,lam):
    tau=np.asarray(tau,dtype=float)
    x=lam*tau
    with np.errstate(divide='ignore',invalid='ignore'):
        slope=np.where(np.abs(x)<1e-7,1-x/2+x*x/6,-np.expm1(-x)/x)
    curve=slope-np.exp(-x)
    return np.column_stack([np.ones_like(tau),slope,curve])


def integrated_B(s,lam):
    s=np.asarray(s,dtype=float)
    return s[:,None]*ns_loadings(s,lam)


def integrated_variance(taus,lam,omega,nquad=60):
    """Compute V(tau)=int_0^tau B(s)'Omega B(s)ds in decimal-rate units.

    Gaussian OU factor law under Q gives the exact integrated-short-rate
    variance from this 1D deterministic quadrature; no simulation needed.
    """
    taus=np.atleast_1d(taus).astype(float)
    if np.any(taus<=0): raise ValueError('Tenors must be positive')
    q,w=leggauss(nquad)
    result=[]
    for tau in taus:
        s=tau*(q+1)/2
        b=integrated_B(s,lam)
        integrand=np.einsum('ni,ij,nj->n',b,omega,b,optimize=False)
        result.append(float((tau/2)*np.sum(w*integrand)))
    result=np.asarray(result)
    if not np.isfinite(result).all() or np.min(result)<-1e-10:
        raise FloatingPointError('Invalid integrated-rate variance')
    return result


def pricing_adjustment(taus,lam,omega):
    return -0.5*integrated_variance(taus,lam,omega)/np.asarray(taus)


def estimate_factors(y,load,adjust):
    if not np.isfinite(y).all() or not np.isfinite(adjust).all():
        raise FloatingPointError('Nonfinite model inputs')
    estimate,_,rank,_=scipy_lstsq(load,(y-adjust[None,:]).T,
                                  lapack_driver='gelsd',check_finite=True)
    if rank!=3: raise ValueError('Factor design matrix has insufficient rank')
    factors=estimate.T
    fitted=np.einsum('ij,mj->im',factors,load,optimize=False)+adjust[None,:]
    if not np.isfinite(factors).all() or not np.isfinite(fitted).all():
        raise FloatingPointError('Invalid model fit')
    if np.max(np.abs(factors))>1.0:
        raise FloatingPointError('AFNS factors exceed 100%: inspect units/calibration')
    return factors,fitted


def estimate_p_innovations(factors):
    """Fit unconstrained monthly real-world VAR(1), get diffusion plug-in.

    This uses Omega approx Var(VAR residual) / DT; the exact continuous-time
    OU transition covariance would require a separate P drift estimation step.
    """
    x=np.column_stack([np.ones(len(factors)-1),factors[:-1]])
    target=factors[1:]
    coef,_,rank,_=scipy_lstsq(x,target,lapack_driver='gelsd',check_finite=True)
    if rank!=4: raise ValueError('VAR(1) transition design is rank deficient')
    e=target - np.einsum('ij,jk->ik',x,coef,optimize=False)
    omega=(e.T@e)/((len(e)-4)*DT)
    omega=(omega+omega.T)/2
    evals,evecs=np.linalg.eigh(omega)
    if evals.min() < -1e-10: raise FloatingPointError('Innovation covariance is not PSD')
    omega=(evecs*np.maximum(evals,0))@evecs.T
    return omega,coef,e


def fit_two_step(y,lam,rounds=12):
    """Iterative plug-in volatility estimate, not joint model likelihood."""
    L=ns_loadings(TENORS,lam)
    zero=np.zeros(len(TENORS))
    factors,_=estimate_factors(y,L,zero)
    omega,_,_=estimate_p_innovations(factors)
    iteration=[]
    for step in range(rounds):
        adjust=pricing_adjustment(TENORS,lam,omega)
        factors,_=estimate_factors(y,L,adjust)
        new,_,_=estimate_p_innovations(factors)
        rel=float(np.linalg.norm(new-omega)/max(np.linalg.norm(omega),1e-12))
        iteration.append(rel)
        omega=new
        if rel<1e-8:
            break
    adjust=pricing_adjustment(TENORS,lam,omega)
    factors,fitted=estimate_factors(y,L,adjust)
    omega2,coef,eps=estimate_p_innovations(factors)
    # Re-check fixed point convergence.  Iterate until stable or refuse to claim so.
    relfinal=float(np.linalg.norm(omega2-omega)/max(np.linalg.norm(omega),1e-12))
    if relfinal>5e-5:
        raise RuntimeError(f'Two-step covariance update did not converge ({relfinal:.3g}); increase iterations')
    return dict(L=L,factors=factors,fit=fitted,adjust=adjust,omega=omega,
                coef=coef,innovations=eps,iter_rel=iteration,reldiff=relfinal)


def bond_price_and_risk(taus,factors,lam,omega):
    L=ns_loadings(taus,lam)
    adj=pricing_adjustment(taus,lam,omega)
    yields=L@factors+adj
    P=np.exp(-taus*yields)
    # A unit one-basis-point parallel shift to all zero yields changes P
    # by -P*tau*1e-4 to first order (these are zero-coupon bonds).
    modified_duration=taus.copy()
    dv01_per_100=100*P*taus*1e-4
    return pd.DataFrame(dict(maturity_years=taus,yield_pct=yields*100,
                              price_per_100_face=100*P,
                              modified_duration=modified_duration,
                              dv01_per_100_face_usd=dv01_per_100))


def analytic_checks(lam):
    # Sigma=0 => AFNS adjustment is zero.
    zz=pricing_adjustment(np.array([1.0,10.0,30.0]),lam,np.zeros((3,3)))
    assert np.max(np.abs(zz))<1e-15
    # Closed-form constant-level-Brownian test: Var= sigma^2*T^3/3,
    # resulting yield adjustment = -sigma^2*T^2/6.
    sigma=.004
    omega=np.diag([sigma*sigma,0,0])
    t=np.array([1.,10.,30.])
    assert np.allclose(pricing_adjustment(t,lam,omega),-sigma*sigma*t*t/6,rtol=1e-10)
    # NS B(s) derivative consistency with the Q-drift matrix exp(-K_Q s).
    from scipy.linalg import expm
    K=np.array([[0.,0.,0.],[0.,lam,-lam],[0.,0.,lam]])
    d=np.array([1.,1.,0.])
    for tau in (0.25,1.0,10.,30.):
        s=np.linspace(0,tau,2401)
        vals=np.array([expm(-K.T*v)@d for v in s])
        b_numerical=trapezoid(vals,s,axis=0)
        b_analytical=integrated_B(np.array([tau]),lam)[0]
        assert np.allclose(b_numerical,b_analytical,rtol=2e-5,atol=1e-6), (
            tau,b_numerical,b_analytical)


def mc_integrated_rate(x,lam,omega,tau=10.,n=70000,seed=27):
    # Gaussian exact-distribution Monte Carlo for INTEGRATED SHORT RATE;
    # not discretized simulated short-rate paths.
    B=integrated_B(np.array([tau]),lam)[0]
    mu=float(B@x)
    v=float(integrated_variance([tau],lam,omega)[0])
    theoretical=np.exp(-mu+0.5*v)
    rng=np.random.default_rng(seed)
    I=mu+np.sqrt(v)*rng.standard_normal(n)
    discounted=np.exp(-I)
    mc_price=float(discounted.mean())
    se=float(discounted.std(ddof=1)/np.sqrt(n))
    return dict(maturity_years=tau,paths=n,
                analytic_price_per_1=theoretical,mc_price_per_1=mc_price,
                mc_standard_error=se,z_distance=(mc_price-theoretical)/se if se>0 else 0.)


def charts(y,df,model,out,lam):
    latest=y[-1]*100
    reconstructed=model['fit'][-1]*100
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(TENORS,latest,marker='o',markersize=3,label='GSW zero yields')
    ax.plot(TENORS,reconstructed,label='Restricted AFNS two-step fit')
    ax.set(xlabel='Maturity (years)',ylabel='Continuously compounded yield (%)',
           title=f'Latest Treasury curve: {df.index[-1].date()}')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'01_latest_afns_curve.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(TENORS,model['adjust']*10000,marker='.',label='Risk-neutral AFNS adjustment')
    ax.axhline(0,color='k',lw=.7)
    ax.set(xlabel='Maturity (years)',ylabel='Yield adjustment (bps)',
           title='Gaussian AFNS no-arbitrage yield adjustment')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'02_no_arbitrage_adjustment.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8))
    names=['Level','Slope','Curvature']
    for j,name in enumerate(names):
        ax.plot(df.index,model['factors'][:,j]*100,label=name,linewidth=1.2)
    ax.set(xlabel='Date',ylabel='Estimated factor (percentage points)',
           title='Two-step AFNS factor history (not Kalman-smoothed)')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'03_afns_factors.png',dpi=160);plt.close(fig)
    err=np.sqrt(np.mean((10000*(model['fit']-y))**2,axis=1))
    fig,ax=plt.subplots(figsize=(10,4.8))
    ax.plot(df.index,err)
    ax.set(xlabel='Date',ylabel='Curve RMSE (bps)',
           title='Two-step AFNS cross-sectional curve fitting error')
    ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'04_afns_fit_rmse.png',dpi=160);plt.close(fig)
    return err


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'project2_phase1_outputs'/'monthly_zero_yields_pct.csv')
    ap.add_argument('--output',type=Path,default=ROOT/'project2_phase4a_outputs')
    ap.add_argument('--lambda-per-year',type=float,default=.25)
    ap.add_argument('--no-monte-carlo',action='store_true')
    a=ap.parse_args()
    if not .05<=a.lambda_per_year<=3.0: raise ValueError('lambda outside supported range [0.05,3]')
    analytic_checks(a.lambda_per_year)
    df,y=load_data(a.input.expanduser())
    out=a.output.expanduser();out.mkdir(parents=True,exist_ok=True)
    model=fit_two_step(y,a.lambda_per_year)
    omega=model['omega']
    eigvals=np.linalg.eigvalsh(omega)
    if eigvals.min() < -1e-12: raise FloatingPointError('Negative covariance eigenvalue')
    err=charts(y,df,model,out,a.lambda_per_year)
    states=pd.DataFrame(model['factors']*100,index=df.index,
                        columns=['level_pct','slope_pct','curvature_pct'])
    states.index.name='Date'
    states.to_csv(out/'phase4a_afns_factors_monthly.csv')
    compare=pd.DataFrame({'maturity_years':TENORS,'observed_zero_yield_pct':y[-1]*100,
        'model_yield_pct':model['fit'][-1]*100,
        'error_bp':(model['fit'][-1]-y[-1])*10000,
        'yield_adjustment_bp':model['adjust']*10000})
    compare.to_csv(out/'phase4a_latest_curve.csv',index=False)
    pd.DataFrame({'Date':df.index,'curve_rmse_bp':err}).to_csv(out/'phase4a_monthly_rmse.csv',index=False)
    zero_bond=bond_price_and_risk(np.array([2.,5.,10.,20.,30.]),model['factors'][-1],a.lambda_per_year,omega)
    zero_bond.to_csv(out/'phase4a_zero_coupon_bond_risk.csv',index=False)
    mc=(None if a.no_monte_carlo else
        mc_integrated_rate(model['factors'][-1],a.lambda_per_year,omega))
    if mc is not None:
        pd.DataFrame([mc]).to_csv(out/'phase4a_mc_integrated_rate_check.csv',index=False)
    Kq=[[0.,0.,0.],[0.,a.lambda_per_year,-a.lambda_per_year],[0.,0.,a.lambda_per_year]]
    info=dict(model='Gaussian 3-factor AFNS; two-step historical volatility plug-in',
              lambda_per_year=a.lambda_per_year,
              first_month=str(df.index[0].date()),last_month=str(df.index[-1].date()),
              observations=len(df),
              yield_rmse_mean_bp=float(err.mean()),yield_rmse_median_bp=float(np.median(err)),
              yield_rmse_latest_bp=float(err[-1]),
              risk_neutral_Kq_per_year=Kq,
              p_var1_intercept_and_coefficients=model['coef'].tolist(),
              omega_decimal_rate_squared_per_year=omega.tolist(),
              annual_diffusion_vol_in_percentage_points=(100*np.sqrt(np.maximum(0,np.diag(omega)))).tolist(),
              final_omega_relative_change=model['reldiff'],
              yield_adjustment_10year_bp=float(model['adjust'][9]*10000),
              yield_adjustment_30year_bp=float(model['adjust'][29]*10000),
              exact_gaussian_integrated_rate_mc=mc,
              estimation_caveats=[
                  'This is an exploratory two-step plug-in AFNS exercise, not Kalman-filter maximum-likelihood estimation.',
                  'P-VAR innovation covariance divided by dt only approximates the continuous-time diffusion matrix.',
                  'No market swaption prices or risk-neutral implied volatilities are used.',
                  'No forecast comparison or statistical significance test is performed here.',
                  'GSW Treasury zero yields are Svensson-fitted estimates, not raw bond trade prices.'])
    (out/'phase4a_methodology.json').write_text(json.dumps(info,indent=2))
    print('\n'+'='*76)
    print('PHASE 4A — GAUSSIAN AFNS NO-ARBITRAGE PRICING FOUNDATION')
    print('='*76)
    print('Period:',df.index[0].date(),'to',df.index[-1].date(),f'({len(df)} monthly curves)')
    print('Q measure: 3-factor Gaussian AFNS, K_Q constrained by lambda=',a.lambda_per_year)
    print('Estimation: two-step historical-volatility plug-in (NOT full Kalman ML)')
    print('Data: continuously compounded yields; input in percent, math in decimals')
    print('Mean AFNS curve fit RMSE:',f'{err.mean():.3f} bps')
    print('Latest AFNS curve fit RMSE:',f'{err[-1]:.3f} bps')
    print('Q adjustment to 10-year yield:',f'{model["adjust"][9]*10000:.3f} bps')
    print('Q adjustment to 30-year yield:',f'{model["adjust"][29]*10000:.3f} bps')
    print('Omega eigenvalues:',np.array2string(eigvals,precision=9))
    print('Maximum covariance iteration relative change:',f'{model["reldiff"]:.2e}')
    if mc is not None:
        print('10Y analytic Q zero-bond price per $1:',f'{mc["analytic_price_per_1"]:.7f}')
        print('10Y exact integrated-rate MC price:',f'{mc["mc_price_per_1"]:.7f}')
        print('MC standard error:',f'{mc["mc_standard_error"]:.7f}',
              '| z distance:',f'{mc["z_distance"]:.3f}')
    print('\nLatest model-implied zero-coupon bond analytics:')
    print(zero_bond.to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nSaved outputs to:',out)
    print('NEXT: Phase 4B — joint AFNS state-space / Kalman likelihood and rolling forecasts')

if __name__=='__main__': main()
