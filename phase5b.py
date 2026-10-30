#!/usr/bin/env python3
"""Phase 5B — G2++ Gaussian two-factor calibration to *ATM normal* swaption quotes.

MODE 1 (default): SYNTHETIC VALIDATION. Reads latest Phase-1 Treasury zero curve to
construct a demonstration discount curve and generates explicitly SYNTHETIC ATM
payer swaption volatilities from known G2++ parameters. Not market calibration.

MODE 2 (real quotes): requires BOTH --curve-csv and --swaption-csv. They must be
same-day SOFR OIS discount factors and ATM normal swaption quotes. This is an
idealized single-curve, cashflow-equivalent physically settled European swaption;
ignores cash settlement, exact daycounts, conventions, and convexity adjustments.

Formats:
  curve.csv: maturity_years,discount_factor   [0 optional, P(0)=1]
    or maturity_years,zero_rate_pct [continuous, percent]
  swaption.csv: expiry_years,tenor_years,normal_vol_bps
    [ATM, annual fixed payments assumed; bps/sqrt(year)]

G2++ under Q: dx=-a*x dt+sigma dW1; dz=-b*z dt+eta dW2, rho=corr.
P(S,T) conditional on x_S,z_S is exponential-affine, with exact initial
curve matching from phi(t). Prices are discounted T=S forward-measure
expectations of positive-part swap PV, integrated by 1-D Gauss-Hermite plus
conditional truncated Gaussian formula (not an Euler Monte Carlo fit).

No swaption market data is bundled or invented. Demo prices are SYNTHETIC.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, brentq
from scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SQRT2PI=math.sqrt(2*math.pi)
GH_NODES, GH_WEIGHTS = hermgauss(24)
GH_WEIGHTS=GH_WEIGHTS/math.sqrt(math.pi)
GL_NODES,GL_WEIGHTS=leggauss(24)

def B(k,t):
    t=np.asarray(t,dtype=float)
    return -np.expm1(-k*t)/k

def Z(k,t):
    return -np.expm1(-k*t)/k

def I(k, j, t):
    # integral_0^t exp(-k*u) B(j,u) du
    return (Z(k,t)-Z(k+j,t))/j

def V(t,p):
    a,b,sig,eta,rho=p
    t=np.asarray(t,dtype=float)
    s=t[...,None]*(GL_NODES+1)/2
    w=t[...,None]*GL_WEIGHTS/2
    ba=B(a,s);bb=B(b,s)
    return np.sum(w*(sig**2*ba**2+eta**2*bb**2+2*rho*sig*eta*ba*bb),axis=-1)

class Curve:
    def __init__(self,years,discounts):
        years=np.array(years,dtype=float); discounts=np.array(discounts,dtype=float)
        order=np.argsort(years); years=years[order]; discounts=discounts[order]
        if len(years)<8 or np.any(np.diff(years)<1e-9) or not np.all(np.isfinite(discounts)):
            raise ValueError('Curve needs >=8 distinct finite maturities.')
        if years[0]>0:
            years=np.r_[0.,years]; discounts=np.r_[1.,discounts]
        if abs(years[0])>1e-9 or abs(discounts[0]-1.)>1e-7:
            raise ValueError('Curve must have P(0,0)=1.')
        if (discounts<=0).any() or (discounts>1.5).any():
            raise ValueError('Discount factors must be positive and plausible.')
        self.t=years; self.logp=np.log(discounts)
    def __call__(self,t):
        t=np.asarray(t,dtype=float)
        if np.any(t<0) or np.any(t>self.t[-1]+1e-10):
            raise ValueError(f'Requested maturity outside discount-curve support [0,{self.t[-1]}].')
        return np.exp(np.interp(t,self.t,self.logp))


def read_curve(path):
    d=pd.read_csv(path)
    if 'maturity_years' not in d.columns:
        raise ValueError('Curve CSV requires maturity_years.')
    t=np.asarray(d['maturity_years'],float)
    if 'discount_factor' in d:
        disc=np.asarray(d['discount_factor'],float)
    elif 'zero_rate_pct' in d:
        disc=np.exp(-t*np.asarray(d['zero_rate_pct'],float)/100.)
    else:
        raise ValueError('Curve CSV requires discount_factor or zero_rate_pct.')
    return Curve(t,disc)


def demo_curve(path):
    d=pd.read_csv(path,index_col=0,parse_dates=True)
    d=d[[f'{i}Y' for i in range(1,31)]].dropna()
    if d.empty: raise ValueError('No complete curves in Phase 1 CSV')
    latest=d.iloc[-1]
    t=np.arange(1,31,dtype=float)
    return Curve(t,np.exp(-t*latest.values.astype(float)/100.)),str(d.index[-1].date())


def swap_cashflows(curve,S,N):
    # Idealized ATM forward-starting OIS swap: annual fixed coupons, with
    # discount and floating projections from same SOFR OIS curve.
    if abs(round(N)-N)>1e-9 or N<1 or N>30 or S<=0:
        raise ValueError('tenor_years must be integer 1..30; expiry >0.')
    t=S+np.arange(1,round(N)+1,dtype=float)
    dfs=curve(t)
    annuity=float(np.sum(dfs))
    k=float((curve(S)-dfs[-1])/annuity)
    coupons=np.full(len(t),k); coupons[-1]+=1.
    return t,k,annuity,coupons


def moments_under_S(S,p):
    a,b,sig,eta,rho=p
    mx= -sig**2*I(a,a,S)-rho*sig*eta*I(a,b,S)
    mz= -eta**2*I(b,b,S)-rho*sig*eta*I(b,a,S)
    vx=sig**2*Z(2*a,S); vz=eta**2*Z(2*b,S)
    cov=rho*sig*eta*Z(a+b,S)
    return mx,mz,vx,vz,cov


def root_z(A,bb,coeff):
    # Sum coeff*A*exp(-bb*z) monotonically decreases from inf to 0.
    def f(z):
        exponent=np.clip(-bb*z,-700,700)
        return float(np.dot(coeff*A,np.exp(exponent))-1.)
    lo,hi=-1.,1.
    while f(lo)<0.: lo*=2
    while f(hi)>0.: hi*=2
    return brentq(f,lo,hi,xtol=1e-11)


def payer_price(curve,S,N,p,gh_nodes=GH_NODES,gh_weights=GH_WEIGHTS):
    a,b,sig,eta,rho=p
    t,k,ann,coupons=swap_cashflows(curve,S,N)
    forward=curve(t)/curve(S)
    dt=t-S
    ba=B(a,dt);bb=B(b,dt)
    correct=.5*(V(dt,p)-V(t,p)+V(S,p))
    F=forward*np.exp(correct)
    mx,mz,vx,vz,cov=moments_under_S(S,p)
    sx=math.sqrt(vx); varcond=max(vz-cov*cov/vx,1e-22)
    sc=math.sqrt(varcond)
    xvals=mx+math.sqrt(2)*sx*gh_nodes
    out=0.
    for xx,w in zip(xvals,gh_weights):
        aa=F*np.exp(-ba*xx)
        zm=mz+cov/vx*(xx-mx)
        zz=root_z(aa,bb,coupons)
        # E[ max(1-sum_i c_i P_i,0) | x ] =
        # Prob(z>z*) - sum c_i E[P_i 1(z>z*) | x].
        prob=ndtr((zm-zz)/sc)
        truncated=np.exp(-bb*zm+.5*bb*bb*varcond)*ndtr((zm-bb*varcond-zz)/sc)
        cond=prob-np.dot(coupons*aa,truncated)
        out+=w*max(0.,float(cond))
    return float(curve(S)*out),k,ann


def equivalent_normal_vol_bps(price,S,ann):
    return float(price*SQRT2PI/(ann*math.sqrt(S))*1e4)


def synth_quotes(curve,true_params):
    grid=[(1,2),(1,5),(1,10),(1,20),
          (2,2),(2,5),(2,10),(2,20),
          (3,2),(3,5),(3,10),(3,20),
          (5,2),(5,5),(5,10),(5,20),
          (7,2),(7,5),(7,10),(7,20),
          (10,2),(10,5),(10,10),(10,20)]
    rows=[]
    for s,n in grid:
        if s+n>curve.t[-1]: continue
        price,k,ann=payer_price(curve,s,n,true_params)
        rows.append({'expiry_years':float(s),'tenor_years':int(n),
                     'normal_vol_bps':equivalent_normal_vol_bps(price,s,ann)})
    return pd.DataFrame(rows)


def precompute_targets(curve,quotes):
    req={'expiry_years','tenor_years','normal_vol_bps'}
    if not req.issubset(quotes):raise ValueError(f'Swaption quotes need {sorted(req)}')
    d=quotes[list(sorted(req))].copy().astype(float)
    if d.isnull().any().any() or (d['normal_vol_bps']<=0).any():
        raise ValueError('Swaptions must have finite positive vols (bps/sqrt(year)).')
    if d.duplicated(['expiry_years','tenor_years']).any():
        raise ValueError('Duplicate option expiry/tenor quotes.')
    if len(d)<12:raise ValueError('At least 12 expiry/tenor quotes required for robust validation.')
    d['price_per_1']=0.
    d['atm_swap_rate_pct']=0.
    for ix,r in d.iterrows():
        s=float(r.expiry_years);n=float(r.tenor_years)
        _,k,ann,cs=swap_cashflows(curve,s,n)
        d.loc[ix,'atm_swap_rate_pct']=100*k
        d.loc[ix,'price_per_1']=float(r.normal_vol_bps)*1e-4*math.sqrt(s)*ann/SQRT2PI
    # deterministic checkerboard holdout, across both expiries and tenors
    d=d.sort_values(['expiry_years','tenor_years']).reset_index(drop=True)
    d['partition']=['holdout' if i%4==0 else 'train' for i in range(len(d))]
    if (d['partition']=='train').sum()<8 or (d['partition']=='holdout').sum()<3:
        raise ValueError('Not enough train and holdout quotes.')
    return d


def calibrate(curve,d,initial=None):
    train=d.query('partition == "train"')
    if initial is None:initial=np.array([0.08,0.6,0.012,0.016,-0.5])
    lo=[0.012,0.10,0.0007,0.0007,-.95]
    hi=[1.0,3.0,.06,.06,.95]
    # Weighted by implied volatility (bps) rather than premium.
    # sigma/b, a/sigma interactions may not be uniquely identified.
    def resid(par):
        arr=[]
        for r in train.itertuples():
            val,_,ann=payer_price(curve,r.expiry_years,r.tenor_years,par)
            pred=equivalent_normal_vol_bps(val,r.expiry_years,ann)
            arr.append((pred-r.normal_vol_bps)/100.)
        return np.asarray(arr)
    guesses=[np.clip(initial, np.array(lo)+1e-4,np.array(hi)-1e-4),
             np.array([.15,1.05,.015,.01,-.3])]
    best=None
    for guess in guesses:
        opt=least_squares(resid,guess,bounds=(lo,hi),max_nfev=220,
                          xtol=1e-9,ftol=1e-9,gtol=1e-9)
        if best is None or np.linalg.norm(opt.fun)<np.linalg.norm(best.fun):
            best=opt
    if not best.success:
        raise RuntimeError('Optimization failed: '+str(best.message))
    return best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--curve-csv',type=str,default=None,help='Same-date SOFR OIS discount curve, not a Treasury curve')
    ap.add_argument('--swaption-csv',type=str,default=None,help='Same-date ATM SOFR European normal volatility quotes')
    ap.add_argument('--phase1-csv',type=str,default=str(HERE/'project2_phase1_outputs'/'daily_zero_yields_pct.csv'))
    ap.add_argument('--output',type=str,default=str(HERE/'project2_phase5b_outputs'))
    args=ap.parse_args()
    if bool(args.curve_csv)!=bool(args.swaption_csv):
        raise SystemExit('Real calibration requires BOTH --curve-csv AND --swaption-csv. Never mix SOFR vols with a Treasury curve.')
    demo=not args.curve_csv
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if demo:
        curve,curve_date=demo_curve(args.phase1_csv)
        true=np.array([.08,.65,.011,.016,-.55])
        quotes=synth_quotes(curve,true)
        quotes.to_csv(out/'synthetic_quotes_NOT_MARKET_DATA.csv',index=False)
        label='SYNTHETIC demonstration ONLY (Treasury curve '+curve_date+')'
        print('\n*** NO REAL SWAPTION QUOTES SUPPLIED: SYNTHETIC DEMONSTRATION ONLY ***\n')
    else:
        curve=read_curve(args.curve_csv)
        quotes=pd.read_csv(args.swaption_csv)
        label='USER-SUPPLIED SOFR curve and ATM swaption quotes (dates must match)'
    d=precompute_targets(curve,quotes)
    print('Calibrating 5 G2++ parameters from ATM option prices...')
    opt=calibrate(curve,d)
    pars=opt.x
    results=[]
    for r in d.itertuples():
        model_price,k,ann=payer_price(curve,r.expiry_years,r.tenor_years,pars)
        vol=equivalent_normal_vol_bps(model_price,r.expiry_years,ann)
        results.append({'expiry_years':r.expiry_years,'tenor_years':r.tenor_years,
                        'partition':r.partition,'quote_normal_vol_bps':r.normal_vol_bps,
                        'model_normal_vol_bps':vol,'vol_error_bps':vol-r.normal_vol_bps,
                        'market_equivalent_price_per_1':r.price_per_1,
                        'model_price_per_1':model_price,'atm_swap_rate_pct':100*k})
    r=pd.DataFrame(results)
    def errors(part):
        x=r.query('partition == @part'); e=x['vol_error_bps'].to_numpy()
        return float(np.sqrt(np.mean(e*e))),float(np.mean(np.abs(e)))
    train,train_mae=errors('train'); hold,hold_mae=errors('holdout')
    # Risk-neutral test: forward-measure bond martingale identity for each
    # expiry and selected maturities, exact for G2++ Gaussian moments.
    max_martingale_err=0.
    for s,n in [(1,5),(2,10),(5,20)]:
        t=s+n
        a,b,sig,eta,rho=pars
        mx,mz,vx,vz,cov=moments_under_S(s,pars)
        ba=B(a,n);bb=B(b,n)
        log_multiplier=.5*(V(n,pars)-V(t,pars)+V(s,pars))
        log_expect=log_multiplier-ba*mx-bb*mz+.5*(ba*ba*vx+bb*bb*vz+2*ba*bb*cov)
        max_martingale_err=max(max_martingale_err,abs(math.expm1(float(log_expect))))
    r.to_csv(out/'phase5b_swaption_fit.csv',index=False)
    summary={'data_label':label,'synthetic_demo':demo,'parameters':{
        'a_per_year':pars[0],'b_per_year':pars[1],
        'sigma_per_sqrt_year':pars[2],'eta_per_sqrt_year':pars[3],
        'rho':pars[4]},'optimizer_success':bool(opt.success),
        'training_quote_count':int((r.partition=='train').sum()),
        'holdout_quote_count':int((r.partition=='holdout').sum()),
        'train_normal_vol_rmse_bps':train,
        'holdout_normal_vol_rmse_bps':hold,
        'max_forward_measure_martingale_relative_error':max_martingale_err}
    if demo:summary['synthetic_true_parameters']={'a':true[0],'b':true[1],'sigma':true[2],'eta':true[3],'rho':true[4]}
    (out/'phase5b_calibration_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(10,5))
    for split,color in [('train','tab:blue'),('holdout','tab:orange')]:
        part=r[r.partition==split]
        ax.scatter(part.quote_normal_vol_bps,part.model_normal_vol_bps,c=color,label=split,s=50)
    allv=np.r_[r.quote_normal_vol_bps,r.model_normal_vol_bps]
    ax.plot([min(allv),max(allv)],[min(allv),max(allv)],'k--',lw=1)
    ax.set(xlabel='Quoted ATM normal volatility (bp/√yr)',ylabel='G2++ implied ATM normal volatility (bp/√yr)',title='G2++ Swaption Calibration'+(' — SYNTHETIC DATA' if demo else ''))
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'phase5b_swaption_fit.png',dpi=180);plt.close(fig)
    print('='*75)
    print('PHASE 5B — G2++ SWAPTION VOL CALIBRATION: '+label)
    print('Optimizer success:',opt.success,'| evaluations:',opt.nfev)
    print('a={:.5f}, b={:.5f}, sigma={:.6f}, eta={:.6f}, rho={:+.5f}'.format(*pars))
    print('Train quotes:',sum(r.partition=='train'),'| holdout quotes:',sum(r.partition=='holdout'))
    print('Train ATM normal-vol RMSE: {:.4f} bp/√yr'.format(train))
    print('Holdout ATM normal-vol RMSE: {:.4f} bp/√yr'.format(hold))
    print('Forward-measure bond price identity max rel error: {:.3e}'.format(max_martingale_err))
    if demo: print('SYNTHETIC TARGET PARAMS: a={:.3f}, b={:.3f}, sigma={:.4f}, eta={:.4f}, rho={:.2f}'.format(*true))
    print('IMPORTANT: ATM-only, simplified one-curve OIS swaption approximation. Not a production dealer pricer.')
    print('Pricing curve and quote dates must match. No historical-Treasury-vol cross-calibration.')
    print('Output:',out)

if __name__=='__main__':
    main()
