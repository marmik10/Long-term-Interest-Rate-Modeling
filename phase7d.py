"""Project 2 — Phase 7D: capital-constrained CVaR & stress-aware Treasury hedging.

Python 3.9+, numpy/pandas/scipy/matplotlib. Uses actual Phase 1 GSW fitted zero
rates, but HYPOTHETICAL constant-tenor zero-coupon positions and assumed costs.

Out-of-sample decisions use only prices and shocks known BEFORE each test day.
Historical CVaR is the empirical 95% expected shortfall of exact repricing
scenarios in the PRIOR 504 trading-day window; it is not a Q-measure option price.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'project2_phase1_outputs' / 'daily_zero_yields_pct.csv'
DEFAULT_OUTPUT = ROOT / 'project2_phase7d_outputs'
TENORS = np.array([2., 5., 10., 20., 30.])
COLS = ['2Y','5Y','10Y','20Y','30Y']
BASE = np.array([10., 20., 30., 20., 20.])  # market value $ millions
HI = np.array([0,2,4])
HT = TENORS[HI]
OOS_START = pd.Timestamp('2018-01-01')
EWMA_DECAY = 0.94
TRAIN_HISTORY = 504
ALPHA = 0.95
TRADE_BP = 1.0
TURNOVER_PENALTY_BP = 5.0
ANNUAL_CAPITAL_CHARGE = 0.01
STRESS_LOSS_LIMIT_M = 3.0

STRESSES = {
    'Parallel +100bp':np.array([100.,100.,100.,100.,100.]),
    'Parallel -100bp':np.array([-100.,-100.,-100.,-100.,-100.]),
    'Bear steepener':np.array([20.,40.,60.,80.,100.]),
    'Bear flattener':np.array([100.,80.,60.,40.,20.]),
    'Long end +100bp':np.array([0.,0.,0.,100.,100.]),
    'Twist':np.array([-50.,-25.,0.,50.,100.]),
}


def read_data(path):
    if not path.exists():
        raise FileNotFoundError('Missing REAL Phase 1 CSV: %s' % path)
    d = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    d = d.loc[~d.index.duplicated(keep='last')]
    absent = [c for c in COLS if c not in d.columns]
    if absent: raise ValueError('Missing maturities: %s' % absent)
    d = d[COLS].apply(pd.to_numeric, errors='coerce').dropna() / 100.
    if len(d) < 700 or not np.isfinite(d.values).all() or d.abs().max().max() > 0.25:
        raise ValueError('Bad or insufficient rates; expected continuously compounded PERCENT inputs')
    return d


def factor(shocks_bp):
    """Exact change in zero price per $1 market value at the prior close."""
    return np.expm1(-np.asarray(shocks_bp, dtype=float)*TENORS*1e-4)


def pnl(mv_m, shock_bp):
    return float(np.dot(mv_m, factor(shock_bp)))


def worst_stress(mv_m):
    return max(-pnl(mv_m, s) for s in STRESSES.values())


def ewma_update(c, shock):
    return EWMA_DECAY*c + (1.-EWMA_DECAY)*np.outer(shock,shock)


def scheduled_rebalance(dt, prev_dt, freq):
    if prev_dt is None: return True
    if freq == 'daily': return True
    if freq == 'weekly': return dt.to_period('W-SUN') != prev_dt.to_period('W-SUN')
    return dt.to_period('M') != prev_dt.to_period('M')


def solve_minvar(cov, cap, prior):
    """Phase 7C-style turnover-aware EWMA minimum-variance benchmark."""
    limit = cap - 100.
    eq = BASE @ TENORS
    z0 = np.array([0., 0., -eq/30.])
    x0 = np.r_[z0, np.abs(z0-prior)]
    from scipy.optimize import LinearConstraint
    aeq = np.zeros((1,6)); aeq[0,:3] = HT/30.
    con = [LinearConstraint(aeq,[-eq/30.],[-eq/30.])]
    signs = np.array([[p,q,r] for p in (-1,1) for q in (-1,1) for r in (-1,1)])
    aa = np.zeros((8,6));aa[:,:3]=signs
    con.append(LinearConstraint(aa,-np.inf,limit))
    a = np.zeros((6,6))
    for j in range(3):
        a[2*j,j]=1.;a[2*j,3+j]=-1.
        a[2*j+1,j]=-1.;a[2*j+1,3+j]=-1.
    con.append(LinearConstraint(a,-np.inf,np.array([prior[0],-prior[0],prior[1],-prior[1],prior[2],-prior[2]])))
    cov = (cov+cov.T)/2 + np.eye(5)*max(np.trace(cov)/5,.01)*1e-7
    f = HT*1e-4
    d0 = BASE*TENORS*1e-4
    def objective(z):
        d=d0.copy();d[HI]+=z[:3]*f
        risk=0.5*d@cov@d
        g=np.zeros(6);g[:3]=(cov@d)[HI]*f
        risk += 0.10*TURNOVER_PENALTY_BP*1e-4*sum(z[3:])
        g[3:]=0.10*TURNOVER_PENALTY_BP*1e-4
        return risk*1000.,g*1000.
    result=minimize(objective,x0,method='SLSQP',jac=True, constraints=con,
                    bounds=[(-limit,limit)]*3+[(0,2*limit+100)]*3,
                    options={'maxiter':200,'ftol':1e-8})
    x=result.x[:3]
    validate(x,cap)
    if not result.success:
        raise RuntimeError('EWMA minimum variance optimization failed: %s' % result.message)
    return x


def validate(x,cap):
    if not np.isfinite(x).all(): raise RuntimeError('Nonfinite hedge weights')
    if abs(HT@x + BASE@TENORS)>1e-4: raise RuntimeError('DV01 neutrality not met')
    if 100+np.abs(x).sum()>cap+1e-4: raise RuntimeError('Gross exposure exceeded cap')


def solve_cvar(shocks_bp, cap, previous, stress_aware=False, alpha=ALPHA):
    """Linear program, exact zero-price shocks, gross cap + turnover penalty.

    Decision variables x[3], abs(x)[3], abs(x-prev)[3], VaR cutoff t,
    and CVaR auxiliary scenario losses z[n]. All dollars are in millions.
    """
    n = len(shocks_bp)
    if n < 100: raise ValueError('Need >=100 prior historical shocks')
    limit=cap-100.
    rets=factor(shocks_bp)   # n x 5 matrix
    base_pnl = np.sum(rets * BASE[np.newaxis, :], axis=1)
    hr=rets[:,HI]
    k=10+n
    x=slice(0,3);u=slice(3,6);v=slice(6,9);t=9;z=slice(10,k)
    obj=np.zeros(k)
    obj[t]=1.
    obj[z]=1./((1.-alpha)*n)
    obj[v]=TURNOVER_PENALTY_BP*1e-4
    # Historical scenario loss <= VaR cutoff + nonnegative positive-part slack
    a=np.zeros((n+13+(len(STRESSES) if stress_aware else 0),k))
    b=np.zeros(a.shape[0])
    a[:n,x]=-hr; a[:n,t]=-1.;a[:n,z]=-np.eye(n)
    b[:n]=base_pnl
    j=n
    # abs hedge exposure
    for i in range(3):
        a[j,i]=1; a[j,3+i]=-1; j+=1
        a[j,i]=-1; a[j,3+i]=-1; j+=1
    # abs turnover
    for i in range(3):
        a[j,i]=1; a[j,6+i]=-1; b[j]=previous[i];j+=1
        a[j,i]=-1;a[j,6+i]=-1;b[j]=-previous[i];j+=1
    a[j,u]=1.;b[j]=limit;j+=1
    # hard maximum loss of $3m under each of six parallel/nonparallel stresses
    if stress_aware:
        for s in STRESSES.values():
            r=factor(s)
            a[j,x]=-r[HI]
            b[j]=STRESS_LOSS_LIMIT_M + BASE@r
            j+=1
    assert j==len(b)
    ae=np.zeros((1,k));ae[0,x]=HT/30.
    be=np.array([-BASE@TENORS/30.])
    bounds=[(None,None)]*3+[(0,None)]*6+[(None,None)]+[(0,None)]*n
    result=linprog(obj,A_ub=a,b_ub=b,A_eq=ae,b_eq=be,bounds=bounds,method='highs')
    if not result.success:
        raise RuntimeError('CVaR LP failed (possibly infeasible stress limit/cap): %s' % result.message)
    solution=result.x[x]
    validate(solution,cap)
    if stress_aware:
        pos=BASE.copy(); pos[HI]+=solution
        loss=worst_stress(pos)
        if loss>STRESS_LOSS_LIMIT_M+1e-5:
            raise RuntimeError('Stress-loss constraint failed: %.6f' % loss)
    return solution


def max_drawdown(pnl_m):
    cs=np.r_[0., np.cumsum(pnl_m)]
    return float(np.min(cs-np.maximum.accumulate(cs)))


def historical_es(net_pnl_m,confidence=.95):
    losses=-np.asarray(net_pnl_m,dtype=float)
    threshold=np.quantile(losses,confidence)
    return float(np.mean(losses[losses>=threshold]))


def run(data_path, output, frequency='weekly', max_oos_days=0, caps=(200.,250.), verbose=True):
    output.mkdir(parents=True,exist_ok=True)
    data=read_data(data_path)
    changes=(data.diff()*1e4).dropna()
    train=changes.loc[changes.index<OOS_START]
    test=changes.loc[changes.index>=OOS_START]
    if max_oos_days: test=test.iloc[:max_oos_days]
    if len(train)<504 or len(test)<20: raise ValueError('Insufficient training/test data')
    ewma=np.cov(train.to_numpy()[:252],rowvar=False)
    for shock in train.to_numpy()[252:]: ewma=ewma_update(ewma,shock)
    history=list(train.to_numpy()[-TRAIN_HISTORY:])
    strategies=['Parallel30Y']
    for cap in caps:
        strategies += [f'EWMA_TurnoverAware_Cap{int(cap)}',f'HistCVaR_Cap{int(cap)}',
                       f'StressCVaR_Cap{int(cap)}']
    # Hold zero-coupon FACE amounts; convert to prior-close market values at each decision
    faces={name:np.zeros(5) for name in strategies}
    records=[]
    previous_dt=None
    terminal_positions={}
    for ii,(dt,obs) in enumerate(test.iterrows()):
        change=obs.to_numpy(dtype=float)
        prev_y=data.shift(1).loc[dt].to_numpy(dtype=float)
        prior_price=np.exp(-TENORS*prev_y)
        base_pnl=pnl(BASE,change)
        rec={'date':dt,'Unhedged_gross_pnl_m':base_pnl}
        planned=scheduled_rebalance(dt,previous_dt,frequency)
        for name in strategies:
            prev_mv=faces[name]*prior_price
            cap=150. if name=='Parallel30Y' else float(name.split('Cap')[-1])
            current_total=BASE+prev_mv
            forced_cap=(100.+np.abs(prev_mv).sum() > cap+1e-5)
            forced_stress=(name.startswith('StressCVaR') and
                           worst_stress(current_total)>STRESS_LOSS_LIMIT_M+1e-5)
            do_trade=(planned or forced_cap or forced_stress)
            traded=0.
            if do_trade:
                if name=='Parallel30Y':
                    hedge=np.array([0.,0.,-(BASE@TENORS)/30.])
                elif name.startswith('EWMA'):
                    hedge=solve_minvar(ewma,cap,prev_mv[HI])
                elif name.startswith('HistCVaR'):
                    hedge=solve_cvar(np.asarray(history),cap,prev_mv[HI],False)
                else:
                    hedge=solve_cvar(np.asarray(history),cap,prev_mv[HI],True)
                desired=np.zeros(5);desired[HI]=hedge
                traded=float(np.abs(desired-prev_mv).sum())
                faces[name]=desired/prior_price
            hedge_mv=faces[name]*prior_price
            full_mv=BASE+hedge_mv
            gross=100.+np.abs(hedge_mv).sum()
            if gross>cap+1e-4: raise RuntimeError('%s exceeds pre-shock gross cap' % name)
            rec[name+'_pnl_m']=pnl(full_mv,change)
            rec[name+'_turnover_m']=traded
            rec[name+'_hedge_gross_m']=float(np.abs(hedge_mv).sum())
            rec[name+'_gross_m']=gross
            rec[name+'_forced_cap']=int(forced_cap)
            rec[name+'_forced_stress']=int(forced_stress)
            rec[name+'_worst_stress_m']=worst_stress(full_mv)
            rec[name+'_dv01_usd_bp']=float((full_mv@TENORS)*100.)  # $m * 1e6 * 1e-4
        records.append(rec)
        history.append(change)  # AFTER today's forecast/P&L
        if len(history)>TRAIN_HISTORY: history.pop(0)
        ewma=ewma_update(ewma,change)
        previous_dt=dt
        if verbose and (ii+1)%500==0:
            print('Evaluated %d/%d OOS days' % (ii+1,len(test)),flush=True)
        if ii==len(test)-1:
            end_y=data.loc[dt].to_numpy(dtype=float)
            terminal_positions={name:BASE+faces[name]*np.exp(-TENORS*end_y)
                                for name in strategies}
    daily=pd.DataFrame(records).set_index('date')
    base_sd=daily.Unhedged_gross_pnl_m.std(ddof=1)
    summaries=[]
    for name in ['Unhedged']+strategies:
        if name=='Unhedged':
            net=daily.Unhedged_gross_pnl_m.to_numpy()
            gross_pnl=net
            trades=np.zeros(len(daily));charge=np.zeros(len(daily));turnover=np.zeros(len(daily))
            exposures=np.full(len(daily),100.);cap=100.;forcedcap=forcedstress=0
            stress_series=np.array([worst_stress(BASE)]*len(daily))
        else:
            gross_pnl=daily[name+'_pnl_m'].to_numpy()
            turnover=daily[name+'_turnover_m'].to_numpy()
            charge=daily[name+'_hedge_gross_m'].to_numpy()*ANNUAL_CAPITAL_CHARGE/252.
            trades=turnover*TRADE_BP*1e-4
            net=gross_pnl-trades-charge
            exposures=daily[name+'_gross_m'].to_numpy()
            cap=150 if name=='Parallel30Y' else float(name.split('Cap')[-1])
            forcedcap=int(daily[name+'_forced_cap'].sum())
            forcedstress=int(daily[name+'_forced_stress'].sum())
            stress_series=daily[name+'_worst_stress_m'].to_numpy()
        summaries.append({
            'strategy':name,'gross_cap_m':cap,'frequency':frequency,'days':len(net),
            'daily_net_pnl_sd_usd':np.std(net,ddof=1)*1e6,
            'vol_reduction_pct':100*(1-np.std(net,ddof=1)/base_sd),
            'historical_95pct_net_loss_usd':float(np.quantile(-net,.95)*1e6),
            'historical_95pct_es_usd':historical_es(net,.95)*1e6,
            'historical_99pct_es_usd':historical_es(net,.99)*1e6,
            'worst_daily_net_loss_usd':float(np.max(-net)*1e6),
            'total_turnover_usd':float(np.sum(turnover)*1e6),
            'execution_cost_usd':float(np.sum(trades)*1e6),
            'capital_charge_usd':float(np.sum(charge)*1e6),
            'net_shock_pnl_sum_usd':float(np.sum(net)*1e6),
            'net_max_drawdown_usd':max_drawdown(net)*1e6,
            'max_gross_exposure_usd':float(max(exposures)*1e6),
            'forced_cap_rebalances':forcedcap,'forced_stress_rebalances':forcedstress,
            'maximum_predicted_stress_loss_at_decision_usd':float(np.max(stress_series)*1e6),
        })
        daily[name+'_net_pnl_m']=net
    summ=pd.DataFrame(summaries).sort_values('historical_95pct_es_usd')
    stress=[]
    for name in ['Unhedged']+strategies:
        mv=BASE if name=='Unhedged' else terminal_positions[name]
        for s,shock in STRESSES.items():
            stress.append({'strategy':name,'scenario':s,
                           'stress_pnl_usd':pnl(mv,shock)*1e6})
    stresses=pd.DataFrame(stress)
    daily.to_csv(output/'phase7d_daily_positions_pnl_costs.csv')
    summ.to_csv(output/'phase7d_strategy_summary.csv',index=False)
    stresses.to_csv(output/'phase7d_latest_stresses.csv',index=False)
    scenario_summary=(stresses.groupby('strategy')['stress_pnl_usd'].min()
                      .rename('worst_latest_stress_pnl_usd'))
    summ=summ.join(scenario_summary,on='strategy')
    summ.to_csv(output/'phase7d_summary_with_stresses.csv',index=False)
    fig,ax=plt.subplots(figsize=(12,6))
    ax.bar(summ.strategy,summ.historical_95pct_es_usd/1e6)
    ax.set_ylabel('Historical 95% ES of net daily losses ($m)')
    ax.set_title('Phase 7D: Tail losses after illustrative costs')
    ax.tick_params(axis='x',rotation=65,labelsize=8)
    fig.tight_layout();fig.savefig(output/'01_historical_expected_shortfall.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6))
    for name in ['Unhedged']+strategies:
        ax.plot(daily.index,daily[name+'_net_pnl_m'].cumsum(),label=name)
    ax.set_ylabel('Cumulative hypothetical shock-only net P&L ($m)')
    ax.set_title('Not self-financing: no base carry/roll/funding offsets')
    ax.legend(fontsize=7,ncol=2)
    fig.tight_layout();fig.savefig(output/'02_cumulative_net_shock_pnl.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    subset=stresses[stresses.scenario.isin(['Parallel +100bp','Long end +100bp','Twist'])]
    pivot=subset.pivot(index='strategy',columns='scenario',values='stress_pnl_usd')/1e6
    pivot.loc[summ.strategy].plot.bar(ax=ax,rot=70)
    ax.set_ylabel('Latest as-of stress P&L ($m)')
    fig.tight_layout();fig.savefig(output/'03_latest_stress_pnl.png',dpi=150);plt.close(fig)
    if verbose:
        print('='*83)
        print('PHASE 7D — HISTORICAL CVaR, STRESS-AWARE & CAPITAL-CONSTRAINED HEDGING')
        print('='*83)
        print('REAL DATA:',data_path)
        print('Training:',train.index.min().date(),'to',train.index.max().date(),len(train),'changes')
        print('OOS:',test.index.min().date(),'to',test.index.max().date(),len(test),'changes')
        print('Daily benchmark $100m; hedge maturities 2Y/10Y/30Y; rebalance:',frequency)
        print('Historical CVaR alpha=%.0f%%, trailing shocks=%d; no future shocks in decisions' % (100*ALPHA,TRAIN_HISTORY))
        print('Hard stress limit for StressCVaR = $%.1fm; scenarios = %s' % (STRESS_LOSS_LIMIT_M, ', '.join(STRESSES)))
        print('Gross caps $m:',caps,'| hedge execution %.1fbp | illustrative capital charge %.1f%% annual' % (TRADE_BP,100*ANNUAL_CAPITAL_CHARGE))
        print('\nOUT-OF-SAMPLE SUMMARY (sorted by historical 95% ES):')
        columns=['strategy','daily_net_pnl_sd_usd','vol_reduction_pct',
                 'historical_95pct_es_usd','historical_99pct_es_usd',
                 'execution_cost_usd','capital_charge_usd','max_gross_exposure_usd',
                 'forced_cap_rebalances','forced_stress_rebalances']
        print(summ[columns].round(2).to_string(index=False))
        print('\nLATEST SIX-SCENARIO WORST STRESS P&L ($m):')
        print((scenario_summary/1e6).round(3).to_string())
        print('\nCAUTIONS: Hypothetical zero-coupon positions, no coupon carry, roll, base rebalance costs,')
        print('market impact, margin, financing offsets or executed option quotes. All risk metrics are OOS')
        print('but CVaR choices/tail threshold (95%) were selected during research, not untouched tests.')
        print('Hard stress limit holds when StressCVaR strategy is reoptimized; its exposures can')
        print('drift intra-day and after shock. Stress tests are specified scenarios, not all possible shocks.')
        print('Output:',output)
    return daily,summ,stresses


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--frequency',choices=['daily','weekly','monthly'],default='weekly')
    p.add_argument('--caps',type=float,nargs='+',default=[200.,250.])
    p.add_argument('--max-oos-days',type=int,default=0,help='For quick software tests only')
    a=p.parse_args()
    run(a.data,a.output,a.frequency,a.max_oos_days,tuple(a.caps))
