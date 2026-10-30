"""Project 2 / Phase 7C: constrained Treasury zero-coupon hedging.

Standalone Python >=3.9; requires numpy, pandas, scipy, matplotlib.
Reads Phase 1's REAL GSW fitted zero-coupon Treasury yields by default.
HYPOTHETICAL constant-tenor $100m Treasury zero-coupon portfolio.
Costs, funding and hedge instruments are assumptions, not executable quotes.
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "project2_phase1_outputs" / "daily_zero_yields_pct.csv"
DEFAULT_OUT = HERE / "project2_phase7c_outputs"
TENORS = np.array([2., 5., 10., 20., 30.])
COLS = ["2Y", "5Y", "10Y", "20Y", "30Y"]
BASE_M = np.array([10., 20., 30., 20., 20.])  # millions of USD, reset daily
HEDGE_IDX = np.array([0, 2, 4])
HEDGE_TENORS = TENORS[HEDGE_IDX]
BASE_DV01_MBP = BASE_M * TENORS * 1e-4  # USD millions per bp
OOS_START = pd.Timestamp("2018-01-01")
GROSS_LIMITS_M = [150., 200., 250.]
EWMA_LAMBDA = .94
COST_BPS = [1., 5.]
FUNDING_SPREADS = [0., .01, .03]  # 0%, 1%, 3% annual capital charge on gross hedge MV
RISK_TOLERANCE_M = .10  # $100,000 daily risk-tolerance scale, hypothetical
TURNOVER_SHADOW_BPS = 5.0  # turnover-aware optimizer, 5bp *shadow* cost, not realized cost


def read_yields(path):
    if not path.exists():
        raise FileNotFoundError("Phase 1 real Treasury CSV not found: " + str(path))
    d = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    d = d.loc[~d.index.duplicated(keep="last")]
    missing = [c for c in COLS if c not in d.columns]
    if missing:
        raise ValueError("Missing required tenor columns: " + str(missing))
    d = d[COLS].apply(pd.to_numeric, errors="coerce").dropna() / 100.
    if len(d) < 500 or not np.isfinite(d.to_numpy()).all() or d.abs().max().max() > .25:
        raise ValueError("Bad or insufficient yield observations; expected yields in PERCENT")
    return d


def cov_bp(data):
    c = np.cov(data, rowvar=False)
    return (c + c.T) / 2


def ewma_step(c, shock):
    c = EWMA_LAMBDA*c + (1-EWMA_LAMBDA)*np.outer(shock, shock)
    return (c+c.T)/2


def rebalance(today, previous, freq):
    if previous is None:
        return True
    if freq == "daily":
        return True
    if freq == "weekly":
        return today.to_period("W-SUN") != previous.to_period("W-SUN")
    return today.to_period("M") != previous.to_period("M")


def bond_shock_pnl(market_values_m, move_bp):
    # EXACT shock repricing in $ millions. Fixed tenor, no coupon/roll/carry.
    return float(np.dot(market_values_m, np.expm1(-TENORS*move_bp*1e-4)))


def solve_hedge(cov, gross_cap_m, prev_m, mode="MinVar", shadow_bps=5.):
    """Minimize covariance risk with exact parallel-DV01 neutral and gross-cap constraints.

    x: market values of signed zero hedges in USD millions at the preceding close.
    TurnoverAware adds an L1 turnover penalty via 3 auxiliary variables. All
    constraints are linear and the objective is convex; no future data used.
    """
    budget = gross_cap_m - BASE_M.sum()
    min_budget = (BASE_M @ TENORS)/max(HEDGE_TENORS)
    if budget + 1e-9 < min_budget:
        raise ValueError("Gross cap is too low to neutralize $142,000/bp: need >= $%.2fm" % (100+min_budget))
    cov = (cov+cov.T)/2
    # Small PSD ridge, for robust optimization in near-collinear yield changes.
    cov = cov + np.eye(5)*max(np.trace(cov)/5., .01)*1e-7
    factors = HEDGE_TENORS*1e-4
    base_d = BASE_DV01_MBP
    initial_x = np.array([0., 0., -(BASE_M @ TENORS)/30.])
    has_penalty = mode == "TurnoverAware"
    n = 6 if has_penalty else 3
    x0 = np.r_[initial_x, np.abs(initial_x-prev_m)] if has_penalty else initial_x
    base_foot = BASE_M @ TENORS

    # Sum(tenor * hedge $m) must equal negative of base duration-weighted $m.
    eq = np.zeros((1, n)); eq[0, :3] = HEDGE_TENORS / 30.
    constraints = [LinearConstraint(eq, [-base_foot/30.], [-base_foot/30.])]
    # Sum |x| <= budget is equivalent to 8 signed linear inequalities.
    signs = np.array(list(product([-1., 1.], repeat=3)))
    a = np.zeros((8, n)); a[:, :3] = signs
    constraints.append(LinearConstraint(a, -np.inf, budget))
    if has_penalty:
        # v_i >= abs(x_i - prev_i), represented with linear inequalities.
        turnover_a = np.zeros((6, n))
        for j in range(3):
            turnover_a[2*j, j] = 1.; turnover_a[2*j, 3+j] = -1.
            turnover_a[2*j+1, j] = -1.; turnover_a[2*j+1, 3+j] = -1.
        upper = np.r_[prev_m[0], -prev_m[0], prev_m[1], -prev_m[1], prev_m[2], -prev_m[2]]
        constraints.append(LinearConstraint(turnover_a, -np.inf, upper))
    bounds = [(-budget, budget)]*3 + ([(0., 2*budget+100.)]*3 if has_penalty else [])

    def fun(z):
        d = base_d.copy()
        d[HEDGE_IDX] += z[:3]*factors
        v = .5 * float(d @ cov @ d)
        g = np.zeros(n)
        g[:3] = (cov @ d)[HEDGE_IDX]*factors
        if has_penalty:
            v += RISK_TOLERANCE_M * shadow_bps*1e-4 * float(np.sum(z[3:]))
            g[3:] = RISK_TOLERANCE_M * shadow_bps*1e-4
        return 1000*v, 1000*g

    res = minimize(fun, x0, method="SLSQP", jac=True, bounds=bounds,
                   constraints=constraints, options={"maxiter":160, "ftol":1e-8})
    x = res.x[:3]
    neutral_error = abs(float(HEDGE_TENORS @ x + base_foot))
    gross = 100. + np.abs(x).sum()
    if not res.success or neutral_error > 1e-4 or gross > gross_cap_m+1e-4:
        raise RuntimeError("Hedge optimization failed: %s; neutrality=%.3g, gross=%.3f" %
                           (res.message, neutral_error, gross))
    return x


def max_drawdown(series):
    cumulative = series.cumsum()
    return float((cumulative - np.maximum.accumulate(np.r_[0., cumulative])[-len(cumulative):]).min())


def run(data_path, output, frequency="weekly", max_oos_days=0, verbose=True):
    output.mkdir(parents=True, exist_ok=True)
    yields = read_yields(data_path)
    changes = (yields.diff()*1e4).dropna()  # bp changes
    train = changes.loc[changes.index < OOS_START]
    test = changes.loc[changes.index >= OOS_START]
    if max_oos_days:
        test = test.iloc[:max_oos_days]
    if len(train) < 252 or len(test) < 40:
        raise ValueError("Need 252 train changes and 40 OOS changes")
    frozen = cov_bp(train.to_numpy())
    ewma = cov_bp(train.to_numpy()[:252])
    for shock in train.to_numpy()[252:]:
        ewma = ewma_step(ewma, shock)

    # 12 constrained strategies plus a feasible $147.33m gross duration hedge.
    keys = [(kind, mode, cap) for kind in ["Frozen", "EWMA"]
            for mode in ["MinVar", "TurnoverAware"] for cap in GROSS_LIMITS_M]
    strategies = ["%s_%s_Cap%d" % (a,b,int(c)) for a,b,c in keys]
    faces = {s: np.zeros(5) for s in strategies + ["Parallel30Y"]}
    prior = None
    records = []
    last_positions = {}
    cost_checks = []

    for dt, row in test.iterrows():
        shock = row.to_numpy(dtype=float)
        prev_y = yields.shift(1).loc[dt].to_numpy(dtype=float)  # known at prior close
        p0 = np.exp(-TENORS*prev_y)
        gross_pnl = bond_shock_pnl(BASE_M, shock)
        rec = {"date":dt, "Unhedged_gross_pnl_m":gross_pnl}
        scheduled = rebalance(dt, prior, frequency)

        # Historical covariance matrices only; today's shock is added AFTER P&L.
        covs = {"Frozen":frozen, "EWMA":ewma}
        for name in strategies + ["Parallel30Y"]:
            pre_m = faces[name]*p0
            drift_exposure_m = 100. + np.abs(pre_m).sum()
            forced = False
            if name == "Parallel30Y":
                cap = 150.
                target = np.array([0., 0., -float(BASE_M @ TENORS)/30.])
            else:
                kind, mode, captext = name.split("_")
                cap = float(captext.replace("Cap", ""))
            if not scheduled and drift_exposure_m > cap + 1e-5:
                forced = True
            do_trade = scheduled or forced
            traded_m = 0.
            if do_trade:
                if name == "Parallel30Y":
                    h = target
                else:
                    h = solve_hedge(covs[kind], cap, pre_m[HEDGE_IDX], mode,
                                    shadow_bps=TURNOVER_SHADOW_BPS)
                desired_m = np.zeros(5)
                desired_m[HEDGE_IDX] = h
                new_faces = desired_m / p0
                traded_m = float(np.sum(np.abs(new_faces-faces[name])*p0))
                faces[name] = new_faces
            hedge_m = faces[name]*p0
            gross_exposure_m = 100. + float(np.abs(hedge_m).sum())
            if gross_exposure_m > cap + 2e-4:
                raise AssertionError("Cap violated at pre-shock decision time: " + name)
            rec[name+"_gross_pnl_m"] = gross_pnl + bond_shock_pnl(hedge_m, shock)
            rec[name+"_traded_m"] = traded_m
            rec[name+"_hedge_gross_m"] = float(np.abs(hedge_m).sum())
            rec[name+"_gross_exposure_m"] = gross_exposure_m
            rec[name+"_forced_rebalances"] = int(forced)
            rec[name+"_parallel_dv01_usd_bp"] = float(1e6*(BASE_DV01_MBP.sum() +
                                      np.dot(hedge_m[HEDGE_IDX], HEDGE_TENORS*1e-4)))
        records.append(rec)
        prior = dt
        ewma = ewma_step(ewma, shock)
        for name in faces:
            last_positions[name] = faces[name] * np.exp(-TENORS*yields.loc[dt].to_numpy(dtype=float))

    daily = pd.DataFrame(records).set_index("date")
    strategies = ["Unhedged", "Parallel30Y"]+strategies
    ref_sd = daily.Unhedged_gross_pnl_m.std(ddof=1)
    summaries=[]
    for name in strategies:
        if name == "Unhedged":
            pnl = daily.Unhedged_gross_pnl_m
            turnover = pd.Series(0., index=daily.index)
            hedge_gross = turnover
            exposure = pd.Series(100., index=daily.index)
            forced = 0
        else:
            pnl = daily[name+"_gross_pnl_m"]
            turnover = daily[name+"_traded_m"]
            hedge_gross = daily[name+"_hedge_gross_m"]
            exposure = daily[name+"_gross_exposure_m"]
            forced = int(daily[name+"_forced_rebalances"].sum())
        for cbps in COST_BPS:
            for funding in FUNDING_SPREADS:
                # Simplified conservative gross-capital charge; NOT executed funding cashflows.
                execution_m = turnover * cbps * 1e-4
                funding_m = hedge_gross * funding / 252.
                net = pnl - execution_m - funding_m
                summaries.append({
                    "strategy":name, "frequency":frequency, "transaction_cost_bp":cbps,
                    "gross_cap_m":(int(name.split("Cap")[-1]) if "Cap" in name else (150 if name=="Parallel30Y" else 100)),
                    "annual_gross_hedge_charge_pct":100*funding,
                    "days":len(net), "daily_net_pnl_sd_usd":net.std(ddof=1)*1e6,
                    "vol_reduction_pct":100*(1-net.std(ddof=1)/ref_sd),
                    "total_hedge_turnover_usd":turnover.sum()*1e6,
                    "execution_cost_usd":execution_m.sum()*1e6,
                    "gross_capital_charge_usd":funding_m.sum()*1e6,
                    "net_shock_pnl_sum_usd":net.sum()*1e6,
                    "net_shock_max_drawdown_usd":max_drawdown(net.to_numpy())*1e6,
                    "historical_95pct_net_loss_usd":float((-net).quantile(.95))*1e6,
                    "historical_99pct_net_loss_usd":float((-net).quantile(.99))*1e6,
                    "worst_daily_net_loss_usd":float((-net).max())*1e6,
                    "max_gross_exposure_usd":exposure.max()*1e6,
                    "forced_cap_rebalances":forced,
                    "mean_abs_parallel_dv01_usd_bp":(0 if name=="Unhedged" else daily[name+"_parallel_dv01_usd_bp"].abs().mean())
                })
    summary = pd.DataFrame(summaries)
    daily.to_csv(output / "phase7c_daily_positions_pnl_and_cost_drivers.csv")
    summary.to_csv(output / "phase7c_cost_capital_sensitivity.csv",index=False)

    scenarios = {
        "Parallel +100bp":np.array([100.,100.,100.,100.,100.]),
        "Parallel -100bp":np.array([-100.,-100.,-100.,-100.,-100.]),
        "Bear steepener":np.array([20.,40.,60.,80.,100.]),
        "Bear flattener":np.array([100.,80.,60.,40.,20.]),
        "Long end +100bp":np.array([0.,0.,0.,100.,100.]),
        "Twist":np.array([-50.,-25.,0.,50.,100.]),
    }
    stress=[]
    for name in strategies:
        mv = BASE_M.copy() if name=="Unhedged" else BASE_M + last_positions[name]
        for shockname,move in scenarios.items():
            stress.append({"strategy":name,"scenario":shockname,
                           "instantaneous_shock_pnl_usd":bond_shock_pnl(mv,move)*1e6})
    stress = pd.DataFrame(stress)
    stress.to_csv(output / "phase7c_latest_stress_scenarios.csv",index=False)

    focus = summary[(summary.transaction_cost_bp==1) &
                    (summary.annual_gross_hedge_charge_pct==1)].copy().sort_values("daily_net_pnl_sd_usd")
    focus.to_csv(output / "phase7c_reference_1bp_1pct_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(13,6))
    ax.bar(focus.strategy,focus.daily_net_pnl_sd_usd)
    ax.set_ylabel("Daily net P&L SD ($)")
    ax.set_title("Phase 7C: 1bp hedge execution + 1% annual gross-hedge capital charge")
    ax.tick_params(axis="x",rotation=70)
    fig.tight_layout();fig.savefig(output/"01_constrained_hedge_volatility.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,6))
    s = summary[(summary.transaction_cost_bp==1)&
                (summary.strategy.str.contains("EWMA_TurnoverAware_Cap"))]
    for cap,g in s.groupby("gross_cap_m"):
        ax.plot(g.annual_gross_hedge_charge_pct,g.net_shock_pnl_sum_usd/1e6,
                marker="o",label="$%dm gross cap"%cap)
    ax.set_xlabel("Annual capital charge on gross hedge MV (%)")
    ax.set_ylabel("Cumulative hypothetical net shock P&L ($m)")
    ax.set_title("Funding/capital charge sensitivity; no carry or base-rebalance cost")
    ax.legend();fig.tight_layout();fig.savefig(output/"02_capital_charge_sensitivity.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,6))
    for name in ["Unhedged", "Parallel30Y", "EWMA_MinVar_Cap150", "EWMA_TurnoverAware_Cap150", "EWMA_TurnoverAware_Cap200"]:
        pnl = daily.Unhedged_gross_pnl_m if name=="Unhedged" else daily[name+"_gross_pnl_m"]
        costs = pd.Series(0.,index=daily.index) if name=="Unhedged" else (
            daily[name+"_traded_m"]*1e-4 + daily[name+"_hedge_gross_m"]*.01/252.)
        ax.plot(daily.index,(pnl-costs).cumsum(),label=name)
    ax.set_ylabel("Cumulative shock-only net P&L ($m)")
    ax.set_title("Illustrative path; NOT an investment return or self-financing portfolio")
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/"03_cumulative_net_shock_pnl.png",dpi=150);plt.close(fig)

    if verbose:
        print("="*81)
        print("PHASE 7C — CAPITAL-CONSTRAINED TREASURY HEDGING, REAL HISTORICAL YIELDS")
        print("="*81)
        print("Input:",data_path)
        print("Train:",train.index[0].date(),"through",train.index[-1].date(),"|",len(train),"changes")
        print("Out-of-sample:",test.index[0].date(),"through",test.index[-1].date(),"|",len(test),"changes")
        print("Portfolio $100m; parallel DV01 $142,000/bp; weekly or monthly schedule:",frequency)
        print("Gross caps: $150m, $200m, $250m; exact parallel-DV01 neutrality at scheduled trades")
        print("Optimization: 2Y/10Y/30Y hypothetical zero-bond hedges; turnover shadow=",TURNOVER_SHADOW_BPS,"bp")
        print("Capital charge is a PROXY on absolute hedge MV, not an observed funding rate")
        print("\nREFERENCE: 1bp trade cost and 1% annual capital charge (ranked by net P&L SD)")
        cols=["strategy","gross_cap_m","daily_net_pnl_sd_usd","vol_reduction_pct","total_hedge_turnover_usd",
              "execution_cost_usd","gross_capital_charge_usd","net_shock_pnl_sum_usd", "max_gross_exposure_usd", "forced_cap_rebalances"]
        print(focus[cols].round(2).to_string(index=False))
        print("\nLATEST LONG-END +100bp STRESS ($m):")
        print((stress[stress.scenario=="Long end +100bp"].set_index("strategy").instantaneous_shock_pnl_usd/1e6).round(3).to_string())
        print("\nLIMITATIONS: Constant-tenor hypothetical zeros, no carry, actual funding cashflows, financing offsets,")
        print("base position turnover, transaction impact, futures/CTD basis or margin. Shadow penalty is illustrative.")
        print("The numerical gross cap applies at pre-shock trading decision times; intraday moves can change exposure.")
        print("Changing real market data AFTER a forecast date cannot change its hedge signal.")
        print("Saved CSVs/figures to:",output)
    return daily, summary, stress


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=DEFAULT_DATA)
    p.add_argument("--output",type=Path,default=DEFAULT_OUT)
    p.add_argument("--frequency",choices=["daily","weekly","monthly"],default="weekly")
    p.add_argument("--max-oos-days",type=int,default=0,help="Only for fast development tests")
    args=p.parse_args()
    run(args.data,args.output,args.frequency,args.max_oos_days)
