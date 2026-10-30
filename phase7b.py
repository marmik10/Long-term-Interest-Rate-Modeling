"""Project 2 - Phase 7B: Dynamic Treasury hedge rebalancing and tail-risk validation.

Requires Phase 1's real Federal Reserve zero-coupon Treasury curve CSV.
Python 3.9+; numpy, pandas, scipy, matplotlib.

This is a HYPOTHETICAL constant-tenor, shock-only portfolio experiment.
No coupon/carry/roll-down, financing, real bond execution, or market liquidity modeling.
Costs are illustrative charges on hedge notional traded. Base holdings are reset
exogenously to constant dollar values every day; their rebalancing costs are excluded.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "project2_phase1_outputs" / "daily_zero_yields_pct.csv"
DEFAULT_OUTPUT = HERE / "project2_phase7b_outputs"
TENORS = np.array([2., 5., 10., 20., 30.])
COLUMNS = ["2Y", "5Y", "10Y", "20Y", "30Y"]
BASE_MV = 1_000_000 * np.array([10., 20., 30., 20., 20.])
HEDGE_IX = np.array([0, 2, 4])  # hypothetical 2Y / 10Y / 30Y zero-bond hedges
OOS_START = pd.Timestamp("2018-01-01")
EWMA_DECAY = 0.94
T_DF = 6  # FIXED Student-t degrees of freedom; not an estimated market parameter
COST_BPS = [0., 1., 5.]  # sensitivity cases, per traded dollar of hedge market value


def read_yields(path):
    if not path.exists():
        raise FileNotFoundError("Missing Phase 1 CSV: " + str(path))
    d = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    d = d.loc[~d.index.duplicated(keep="last")]
    missing = [c for c in COLUMNS if c not in d.columns]
    if missing:
        raise ValueError("Missing expected maturity columns: " + str(missing))
    d = d[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna() / 100.
    if not np.isfinite(d.to_numpy()).all() or d.abs().to_numpy().max() > 0.25:
        raise ValueError("Nonfinite or implausible yields; expected input units are percent")
    return d


def cov_bp(x):
    return np.atleast_2d(np.cov(x, rowvar=False))


def ewma_step(s, shock):
    return EWMA_DECAY * s + (1.0 - EWMA_DECAY) * np.outer(shock, shock)


def dv01(mv):
    return mv * TENORS * 1e-4


def shock_pnl(mv, dy_bp):
    return float(np.dot(mv, np.expm1(-TENORS * dy_bp * 1e-4)))


def minimum_variance_hedge(base_d, cov):
    """Return signed 2Y/10Y/30Y hedge market values; parallel DV01 neutral."""
    idx = HEDGE_IX
    sub = cov[np.ix_(idx, idx)]
    regularizer = max(float(np.trace(sub)) / 3.0, 1e-12) * .01
    a = sub + np.eye(3) * regularizer
    raw = -np.linalg.solve(a, (cov @ base_d)[idx])
    weight = np.linalg.solve(a, np.ones(3))
    goal = -base_d.sum()
    hedge_d = raw + (goal - raw.sum()) * weight / weight.sum()
    hedge_mvs = np.zeros(5)
    hedge_mvs[idx] = hedge_d / (TENORS[idx] * 1e-4)
    return hedge_mvs


def desired_hedge(method, base_d, frozen, ewma):
    if method == "Parallel10Y":
        h = np.zeros(5)
        h[2] = -base_d.sum() / (TENORS[2] * 1e-4)
        return h
    return minimum_variance_hedge(base_d, frozen if method == "MinVarFrozen" else ewma)


def rebalance_now(today, previous, freq):
    if previous is None or freq == "daily":
        return True
    if freq == "weekly":
        return today.to_period("W-SUN") != previous.to_period("W-SUN")
    return today.to_period("M") != previous.to_period("M")


def kupiec_and_independence(breaches, confidence):
    """Kupiec unconditional coverage and Christoffersen independence diagnostics."""
    b = np.asarray(breaches, dtype=int)
    n, count = len(b), int(b.sum())
    p = 1.0 - confidence
    tiny = 1e-12

    def bern_log(prob, failures, successes):
        prob = np.clip(prob, tiny, 1.0 - tiny)
        return failures * np.log1p(-prob) + successes * np.log(prob)

    q = count / n
    uc = 2 * (bern_log(q, n-count, count) - bern_log(p, n-count, count))
    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    p01 = n01 / max(n00+n01, 1)
    p11 = n11 / max(n10+n11, 1)
    all_rate = (n01+n11) / max(n-1, 1)
    ll_ind = bern_log(all_rate, n00+n10, n01+n11)
    ll_dep = bern_log(p01, n00, n01) + bern_log(p11, n10, n11)
    independence = max(0., 2 * (ll_dep-ll_ind))
    return float(chi2.sf(max(uc, 0.), 1)), float(chi2.sf(independence, 1))


def var_and_es(sd, confidence, distribution):
    if distribution == "Normal":
        z = norm.ppf(confidence)
        return sd*z, sd*norm.pdf(z)/(1-confidence)
    # Standardized t(df=6) has variance 1, i.e. volatility matches Normal case.
    z = student_t.ppf(confidence, T_DF)
    scale = np.sqrt((T_DF - 2)/T_DF)
    es = (T_DF + z*z) / (T_DF-1) * student_t.pdf(z, T_DF)/(1-confidence)
    return sd*scale*z, sd*scale*es


def evaluate(data_path, output, verbose=True):
    output.mkdir(exist_ok=True, parents=True)
    yields = read_yields(data_path)
    changes = (yields.diff() * 10000).dropna()
    train = changes.loc[changes.index < OOS_START]
    test = changes.loc[changes.index >= OOS_START]
    if len(train) < 252 or len(test) < 100:
        raise ValueError("Need >=252 training changes and >=100 out-of-sample changes")

    frozen = cov_bp(train.to_numpy())
    ewma = cov_bp(train.to_numpy()[:252])
    for x in train.to_numpy()[252:]:
        ewma = ewma_step(ewma, x)
    rolling = train.to_numpy()[-252:].copy()
    base_d = dv01(BASE_MV)

    methods = ("Parallel10Y", "MinVarFrozen", "MinVarEWMA")
    frequencies = ("daily", "weekly", "monthly")
    strategies = [f"{method}_{frequency}" for method in methods for frequency in frequencies]
    faces = {s: np.zeros(5) for s in strategies}
    records, var_rows = [], []
    prior_date = None
    prev_yields = yields.shift(1).loc[test.index]

    for date, shock_s in test.iterrows():
        shock = shock_s.to_numpy(dtype=float)
        y0 = prev_yields.loc[date].to_numpy(dtype=float)
        p0 = np.exp(-TENORS*y0)  # prev-close zero-bond prices per unit face
        row = {"date": date, "Unhedged_gross_pnl": shock_pnl(BASE_MV, shock)}
        covs = {"Frozen": frozen, "Rolling252": cov_bp(rolling), "EWMA094": ewma}

        for s in strategies:
            method, freq = s.rsplit("_", 1)
            dollar_turnover = 0.0
            if rebalance_now(date, prior_date, freq):
                desired_mv = desired_hedge(method, base_d, frozen, ewma)
                new_faces = desired_mv / p0
                dollar_turnover = float(np.sum(np.abs(new_faces - faces[s])*p0))
                faces[s] = new_faces
            hedge_mv = faces[s] * p0  # actual previous-close market value after drift
            pnl = row["Unhedged_gross_pnl"] + shock_pnl(hedge_mv, shock)
            row[f"{s}_gross_pnl"] = pnl
            row[f"{s}_turnover_usd"] = dollar_turnover
            row[f"{s}_gross_exposure_usd"] = BASE_MV.sum() + np.abs(hedge_mv).sum()
            row[f"{s}_net_parallel_dv01"] = float((base_d + dv01(hedge_mv)).sum())

        # VaR/ES forecasts for the UNHEDGED $100m portfolio only.
        # Model parameters and covariance are frozen at previous day's close.
        for name, cov in covs.items():
            sd = np.sqrt(max(float(base_d @ cov @ base_d), 0.))
            for distribution in ("Normal", "StudentT6"):
                for confidence in (.95, .99):
                    var, es = var_and_es(sd, confidence, distribution)
                    var_rows.append({"date": date, "risk_model": name,
                                     "distribution": distribution,
                                     "confidence": confidence, "var_usd": var,
                                     "es_usd": es,
                                     "actual_loss_usd": -row["Unhedged_gross_pnl"],
                                     "breach": -row["Unhedged_gross_pnl"] > var})
        records.append(row)
        prior_date = date
        ewma = ewma_step(ewma, shock)  # ONLY now introduce today's shock
        rolling = np.vstack((rolling[1:], shock))

    daily = pd.DataFrame(records).set_index("date")
    var_daily = pd.DataFrame(var_rows)
    ref_sd = daily["Unhedged_gross_pnl"].std(ddof=1)
    summaries = []
    for s in ["Unhedged"] + strategies:
        pnl = daily["Unhedged_gross_pnl"] if s == "Unhedged" else daily[f"{s}_gross_pnl"]
        turnover = pd.Series(0., index=daily.index) if s == "Unhedged" else daily[f"{s}_turnover_usd"]
        exposure = pd.Series(BASE_MV.sum(), index=daily.index) if s == "Unhedged" else daily[f"{s}_gross_exposure_usd"]
        for cost_bp in COST_BPS:
            cost = turnover * cost_bp * 1e-4
            net = pnl - cost
            summaries.append({"strategy": s, "cost_bp_per_traded_dollar": cost_bp,
                              "days": len(net), "net_pnl_sd_usd": net.std(ddof=1),
                              "vol_reduction_pct_vs_unhedged": 100*(1-net.std(ddof=1)/ref_sd),
                              "total_hedge_turnover_usd": turnover.sum(),
                              "total_hedge_cost_usd": cost.sum(),
                              "net_shock_pnl_sum_usd": net.sum(),
                              "historical_95pct_net_loss_usd": (-net).quantile(.95),
                              "worst_daily_net_loss_usd": (-net).max(),
                              "avg_gross_exposure_usd": exposure.mean(),
                              "max_gross_exposure_usd": exposure.max()})
    hedge_summary = pd.DataFrame(summaries)

    var_summaries = []
    for (model, distribution, confidence), g in var_daily.groupby(
            ["risk_model", "distribution", "confidence"], sort=True):
        b = g.breach.to_numpy(dtype=bool)
        p_uc, p_ind = kupiec_and_independence(b, confidence)
        loss_on_breach = g.loc[b, "actual_loss_usd"]
        es_on_breach = g.loc[b, "es_usd"]
        var_summaries.append({"model": model, "distribution": distribution,
                              "confidence": confidence, "days": len(g),
                              "breaches": int(b.sum()), "expected_breaches": len(g)*(1-confidence),
                              "breach_pct": 100*b.mean(),
                              "kupiec_pvalue": p_uc,
                              "independence_pvalue": p_ind,
                              "mean_predicted_var_usd": g.var_usd.mean(),
                              "mean_predicted_es_usd": g.es_usd.mean(),
                              "realized_breach_loss_over_predicted_es":
                              float(loss_on_breach.mean()/es_on_breach.mean()) if len(loss_on_breach) else np.nan})
    var_summary = pd.DataFrame(var_summaries)

    daily.to_csv(output / "phase7b_daily_hedge_pnl_and_turnover.csv")
    hedge_summary.to_csv(output / "phase7b_cost_and_rebalance_summary.csv", index=False)
    var_daily.to_csv(output / "phase7b_daily_var_es.csv", index=False)
    var_summary.to_csv(output / "phase7b_var_es_backtest_summary.csv", index=False)

    # Cost-adjusted model comparisons. Trading cost convention is 1bp per traded dollar.
    reference = hedge_summary.loc[hedge_summary.cost_bp_per_traded_dollar == 1.].copy()
    reference = reference.sort_values("net_pnl_sd_usd")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(reference.strategy, reference.net_pnl_sd_usd)
    ax.set_title("Out-of-sample daily shock + hedge-cost P&L volatility (1bp cost)")
    ax.set_ylabel("P&L standard deviation (USD)")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(output / "01_cost_adjusted_hedge_volatility.png", dpi=150)
    plt.close(fig)

    sensitivity = hedge_summary[hedge_summary.strategy.str.contains("MinVarEWMA")].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for s, part in sensitivity.groupby("strategy"):
        ax.plot(part.cost_bp_per_traded_dollar, part.total_hedge_cost_usd/1e6,
                marker="o", label=s)
    ax.set_xlabel("Cost per traded hedge dollar (bps)")
    ax.set_ylabel("Total modeled hedge costs ($ millions)")
    ax.set_title("Transaction cost sensitivity: EWMA hedge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "02_transaction_cost_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for s in ("Unhedged", "Parallel10Y_weekly", "MinVarFrozen_weekly", "MinVarEWMA_weekly"):
        pnl = daily["Unhedged_gross_pnl"] if s == "Unhedged" else daily[f"{s}_gross_pnl"] - daily[f"{s}_turnover_usd"]*1e-4
        ax.plot((pnl.cumsum()/1e6).index, pnl.cumsum()/1e6, label=s)
    ax.set_title("Cumulative hypothetical shock-only P&L, 1bp hedge cost")
    ax.set_ylabel("Cumulative P&L ($ millions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "03_cumulative_shock_pnl.png", dpi=150)
    plt.close(fig)

    if verbose:
        print("="*78)
        print("PHASE 7B — REAL TREASURY DYNAMIC HEDGING & TAIL RISK")
        print("="*78)
        print("Input:", data_path)
        print("Training:", train.index[0].date(), "to", train.index[-1].date(), "|", len(train), "daily changes")
        print("OOS:", test.index[0].date(), "to", test.index[-1].date(), "|", len(test), "daily changes")
        print("Portfolio: $100m constant-dollar hypothetical zero-coupon holdings")
        print("Hedge methods: parallel 10Y, frozen min-variance, EWMA min-variance")
        print("Schedules: daily, weekly, monthly; cost scenarios:", COST_BPS, "bps/traded hedge dollar")
        print("\nHEDGE COMPARISON AT 1 BP COST (ranked by net daily P&L volatility):")
        cols = ["strategy", "net_pnl_sd_usd", "vol_reduction_pct_vs_unhedged",
                "total_hedge_turnover_usd", "total_hedge_cost_usd", "max_gross_exposure_usd"]
        print(reference[cols].round(2).to_string(index=False))
        print("\n99% UNHEDGED VaR/ES BACKTEST:")
        cols = ["model", "distribution", "breaches", "expected_breaches",
                "kupiec_pvalue", "independence_pvalue", "realized_breach_loss_over_predicted_es"]
        print(var_summary.loc[var_summary.confidence == .99, cols].round(4).to_string(index=False))
        print("\nNOTES: Student-t df=6 fixed; ES ratio is descriptive, not a formal ES backtest.")
        print("Shock-only, constant-tenor bond repricing; base daily-reset cost excluded.")
        print("Hedges use synthetic zero positions; no financing, carry, roll, execution or market impact.")
        print("Any high hedge gross exposure may make results economically infeasible.")
        print("Saved 4 CSVs and 3 figures in:", output)
    return daily, hedge_summary, var_daily, var_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    evaluate(args.data, args.output)
