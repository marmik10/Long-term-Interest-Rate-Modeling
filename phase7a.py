"""Project 2 - Phase 7A: Daily Treasury portfolio VaR, hedging, and stresses.

Uses REAL fitted Federal Reserve zero-coupon yields saved in Phase 1.
No synthetic market data is created by this script.
Python 3.9+ | numpy, pandas, scipy, matplotlib.

Assumptions: constant-dollar rebalanced zero-coupon exposures at constant maturities;
shock-only daily mark-to-market P&L (no accrual, roll, coupon, funding or costs).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "project2_phase1_outputs" / "daily_zero_yields_pct.csv"
OUTPUT = ROOT / "project2_phase7a_outputs"
OUTPUT.mkdir(exist_ok=True)

TENORS = np.array([2., 5., 10., 20., 30.])
COLUMNS = [f"{int(t)}Y" for t in TENORS]
# Hypothetical market values; these five positions sum to $100 million.
BASE_MV = np.array([10., 20., 30., 20., 20.]) * 1_000_000
START_OOS = "2018-01-01"
EWMA_LAMBDA = 0.94
HEDGE_TENOR_INDICES = np.array([0, 2, 4])  # 2Y, 10Y, 30Y


def load_rates(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Phase 1 first, or copy its daily_zero_yields_pct.csv here."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Treasury tenor columns: {missing}")
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.loc[~df.index.duplicated(keep="last")]
    if len(df) < 300:
        raise ValueError("Need at least 300 complete daily yield curves.")
    if not np.isfinite(df.to_numpy()).all() or df.abs().to_numpy().max() > 25:
        raise ValueError("Nonfinite / implausibly large yields. Check percent units.")
    return df / 100.0  # Convert quoted percent to decimal continuous yields


def cov_bp(x):
    """One-day covariance of bp changes (NOT annualized)."""
    return np.atleast_2d(np.cov(x, rowvar=False))


def ewma_update(cov, x, lam=EWMA_LAMBDA):
    # Mean-zero innovation convention; yesterday's shock applied AFTER yesterday's forecast.
    return lam * cov + (1.0 - lam) * np.outer(x, x)


def risk_dv01(mv):
    """Signed dollar sensitivity for a one-basis-point move at each tenor."""
    return mv * TENORS * 0.0001


def exact_shock_pnl(mv, dy_bp):
    """Zero-coupon fixed-T mark-to-market shock; ignores elapsed time/carry."""
    return float(np.sum(mv * np.expm1(-TENORS * dy_bp * 0.0001)))


def normal_var_es(dv01, covariance, confidence):
    sigma = float(np.sqrt(max(dv01 @ covariance @ dv01, 0.0)))
    z = norm.ppf(confidence)
    return sigma * z, sigma * norm.pdf(z) / (1 - confidence)


def min_variance_level_neutral_hedge(base_dv01, cov):
    """3-instrument hedge: neutralize parallel DV01, then minimize covariance risk.

    Returns 2Y/10Y/30Y hedge dollar DV01s.  A 1% ridge stabilizes
    near-singular yield-shock covariance (not a transaction-cost model).
    """
    ix = HEDGE_TENOR_INDICES
    c = cov[np.ix_(ix, ix)]
    ridge = 0.01 * np.trace(c) / len(ix)
    a = c + ridge * np.eye(len(ix))
    rhs = (cov @ base_dv01)[ix]
    u = np.ones(len(ix))
    raw = -np.linalg.solve(a, rhs)
    v = np.linalg.solve(a, u)
    target = -float(np.sum(base_dv01))
    h = raw + ((target - np.sum(raw)) / np.sum(v)) * v
    return h


def hedge_mv_from_dv01(h):
    mv = np.zeros(len(TENORS))
    mv[HEDGE_TENOR_INDICES] = h / (TENORS[HEDGE_TENOR_INDICES] * 0.0001)
    return mv


def kupiec_test(breaches, coverage):
    """Unconditional coverage likelihood-ratio test, approximate chi-square(1)."""
    n = int(len(breaches))
    x = int(np.sum(breaches))
    p = 1.0 - coverage
    q = x / n
    eps = 1e-300
    null_ll = (n - x) * np.log1p(-p) + x * np.log(p)
    alt_ll = ((n - x) * np.log(max(1.0 - q, eps))
              + x * np.log(max(q, eps)))
    lr = max(0., -2.0 * (null_ll - alt_ll))
    return x, n * p, float(chi2.sf(lr, df=1))


def evaluate_hedge(name, pnl, unhedged_std):
    return {
        "strategy": name,
        "observations": len(pnl),
        "daily_pnl_sd_usd": np.std(pnl, ddof=1),
        "daily_pnl_mean_usd_shock_only": np.mean(pnl),
        "historical_95pct_loss_usd": np.quantile(-pnl, 0.95),
        "worst_daily_loss_usd": np.max(-pnl),
        "volatility_reduction_pct": 100 * (1 - np.std(pnl, ddof=1) / unhedged_std),
    }


def main():
    yields = load_rates(DATA_FILE)
    # For every day t, compare current and previous published yield estimate.
    dy_bps = yields.diff().dropna() * 10000
    train = dy_bps.loc[dy_bps.index < pd.Timestamp(START_OOS)]
    test = dy_bps.loc[dy_bps.index >= pd.Timestamp(START_OOS)]
    if len(train) < 252 or len(test) < 30:
        raise ValueError("Need >=252 training yield changes and >=30 test yield changes.")

    frozen = cov_bp(train.to_numpy())
    history = dy_bps.to_numpy()
    # Initialize EWMA on first 252 historical changes, then update to end of 2017.
    ewma = cov_bp(train.to_numpy()[:252])
    for shock in train.to_numpy()[252:]:
        ewma = ewma_update(ewma, shock)
    last_252 = train.to_numpy()[-252:].copy()
    d = risk_dv01(BASE_MV)

    pnl_records = []
    var_records = []
    hedge_records = []

    for date, shock_series in test.iterrows():
        shock = shock_series.to_numpy(dtype=float)
        rolling = cov_bp(last_252)
        covs = {"Frozen": frozen, "Rolling252": rolling, "EWMA094": ewma}
        # Hedge is selected BEFORE today's rate move is observed.
        mv_parallel = BASE_MV.copy()
        mv_parallel[2] += -d.sum() / (TENORS[2] * 0.0001)
        mv_frozen = BASE_MV + hedge_mv_from_dv01(min_variance_level_neutral_hedge(d, frozen))
        mv_ewma = BASE_MV + hedge_mv_from_dv01(min_variance_level_neutral_hedge(d, ewma))
        positions = {
            "Unhedged": BASE_MV,
            "Parallel_10Y": mv_parallel,
            "MinVar_Frozen": mv_frozen,
            "MinVar_EWMA": mv_ewma,
        }

        row = {"date": date}
        for name, mv in positions.items():
            row[name] = exact_shock_pnl(mv, shock)
        pnl_records.append(row)

        for name, cov in covs.items():
            for confidence in (0.95, 0.99):
                VaR, ES = normal_var_es(d, cov, confidence)
                loss = -row["Unhedged"]
                var_records.append({
                    "date": date, "risk_model": name, "confidence": confidence,
                    "var_usd": VaR, "es_usd": ES, "realized_loss_usd": loss,
                    "breach": loss > VaR,
                })

        hedge_records.append({
            "date": date,
            "parallel_net_dv01_usd_per_bp": risk_dv01(mv_parallel).sum(),
            "frozen_net_dv01_usd_per_bp": risk_dv01(mv_frozen).sum(),
            "ewma_net_dv01_usd_per_bp": risk_dv01(mv_ewma).sum(),
            "frozen_gross_mv_usd": np.sum(np.abs(mv_frozen)),
            "ewma_gross_mv_usd": np.sum(np.abs(mv_ewma)),
            **{f"ewma_mv_{col}_usd": mv_ewma[i] for i, col in enumerate(COLUMNS)},
        })
        # Only now may today's observation enter tomorrow's forecasts.
        ewma = ewma_update(ewma, shock)
        last_252 = np.vstack([last_252[1:], shock])

    pnl_df = pd.DataFrame(pnl_records).set_index("date")
    risk_df = pd.DataFrame(var_records)
    hedge_df = pd.DataFrame(hedge_records).set_index("date")

    summary = []
    for (model, confidence), g in risk_df.groupby(["risk_model", "confidence"]):
        breaks, expected, pvalue = kupiec_test(g.breach.to_numpy(), confidence)
        tail = g.loc[g.breach, "realized_loss_usd"]
        summary.append({
            "risk_model": model, "confidence": confidence,
            "days": len(g), "breaches": breaks,
            "expected_breaches": expected,
            "breach_rate_pct": 100 * breaks / len(g),
            "kupiec_pvalue_approx": pvalue,
            "mean_predicted_var_usd": g.var_usd.mean(),
            "mean_predicted_es_usd": g.es_usd.mean(),
            "mean_realized_breach_loss_usd": tail.mean() if len(tail) else np.nan,
        })
    var_summary = pd.DataFrame(summary).sort_values(["confidence", "risk_model"])
    unhedged_std = pnl_df["Unhedged"].std(ddof=1)
    hedge_summary = pd.DataFrame([
        evaluate_hedge(c, pnl_df[c].to_numpy(), unhedged_std)
        for c in pnl_df.columns
    ]).sort_values("daily_pnl_sd_usd")

    final_cov = ewma  # Includes final day's rates: valid for today's closing risk/stress snapshot.
    final_ewma_hedge = BASE_MV + hedge_mv_from_dv01(min_variance_level_neutral_hedge(d, final_cov))
    final_parallel = BASE_MV.copy()
    final_parallel[2] -= d.sum() / (TENORS[2] * 0.0001)
    final_frozen = BASE_MV + hedge_mv_from_dv01(min_variance_level_neutral_hedge(d, frozen))
    scenarios = {
        "Parallel +100bp": [100, 100, 100, 100, 100],
        "Parallel -100bp": [-100, -100, -100, -100, -100],
        "Bear steepener": [0, 25, 50, 75, 100],
        "Bear flattener": [100, 75, 50, 25, 0],
        "Twist": [-50, -25, 0, 50, 100],
        "Long-end +100bp": [0, 0, 0, 100, 100],
    }
    stress_records = []
    for scen, bp in scenarios.items():
        shocks = np.array(bp, dtype=float)
        for label, mv in {
            "Unhedged": BASE_MV, "Parallel_10Y": final_parallel,
            "MinVar_Frozen": final_frozen, "MinVar_EWMA": final_ewma_hedge
        }.items():
            dy = shocks / 10000
            linear = -float(np.dot(risk_dv01(mv), shocks))
            convex = float(0.5 * np.sum(mv * (TENORS * dy)**2))
            stress_records.append({
                "scenario": scen, "strategy": label,
                "exact_pnl_usd": exact_shock_pnl(mv, shocks),
                "linear_pnl_usd": linear,
                "duration_convexity_approx_usd": linear + convex,
            })
    stress_df = pd.DataFrame(stress_records)

    # Persist exact daily risk forecasts so results can be audited later.
    var_summary.to_csv(OUTPUT / "phase7a_var_backtest_summary.csv", index=False)
    risk_df.to_csv(OUTPUT / "phase7a_daily_var_predictions.csv", index=False)
    hedge_summary.to_csv(OUTPUT / "phase7a_hedge_comparison.csv", index=False)
    hedge_df.to_csv(OUTPUT / "phase7a_daily_hedge_exposures.csv")
    pnl_df.to_csv(OUTPUT / "phase7a_daily_shock_pnl.csv")
    stress_df.to_csv(OUTPUT / "phase7a_stress_tests.csv", index=False)

    # Charts
    fig, ax = plt.subplots(figsize=(11, 5))
    hedge_summary.sort_values("daily_pnl_sd_usd").plot(
        x="strategy", y="daily_pnl_sd_usd", kind="bar", ax=ax, legend=False
    )
    ax.set_title("Realized daily P&L volatility: Treasury hedge strategies")
    ax.set_ylabel("USD per day")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_hedge_volatility.png", dpi=140)
    plt.close(fig)

    view = risk_df.loc[(risk_df.confidence == 0.99) & (risk_df.risk_model.isin(["Frozen", "EWMA094"]))]
    fig, ax = plt.subplots(figsize=(12, 5))
    for model, g in view.groupby("risk_model"):
        ax.plot(g.date, g.var_usd / 1e6, label=model + " VaR")
    ax.plot(pnl_df.index, -pnl_df.Unhedged / 1e6, color="black", alpha=0.3,
            linewidth=0.65, label="Realized daily loss")
    ax.set_title("One-day 99% VaR versus realized shock-only loss")
    ax.set_ylabel("USD millions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_backtested_99pct_var.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    unhedged = stress_df.loc[stress_df.strategy == "Unhedged"]
    ewma_stress = stress_df.loc[stress_df.strategy == "MinVar_EWMA"]
    xx = np.arange(len(unhedged))
    ax.bar(xx - 0.18, unhedged.exact_pnl_usd.to_numpy() / 1e6, width=0.36, label="Unhedged")
    ax.bar(xx + 0.18, ewma_stress.exact_pnl_usd.to_numpy() / 1e6, width=0.36, label="EWMA hedge")
    ax.set_xticks(xx)
    ax.set_xticklabels(unhedged.scenario, rotation=22, ha="right")
    ax.set_ylabel("USD millions")
    ax.set_title("Hypothetical yield-curve stress: current rebalanced positions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_stress_pnl.png", dpi=140)
    plt.close(fig)

    print("=" * 80)
    print("PHASE 7A — REAL TREASURY ZERO-COUPON PORTFOLIO RISK & HEDGING")
    print("=" * 80)
    print(f"Data: {DATA_FILE}")
    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} changes)")
    print(f"OOS: {test.index[0].date()} to {test.index[-1].date()} ({len(test)} changes)")
    print("Portfolio: $100m daily rebalanced market value across 2Y, 5Y, 10Y, 20Y, 30Y")
    print(f"Initial dollar DV01: ${d.sum():,.0f}/bp; parallel modified duration: "
          f"{np.dot(BASE_MV, TENORS) / BASE_MV.sum():.2f} years")
    print("\nOOS PARAMETRIC NORMAL VaR BACKTEST (daily):")
    print(var_summary.round(3).to_string(index=False))
    print("\nOOS HEDGE COMPARISON (daily SHOCK-ONLY P&L):")
    print(hedge_summary.round(2).to_string(index=False))
    print("\nAS-OF-LATEST STRESS P&L (USD):")
    print(stress_df.pivot(index="scenario", columns="strategy", values="exact_pnl_usd").round(0).to_string())
    print("\nLIMITATIONS: synthetic zero-coupon positions built on REAL fitted Treasury zero yields;")
    print("no coupons, accrued interest, roll-down, financing, futures CTD/basis, or transaction costs.")
    print("Rate shocks evaluated at constant tenors; positions are reset to fixed dollar values daily.")
    print("Hedges in 2Y/10Y/30Y zeros are hypothetical and can involve large short market values.")
    print("Normal VaR uses linear DV01 approximation; realized P&L uses exact zero-bond shock repricing.")
    print("Kupiec p-values are approximate, unconditional only; 99% tail has few observations.")
    print("No actual bond trade backtest and no proof of future risk model superiority.")
    print(f"\nSaved six CSVs and three charts to {OUTPUT}")


if __name__ == "__main__":
    main()
