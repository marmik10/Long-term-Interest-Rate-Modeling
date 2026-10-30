
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PHASE 6D — TIME-VARYING HJM FORWARD-RATE COVARIANCE
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "project2_phase1_outputs" / "daily_instantaneous_forwards_pct.csv"
OUT = ROOT / "project2_phase6d_outputs"
OUT.mkdir(exist_ok=True)

TENORS = [f"{i}Y" for i in range(1, 31)]
ANNUAL_DAYS = 252

# 1. LOAD REAL TREASURY DATA

df = pd.read_csv(DATA, index_col=0, parse_dates=True)
df = df.sort_index()

rates = df[TENORS].astype(float) / 100
changes = rates.diff().dropna()

X = changes.to_numpy()
dates = changes.index

print("Loaded", len(changes), "daily forward-rate changes")

# 2. COVARIANCE ESTIMATORS

def covariance(data):
    return np.cov(data, rowvar=False) * ANNUAL_DAYS


def ewma_covariance(data, decay=0.94):
    cov = np.cov(data[:252], rowvar=False)
    mean = data[:252].mean(axis=0)

    for row in data[252:]:
        change = row - mean
        cov = decay * cov + (1 - decay) * np.outer(change, change)

    return cov * ANNUAL_DAYS


def pca_covariance(data, factors=3):
    cov = covariance(data)
    values, vectors = np.linalg.eigh(cov)

    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    result = np.zeros_like(cov)

    for i in range(factors):
        result += max(values[i], 0) * np.outer(
            vectors[:, i], vectors[:, i]
        )

    # Retain residual diagonal variances
    residual = np.maximum(np.diag(cov) - np.diag(result), 0)
    result += np.diag(residual)

    return result


def covariance_error(predicted, actual):
    return (
        np.linalg.norm(predicted - actual, "fro")
        / np.linalg.norm(actual, "fro")
        * 100
    )


def shape_error(predicted, actual):
    p = predicted / np.linalg.norm(predicted, "fro")
    a = actual / np.linalg.norm(actual, "fro")
    return np.linalg.norm(p - a, "fro") * 100


# 3. INITIAL TRAINING: 2010–2017

train = changes.loc[: "2017-12-31"]

frozen_cov = covariance(train.to_numpy())

print("Training observations:", len(train))

# 4. MONTHLY WALK-FORWARD EVALUATION

results = []

months = changes.index.to_period("M").unique()

for month in months:
    if month < pd.Period("2018-01", freq="M"):
        continue

    # Exclude incomplete final month
    if month == months[-1]:
        continue

    start = month.to_timestamp()

    history = changes.loc[changes.index < start]
    future = changes.loc[changes.index.to_period("M") == month]

    if len(history) < 252 or len(future) < 12:
        continue

    historical = history.to_numpy()
    realized = covariance(future.to_numpy())

    models = {
        "Frozen": frozen_cov,
        "Rolling 63d": covariance(historical[-63:]),
        "Rolling 252d": covariance(historical[-252:]),
        "EWMA 0.94": ewma_covariance(historical, 0.94),
        "EWMA 0.97": ewma_covariance(historical, 0.97),
        "Rolling PCA3": pca_covariance(historical[-252:], 3),
    }

    for name, predicted in models.items():
        error = covariance_error(predicted, realized)
        shape = shape_error(predicted, realized)

        variance_ratio = (
            np.trace(predicted) / np.trace(realized)
        )

        results.append({
            "Month": str(month),
            "Model": name,
            "Covariance_Error_Pct": error,
            "Shape_Error_Pct": shape,
            "Variance_Ratio": variance_ratio,
        })

# 5. MODEL COMPARISON

results = pd.DataFrame(results)

summary = (
    results.groupby("Model")
    .agg(
        Mean_Error=("Covariance_Error_Pct", "mean"),
        Median_Error=("Covariance_Error_Pct", "median"),
        Mean_Shape_Error=("Shape_Error_Pct", "mean"),
        Median_Variance_Ratio=("Variance_Ratio", "median"),
        Months=("Month", "count"),
    )
    .sort_values("Mean_Error")
)

print("\n" + "=" * 75)
print("PHASE 6D — REAL TREASURY ROLLING COVARIANCE RESULTS")
print("=" * 75)
print("\nOUT-OF-SAMPLE MODEL COMPARISON")
print(summary.round(3).to_string())

# 6. REGIME COMPARISON

results["Year"] = results["Month"].str[:4].astype(int)

def regime(year):
    if year <= 2019:
        return "2018-2019"
    elif year <= 2021:
        return "2020-2021"
    elif year <= 2023:
        return "2022-2023"
    else:
        return "2024-2026"

results["Regime"] = results["Year"].apply(regime)

regimes = (
    results.groupby(["Regime", "Model"])
    ["Covariance_Error_Pct"]
    .mean()
    .unstack()
)

print("\nREGIME ANALYSIS")
print(regimes.round(2).to_string())

# 7. SAVE RESULTS

results.to_csv(OUT / "phase6d_monthly_results.csv", index=False)
summary.to_csv(OUT / "phase6d_model_summary.csv")
regimes.to_csv(OUT / "phase6d_regime_comparison.csv")

# 8. VISUALIZATION — AVERAGE FORECAST ERROR

plt.figure(figsize=(11, 5))
summary["Mean_Error"].plot(kind="bar")
plt.ylabel("Covariance Forecast Error (%)")
plt.title("Phase 6D — Out-of-Sample Covariance Comparison")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT / "01_model_comparison.png", dpi=150)
plt.close()

# 9. VISUALIZATION — ERRORS THROUGH TIME

plt.figure(figsize=(12, 5))

for model in results["Model"].unique():
    subset = results[results["Model"] == model]
    plt.plot(
        pd.to_datetime(subset["Month"]),
        subset["Covariance_Error_Pct"].rolling(6).mean(),
        label=model,
    )

plt.title("Rolling 6-Month Average Covariance Forecast Error")
plt.ylabel("Error (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "02_rolling_errors.png", dpi=150)
plt.close()

# 10. VISUALIZATION — VARIANCE RATIOS

plt.figure(figsize=(12, 5))

for model in ["Frozen", "Rolling 252d", "EWMA 0.94"]:
    subset = results[results["Model"] == model]
    plt.plot(
        pd.to_datetime(subset["Month"]),
        subset["Variance_Ratio"],
        label=model,
    )

plt.axhline(1, color="black", linestyle="--")
plt.title("Forecast / Realized Total Variance")
plt.ylabel("Variance Ratio")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "03_variance_ratio.png", dpi=150)
plt.close()

print("\nSaved CSV files and charts to:", OUT)
print("PHASE 6D COMPLETED")
