"""PROJECT 2 — PHASE 1: U.S. Treasury yield-curve data and diagnostics.

Goal
----
Prepare a reproducible empirical foundation for Dynamic Nelson-Siegel, AFNS,
G2++, and HJM.  The Gürkaynak-Sack-Wright Federal Reserve research series contain
continuously compounded fitted ZERO yields (SVENYxx) and INSTANTANEOUS
FORWARD rates (SVENFxx), at integer maturities from 1Y to 30Y.

Run in VS Code (Python 3.9+):
    python -m pip install numpy pandas matplotlib scikit-learn requests
    python project2_phase1_treasury_data.py

If Fed blocks the automatic download:
    1) Open https://www.federalreserve.gov/data/nominal-yield-curve.htm
    2) Click "CSV" and save feds200628.csv next to this script.
    3) Run: python project2_phase1_treasury_data.py --local-csv feds200628.csv

By default, the sample begins in January 2010. Output files are placed in
project2_phase1_outputs next to this script. This is descriptive analysis,
NOT an out-of-sample prediction or a risk-neutral derivatives calibration.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Charts save correctly in a normal terminal / VS Code.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA


SOURCE_PAGE = "https://www.federalreserve.gov/data/nominal-yield-curve.htm"
SOURCE_CSV = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
ROOT = Path(__file__).resolve().parent
YEARS = list(range(1, 31))
ZERO_COLS = [f"SVENY{n:02d}" for n in YEARS]
FWD_COLS = [f"SVENF{n:02d}" for n in YEARS]
TENOR_NAMES = [f"{n}Y" for n in YEARS]
PCA_YEARS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
PCA_TENORS = [f"{n}Y" for n in PCA_YEARS]


def fetch_source(local_csv: str | None, output_dir: Path) -> tuple[bytes, str]:
    """Download the official CSV; cache raw bytes for reproducibility and offline use."""
    raw = output_dir / "feds200628_raw.csv"
    if local_csv is not None:
        local_file = Path(local_csv).expanduser().resolve()
        if not local_file.exists():
            raise FileNotFoundError(f"Local CSV does not exist: {local_file}")
        payload = local_file.read_bytes()
        origin = f"local file: {local_file}"
    else:
        try:
            print("Downloading the Federal Reserve nominal yield-curve CSV...")
            response = requests.get(
                SOURCE_CSV,
                timeout=55,
                headers={"User-Agent": "Mozilla/5.0 (academic fixed income research)"},
            )
            response.raise_for_status()
            payload = response.content
            origin = SOURCE_CSV
        except requests.RequestException as exc:
            if not raw.exists():
                raise RuntimeError(
                    "Fed CSV download failed and there is no cached copy.\n"
                    f"Open {SOURCE_PAGE}, download the CSV, and run:\n"
                    "python project2_phase1_treasury_data.py --local-csv feds200628.csv"
                ) from exc
            print(f"Download failed ({exc}). Using the cached raw CSV instead.")
            payload = raw.read_bytes()
            origin = f"cached file: {raw}"

    # Never silently replace a good cached dataset with an HTML error page.
    header_probe = payload[:12000].decode("utf-8-sig", errors="replace")
    if "SVENY10" not in header_probe or "SVENF10" not in header_probe:
        raise ValueError("File does not look like the official SVENY/SVENF CSV.")
    raw.write_bytes(payload)
    return payload, origin


def read_fed_csv(payload: bytes) -> pd.DataFrame:
    """Skip the Fed's possible descriptive preamble and parse day-month-year dates."""
    decoded = payload.decode("utf-8-sig", errors="replace")
    lines = decoded.splitlines()
    first_header = None
    for i, line in enumerate(lines):
        fields = [value.strip().strip('"') for value in line.split(",")]
        if fields and fields[0] == "Date" and "SVENY10" in fields and "SVENF10" in fields:
            first_header = i
            break
    if first_header is None:
        raise ValueError("Cannot locate the 'Date,...,SVENY10,...,SVENF10' header.")

    data = pd.read_csv(BytesIO("\n".join(lines[first_header:]).encode()), low_memory=False)
    data.columns = data.columns.astype(str).str.strip().str.strip('"')
    needed = ZERO_COLS + FWD_COLS
    missing = sorted(set(needed) - set(data.columns))
    if missing:
        raise ValueError(f"Expected 1Y–30Y spot and forward columns are missing: {missing}")

    # Official nominal-curve dates are day-month-year, e.g. 16-10-2026.
    raw_dates = data["Date"].astype(str).str.strip()
    dates = pd.to_datetime(raw_dates, format="%d-%m-%Y", errors="coerce")
    if dates.notna().sum() == 0:
        # Alternative for an older / manually transformed vintage.
        dates = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    if dates.notna().sum() == 0:
        raise ValueError("Could not parse dates as DD-MM-YYYY or YYYY-MM-DD.")

    values = data[needed].apply(pd.to_numeric, errors="coerce")
    values = values.replace([-999.99, -9999, 999.99, 9999], np.nan)
    # A non-finite number must never be passed into PCA or discounting.
    values = values.replace([np.inf, -np.inf], np.nan)
    values.index = pd.DatetimeIndex(dates, name="Date")
    values = values[values.index.notna()].sort_index()
    values = values[~values.index.duplicated(keep="last")]
    return values


def check_curve_values(all_rates: pd.DataFrame, output_dir: Path):
    """Quarantine implausible yields and report all changed cells.

    Broad bounds are safeguards against corrupt CSV values, NOT estimates of
    the actual historical range. Review any flagged cells before using the data
    in empirical work. All rates here are percentages, not decimals.
    """
    bounds = {**{c: (-10.0, 30.0) for c in ZERO_COLS},
              **{c: (-20.0, 50.0) for c in FWD_COLS}}
    issue_rows = []
    cleaned = all_rates.copy()
    for col, (lower, upper) in bounds.items():
        series = cleaned[col]
        invalid = series.notna() & (~np.isfinite(series) | (series < lower) | (series > upper))
        for date, value in series.loc[invalid].items():
            issue_rows.append({"Date": date, "series": col, "value_pct": value,
                               "allowed_min_pct": lower, "allowed_max_pct": upper})
        cleaned.loc[invalid, col] = np.nan
    issues = pd.DataFrame(issue_rows, columns=[
        "Date", "series", "value_pct", "allowed_min_pct", "allowed_max_pct"
    ])
    issues.to_csv(output_dir / "flagged_rate_values.csv", index=False)
    if not issues.empty:
        print(f"WARNING: flagged {len(issues)} implausible rate values. "
              f"See {output_dir / 'flagged_rate_values.csv'}")
        print(issues.head(8).to_string(index=False))
    return cleaned, issues


def make_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Use the last observed complete daily curve of each *completed* month."""
    monthly = daily.resample("ME").last().dropna(how="any")
    # Prevent any accidentally incomplete current month from entering the PCA.
    today_month = pd.Timestamp.today().to_period("M")
    return monthly[monthly.index.to_period("M") < today_month]


def pca_on_changes(monthly_zeros: pd.DataFrame):
    """PCA of monthly yield changes (bps), with NO standardization by tenor."""
    changes_bps = monthly_zeros[PCA_TENORS].diff().dropna() * 100
    if len(changes_bps) < 24:
        raise ValueError("Too few monthly observations for PCA diagnostics.")
    if not np.isfinite(changes_bps.to_numpy(dtype=float)).all():
        raise ValueError("PCA input has non-finite yield changes. Inspect the CSV.")
    # sklearn's automatic solver may form X.T @ X (covariance_eigh).  The full
    # SVD avoids this avoidable numerical instability on some NumPy/BLAS builds.
    pca = PCA(n_components=3, svd_solver="full")
    scores = pca.fit_transform(changes_bps.to_numpy(dtype=float))
    if not np.isfinite(scores).all() or not np.isfinite(pca.explained_variance_ratio_).all():
        raise ArithmeticError("Non-finite PCA result; do not use estimated factors.")
    explained = pd.Series(
        pca.explained_variance_ratio_, index=["PC1", "PC2", "PC3"],
        name="fraction_of_variance",
    )
    loadings = pd.DataFrame(
        pca.components_.T, index=PCA_TENORS, columns=["PC1", "PC2", "PC3"]
    )
    score_frame = pd.DataFrame(
        scores, index=changes_bps.index, columns=["PC1", "PC2", "PC3"]
    )
    return explained, loadings, score_frame


def export_figure(target: Path):
    plt.tight_layout()
    plt.savefig(target, dpi=175, bbox_inches="tight")
    plt.close()


def plot_results(daily_z: pd.DataFrame, daily_f: pd.DataFrame,
                 monthly_z: pd.DataFrame, explained: pd.Series,
                 loadings: pd.DataFrame, output_dir: Path) -> None:
    curve_date = daily_z.index[-1].strftime("%Y-%m-%d")
    plt.figure(figsize=(10, 5))
    plt.plot(YEARS, daily_z.iloc[-1], label="Zero-coupon yield", linewidth=2)
    plt.plot(YEARS, daily_f.iloc[-1], label="Instantaneous forward rate", linewidth=2)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Continuously compounded rate (%)")
    plt.title(f"Treasury zero and forward curves | {curve_date}")
    plt.legend()
    plt.grid(alpha=0.25)
    export_figure(output_dir / "01_latest_zero_and_forward_curves.png")

    plt.figure(figsize=(11, 5))
    for tenor in ["2Y", "5Y", "10Y", "20Y", "30Y"]:
        plt.plot(monthly_z.index, monthly_z[tenor], label=tenor)
    plt.title("Treasury fitted zero-coupon yields — completed calendar months")
    plt.ylabel("Yield (%)")
    plt.legend(ncol=5)
    plt.grid(alpha=0.25)
    export_figure(output_dir / "02_historical_yields.png")

    plt.figure(figsize=(11, 5))
    plt.plot(monthly_z.index, 100 * (monthly_z["10Y"] - monthly_z["2Y"]), label="10Y minus 2Y")
    plt.plot(monthly_z.index, 100 * (monthly_z["30Y"] - monthly_z["10Y"]), label="30Y minus 10Y")
    plt.axhline(0, color="gray", linewidth=1)
    plt.ylabel("Zero-coupon yield spread (bps)")
    plt.title("Treasury curve slopes and inversions")
    plt.legend()
    plt.grid(alpha=0.25)
    export_figure(output_dir / "03_curve_slopes.png")

    plt.figure(figsize=(9, 5))
    for name in loadings:
        plt.plot(PCA_YEARS, loadings[name], marker="o", label=name)
    plt.axhline(0, color="gray", linewidth=1)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Principal component loading (sign arbitrary)")
    plt.title("PCA of monthly changes in Treasury zero yields (bps)")
    plt.legend()
    plt.grid(alpha=0.25)
    export_figure(output_dir / "04_pca_loadings.png")

    plt.figure(figsize=(7, 4.5))
    plt.bar(["PC1", "PC2", "PC3"], explained.to_numpy() * 100)
    plt.ylabel("Explained variance (%)")
    plt.title("First three principal components")
    plt.grid(alpha=0.2, axis="y")
    export_figure(output_dir / "05_pca_explained_variance.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv", default=None, help="Path to manually downloaded Fed CSV")
    parser.add_argument("--start", default="2010-01-01", help="Inclusive sample start, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Inclusive sample end, YYYY-MM-DD")
    parser.add_argument("--output", default=str(ROOT / "project2_phase1_outputs"), help="Output folder")
    args = parser.parse_args()

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload, origin = fetch_source(args.local_csv, out)
    all_rates = read_fed_csv(payload)
    all_rates = all_rates.loc[args.start:args.end]
    if all_rates.empty:
        raise ValueError("No data found in the requested date range.")

    all_rates, flagged_values = check_curve_values(all_rates, out)
    missing_rows = all_rates.isna().any(axis=1)
    zeros = all_rates[ZERO_COLS].rename(columns=dict(zip(ZERO_COLS, TENOR_NAMES)))
    forwards = all_rates[FWD_COLS].rename(columns=dict(zip(FWD_COLS, TENOR_NAMES)))
    zero_complete = zeros.notna().all(axis=1)
    forward_complete = forwards.notna().all(axis=1)
    both_complete = zero_complete & forward_complete
    if both_complete.sum() < 250:
        raise ValueError(f"Only {both_complete.sum()} complete daily curves; require at least 250.")

    daily_z = zeros.loc[both_complete]
    daily_f = forwards.loc[both_complete]
    monthly_z = make_monthly(daily_z)
    monthly_f = make_monthly(daily_f)
    if len(monthly_z) < 36:
        raise ValueError("Fewer than 36 completed months; expand your date window.")
    # Continuously compounded zero yields (percent) -> discount factors.
    tenors = np.asarray(YEARS, dtype=float)[None, :]
    discounts = pd.DataFrame(
        np.exp(-(daily_z.to_numpy() / 100.0) * tenors),
        index=daily_z.index, columns=TENOR_NAMES,
    )

    # Guard against large, finite observation glitches that can dominate PCA.
    monthly_changes = monthly_z[PCA_TENORS].diff().dropna() * 100.0
    largest_move_bps = monthly_changes.abs().max().max()
    suspicious_moves = monthly_changes.where(monthly_changes.abs() > 300.0).stack()
    suspicious_moves.to_csv(out / "monthly_moves_over_300bp.csv", header=["change_bps"])
    if len(suspicious_moves):
        print("WARNING: Monthly rate movements above 300 bp detected. "
              "Review monthly_moves_over_300bp.csv before relying on PCA.")
    explained, loadings, scores = pca_on_changes(monthly_z)
    export_tables = {
        "daily_zero_yields_pct.csv": daily_z,
        "daily_instantaneous_forwards_pct.csv": daily_f,
        "monthly_zero_yields_pct.csv": monthly_z,
        "monthly_instantaneous_forwards_pct.csv": monthly_f,
        "daily_discount_factors.csv": discounts,
        "pca_explained_variance.csv": explained,
        "pca_loadings.csv": loadings,
        "pca_factor_scores_monthly.csv": scores,
    }
    for filename, table in export_tables.items():
        table.to_csv(out / filename, float_format="%.8f")
    plot_results(daily_z, daily_f, monthly_z, explained, loadings, out)

    print("\n" + "=" * 68)
    print("PHASE 1 — DATA VALIDATION / PCA REPORT")
    print("=" * 68)
    print(f"Raw source: {origin}")
    print(f"SHA-256 source fingerprint: {sha256(payload).hexdigest()}")
    print(f"Requested sample: {args.start} to {args.end or 'latest available'}")
    print(f"First retained observation: {daily_z.index.min().date()}")
    print(f"Last retained observation:  {daily_z.index.max().date()}")
    print(f"Daily candidate dates in range: {len(all_rates):,}")
    print(f"Rows with at least one missing curve value: {missing_rows.sum():,}")
    print(f"Out-of-range data values flagged: {len(flagged_values):,}")
    print(f"Largest absolute monthly yield change: {largest_move_bps:.2f} bp")
    print(f"Monthly changes greater than 300 bp: {len(suspicious_moves):,}")
    print(f"PCA solver: full SVD (not covariance eigendecomposition)")
    print(f"Complete aligned daily observations: {len(daily_z):,}")
    print(f"Completed calendar month observations: {len(monthly_z):,}")
    print("Zero and forward yields are continuously compounded rates in PERCENT.")
    for name, fraction in explained.items():
        print(f"{name} explained variance: {fraction * 100:.2f}%")
    print(f"Total first-three explained variance: {explained.sum() * 100:.2f}%")
    print(f"\nSaved 8 analysis CSVs, 2 diagnostics CSVs, 5 figures, and a text audit to:\n{out}")
    report = (
        f"Source: {SOURCE_PAGE}\nOrigin: {origin}\n"
        f"Raw CSV SHA-256: {sha256(payload).hexdigest()}\n"
        f"Start: {daily_z.index.min().date()}\nEnd: {daily_z.index.max().date()}\n"
        f"Candidate rows: {len(all_rates)}\nRows with missing rates: {missing_rows.sum()}\n"
        f"Out-of-range data values flagged: {len(flagged_values)}\n"
        f"Largest monthly yield change in abs(bps): {largest_move_bps:.4f}\n"
        f"Monthly changes above 300 bps: {len(suspicious_moves)}\n"
        "PCA solver: full SVD\n"
        f"Complete daily observations: {len(daily_z)}\nCompleted months: {len(monthly_z)}\n"
        f"Explained variance PC1: {explained.iloc[0] * 100:.3f}%\n"
        f"Explained variance PC2: {explained.iloc[1] * 100:.3f}%\n"
        f"Explained variance PC3: {explained.iloc[2] * 100:.3f}%\n"
        "\nCAVEATS:\n"
        "1. GSW curves are fitted to off-the-run Treasury note/bond prices; not raw quotes.\n"
        "2. The dataset can be revised and is published with a reporting delay.\n"
        "3. Published fitted yields are not independent market evidence for Svensson validation.\n"
        "4. Full-sample PCA is descriptive; refit using training data for forecasts.\n"
        "5. Historical yield changes are physical-measure observations. No derivative\n"
        "   implied-volatility surface or risk-neutral G2++/HJM calibration is supplied here.\n"
        "6. Annual-maturity grid starts at 1Y; it does not directly observe today's\n"
        "   overnight/SOFR short rate or the t=0 instantaneous forward rate.\n"
        "7. Do not treat fitted off-the-run Treasury yields as SOFR swap discount rates.\n"
        "8. Implausible rate values are quarantined, not imputed; see flagged_rate_values.csv.\n"
        "9. Check monthly_moves_over_300bp.csv for anomalous finite yield changes.\n"
    )
    (out / "phase1_data_audit.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
