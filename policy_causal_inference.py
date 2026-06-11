"""
Advanced causal inference on public policy data using Python.

Research question:
What was the effect of mandatory stay-at-home orders on county-level workplace mobility
during the early COVID-19 period in the United States?

Method:
Difference-in-Differences with county fixed effects and date fixed effects, using CDC
county-day stay-at-home order data and Google's county mobility data.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


START_DATE = "2020-03-01"
END_DATE = "2020-06-30"
CDC_POLICY_URL = (
    "https://data.cdc.gov/resource/y2iy-8irm.csv"
    "?$select=state_tribe_territory,county_name,fips_state,fips_county,date,stay_at_home_order"
    "&$limit=2000000"
)
GOOGLE_MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"


@dataclass
class DidResult:
    name: str
    outcome: str
    treatment: str
    beta: float
    se: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_counties: int
    n_states: int


def load_policy_data(cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        print("Downloading CDC county stay-at-home order data...")
        df = pd.read_csv(CDC_POLICY_URL)
        df.to_csv(cache_path, index=False)
    else:
        print(f"Using cached CDC policy data at {cache_path}")
        df = pd.read_csv(cache_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.loc[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    df["fips_state"] = pd.to_numeric(df["fips_state"], errors="coerce")
    df["fips_county"] = pd.to_numeric(df["fips_county"], errors="coerce")
    df = df.dropna(subset=["fips_state", "fips_county"])
    df["county_fips"] = (df["fips_state"].astype(int) * 1000 + df["fips_county"].astype(int)).astype(int)

    order_text = df["stay_at_home_order"].fillna("")
    df["mandatory_broad"] = order_text.str.contains("Mandatory", case=False)
    df["mandatory_strict"] = order_text.eq("Mandatory for all individuals")

    return df.loc[
        :,
        [
            "county_fips",
            "date",
            "state_tribe_territory",
            "mandatory_broad",
            "mandatory_strict",
        ],
    ].drop_duplicates()


def load_mobility_data(cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        print("Downloading and filtering Google county mobility data...")
        chunks = []
        for chunk in pd.read_csv(
            GOOGLE_MOBILITY_URL,
            chunksize=250_000,
            parse_dates=["date"],
            low_memory=False,
        ):
            mask = (
                (chunk["country_region_code"] == "US")
                & chunk["sub_region_1"].notna()
                & chunk["sub_region_2"].notna()
                & chunk["census_fips_code"].notna()
                & (chunk["date"] >= START_DATE)
                & (chunk["date"] <= END_DATE)
            )
            filtered = chunk.loc[
                mask,
                [
                    "census_fips_code",
                    "sub_region_1",
                    "sub_region_2",
                    "date",
                    "workplaces_percent_change_from_baseline",
                    "residential_percent_change_from_baseline",
                ],
            ]
            chunks.append(filtered)

        df = pd.concat(chunks, ignore_index=True)
        df.to_csv(cache_path, index=False)
    else:
        print(f"Using cached Google mobility data at {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])

    df["county_fips"] = pd.to_numeric(df["census_fips_code"], errors="coerce").astype("Int64")
    df["workplaces_percent_change_from_baseline"] = pd.to_numeric(
        df["workplaces_percent_change_from_baseline"], errors="coerce"
    )
    df["residential_percent_change_from_baseline"] = pd.to_numeric(
        df["residential_percent_change_from_baseline"], errors="coerce"
    )
    df = df.dropna(subset=["county_fips"]).copy()
    df["county_fips"] = df["county_fips"].astype(int)

    return df.loc[
        :,
        [
            "county_fips",
            "sub_region_1",
            "sub_region_2",
            "date",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ],
    ]


def build_panel(policy_df: pd.DataFrame, mobility_df: pd.DataFrame) -> pd.DataFrame:
    panel = mobility_df.merge(policy_df, on=["county_fips", "date"], how="inner")
    panel["state_code"] = panel["county_fips"] // 1000
    return panel


def two_way_demean(
    df: pd.DataFrame,
    column: str,
    entity_col: str,
    time_col: str,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> np.ndarray:
    values = df[column].astype(float).to_numpy()
    residual = values.copy()
    overall_mean = values.mean()
    entity = df[entity_col]
    time = df[time_col]

    for _ in range(max_iter):
        old = residual.copy()
        residual = residual - pd.Series(residual).groupby(entity).transform("mean").to_numpy()
        residual = residual - pd.Series(residual).groupby(time).transform("mean").to_numpy()
        residual = residual + overall_mean
        if np.max(np.abs(residual - old)) < tol:
            break

    return residual


def run_did(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    result_name: str,
) -> DidResult:
    work = df.dropna(subset=[outcome_col]).copy().reset_index(drop=True)
    y_tilde = two_way_demean(work, outcome_col, "county_fips", "date")
    d_tilde = two_way_demean(work, treatment_col, "county_fips", "date")

    xx = float(np.dot(d_tilde, d_tilde))
    beta = float(np.dot(d_tilde, y_tilde) / xx)
    residual = y_tilde - beta * d_tilde

    clusters = work["state_code"].to_numpy()
    unique_clusters = np.unique(clusters)
    meat = 0.0
    for cluster in unique_clusters:
        idx = clusters == cluster
        meat += float(np.dot(d_tilde[idx], residual[idx]) ** 2)

    g = len(unique_clusters)
    n = len(work)
    variance = (g / (g - 1)) * meat / (xx**2)
    se = sqrt(variance)
    t_stat = beta / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=g - 1)
    critical = stats.t.ppf(0.975, df=g - 1)

    return DidResult(
        name=result_name,
        outcome=outcome_col,
        treatment=treatment_col,
        beta=beta,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=beta - critical * se,
        ci_high=beta + critical * se,
        n_obs=n,
        n_counties=int(work["county_fips"].nunique()),
        n_states=int(work["state_code"].nunique()),
    )


def summarize_results(results: list[DidResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": r.name,
                "outcome": r.outcome,
                "treatment": r.treatment,
                "estimate": r.beta,
                "std_error": r.se,
                "p_value": r.p_value,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "n_obs": r.n_obs,
                "n_counties": r.n_counties,
                "n_states": r.n_states,
            }
            for r in results
        ]
    )


def build_event_profile(panel: pd.DataFrame) -> pd.DataFrame:
    treated = panel.loc[panel["mandatory_broad"]].groupby("county_fips", as_index=False)["date"].min()
    treated = treated.rename(columns={"date": "first_treat_date"})

    event_panel = panel.merge(treated, on="county_fips", how="inner")
    event_panel["event_day"] = (
        event_panel["date"] - event_panel["first_treat_date"]
    ).dt.days
    event_panel = event_panel.loc[
        event_panel["event_day"].between(-21, 21)
        & event_panel["workplaces_percent_change_from_baseline"].notna()
    ].copy()

    profile = (
        event_panel.groupby("event_day", as_index=False)
        .agg(
            mean_workplace_change=("workplaces_percent_change_from_baseline", "mean"),
            county_days=("county_fips", "size"),
        )
        .sort_values("event_day")
    )
    return profile


def create_visualization(results: list[DidResult], event_profile: pd.DataFrame, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    labels = [r.name for r in results]
    estimates = [r.beta for r in results]
    lower = [r.beta - r.ci_low for r in results]
    upper = [r.ci_high - r.beta for r in results]
    y_pos = np.arange(len(labels))

    axes[0].errorbar(
        estimates,
        y_pos,
        xerr=[lower, upper],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
    )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Difference-in-Differences Estimates")
    axes[0].set_xlabel("Estimated treatment effect")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].plot(
        event_profile["event_day"],
        event_profile["mean_workplace_change"],
        color="#d62728",
        linewidth=2,
    )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Average Workplace Mobility Around First Mandatory Order")
    axes[1].set_xlabel("Days relative to first mandatory order")
    axes[1].set_ylabel("Workplace mobility change from baseline")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(
    pdf_path: Path,
    results: list[DidResult],
    result_table: pd.DataFrame,
    event_profile: pd.DataFrame,
    figure_path: Path,
    panel: pd.DataFrame,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    main_result = results[0]
    strict_result = results[1]
    residential_result = results[2]

    sample_summary = (
        f"Final workplace-mobility panel: {main_result.n_obs:,} county-day observations, "
        f"{main_result.n_counties:,} counties, {main_result.n_states} states, "
        f"from {panel['date'].min().date()} to {panel['date'].max().date()}."
    )

    methodology = (
        "I used a Difference-in-Differences design with county fixed effects and date fixed "
        "effects. The treatment variable is whether a county-day was under a mandatory "
        "stay-at-home order according to the CDC policy dataset. The main outcome is "
        "Google's workplace mobility change from baseline. Standard errors are clustered at "
        "the state level."
    )

    findings = (
        f"The main estimate is {main_result.beta:.2f} percentage points "
        f"(95% CI {main_result.ci_low:.2f} to {main_result.ci_high:.2f}, "
        f"p-value {main_result.p_value:.4g}). This means mandatory orders are associated "
        "with a large additional decline in workplace mobility after accounting for county "
        "and date fixed effects. The stricter treatment definition produces an estimate of "
        f"{strict_result.beta:.2f}, which points in the same direction and is slightly "
        "larger in magnitude. Using residential mobility as an alternative outcome gives an "
        f"estimate of {residential_result.beta:.2f}, which is also theory-consistent because "
        "staying home should increase residential presence."
    )

    caveats = (
        "The design is appropriate for policy timing in panel data, but this setting still "
        "has limitations. Mobility was already changing rapidly in March 2020, and some "
        "behavior likely shifted before formal orders. So the estimate should be read as the "
        "incremental effect associated with an active mandatory order, not as a perfectly "
        "clean laboratory-style causal effect."
    )

    policy_implications = (
        "From a policy standpoint, the results suggest that mandatory stay-at-home orders "
        "meaningfully changed behavior during the early pandemic. For decision-makers, that "
        "matters because it shows the orders were not purely symbolic: they were associated "
        "with a measurable reduction in workplace activity and a corresponding increase in "
        "residential stay patterns."
    )

    wrapped_blocks = [
        ("Research question", "Did mandatory stay-at-home orders causally reduce workplace mobility?"),
        ("Data", "CDC county-day stay-at-home orders merged with Google's county mobility data."),
        ("Method", methodology),
        ("Sample", sample_summary),
        ("Main findings", findings),
        ("Caveats", caveats),
        ("Policy implications", policy_implications),
        (
            "Data sources",
            "CDC policy data: https://data.cdc.gov/Policy-Surveillance/U-S-State-and-Territorial-Stay-At-Home-Orders-Marc/y2iy-8irm | "
            "Google mobility data: https://www.google.com/covid19/mobility/",
        ),
    ]

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        y = 0.97
        fig.text(0.05, y, "Causal Inference Report: Stay-at-Home Orders and Mobility", fontsize=16, weight="bold")
        y -= 0.05

        for heading, body in wrapped_blocks:
            fig.text(0.05, y, heading, fontsize=11, weight="bold")
            y -= 0.025
            wrapped = textwrap.fill(body, width=100)
            fig.text(0.05, y, wrapped, fontsize=10, va="top")
            y -= 0.06 + (wrapped.count("\n") * 0.018)

        y -= 0.01
        fig.text(0.05, y, "Model summary", fontsize=11, weight="bold")
        y -= 0.025
        display_table = result_table.loc[:, ["model", "estimate", "std_error", "p_value"]].copy()
        display_table["estimate"] = display_table["estimate"].map(lambda x: f"{x:.2f}")
        display_table["std_error"] = display_table["std_error"].map(lambda x: f"{x:.2f}")
        display_table["p_value"] = display_table["p_value"].map(lambda x: f"{x:.4g}")
        table = plt.table(
            cellText=display_table.values,
            colLabels=display_table.columns,
            cellLoc="center",
            colLoc="center",
            bbox=[0.05, y - 0.18, 0.9, 0.16],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig2, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        image = plt.imread(figure_path)
        ax.imshow(image)
        ax.set_title("Estimated Effects and Event-Time Pattern", fontsize=14, pad=12)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)


def main() -> None:
    task_root = Path(__file__).resolve().parent
    data_dir = task_root / "data"
    output_png = task_root / "causal_effects.png"
    output_pdf = task_root / "causal_policy_report.pdf"
    output_csv = task_root / "did_results.csv"

    policy_df = load_policy_data(data_dir / "cdc_stay_home_orders.csv")
    mobility_df = load_mobility_data(data_dir / "google_us_county_mobility_mar_jun_2020.csv")
    panel = build_panel(policy_df, mobility_df)

    workplace_main = run_did(
        panel,
        outcome_col="workplaces_percent_change_from_baseline",
        treatment_col="mandatory_broad",
        result_name="Main DiD: workplace mobility, broad order",
    )
    workplace_strict = run_did(
        panel,
        outcome_col="workplaces_percent_change_from_baseline",
        treatment_col="mandatory_strict",
        result_name="Robustness: workplace mobility, strict order",
    )
    residential_alt = run_did(
        panel,
        outcome_col="residential_percent_change_from_baseline",
        treatment_col="mandatory_broad",
        result_name="Robustness: residential mobility, broad order",
    )

    results = [workplace_main, workplace_strict, residential_alt]
    result_table = summarize_results(results)
    result_table.to_csv(output_csv, index=False)

    event_profile = build_event_profile(panel)
    create_visualization(results, event_profile, output_png)
    write_pdf_report(output_pdf, results, result_table, event_profile, output_png, panel)

    print("\nCausal inference analysis complete.\n")
    print(result_table.to_string(index=False))
    print(f"\nFigure saved to: {output_png}")
    print(f"PDF report saved to: {output_pdf}")
    print(f"Result table saved to: {output_csv}")


if __name__ == "__main__":
    main()
