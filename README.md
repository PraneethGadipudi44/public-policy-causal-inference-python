# Public Policy Causal Inference in Python

This project studies the causal impact of mandatory stay-at-home orders on mobility during the early COVID-19 period in the United States.

The analysis uses public policy and mobility data, applies a Difference-in-Differences design, generates a visualization, and produces a short statistical report.

## Research Question

Did mandatory stay-at-home orders cause a measurable reduction in workplace mobility?

## Data Sources

This project uses two public datasets:

- CDC county-level stay-at-home order data  
  https://data.cdc.gov/Policy-Surveillance/U-S-State-and-Territorial-Stay-At-Home-Orders-Marc/y2iy-8irm

- Google COVID-19 Community Mobility Reports  
  https://www.google.com/covid19/mobility/

## Method

The main method used in this project is Difference-in-Differences (DiD).

The design compares county-level mobility before and after mandatory stay-at-home orders while controlling for:

- county fixed effects
- date fixed effects
- state-clustered standard errors

## Files in This Repository

- `policy_causal_inference.py` — main analysis script
- `causal_policy_report.pdf` — statistical report
- `causal_policy_report.tex` — LaTeX source for the report
- `causal_effects.png` — generated visualization
- `did_results.csv` — exported model results

## What the Script Does

The script:

- downloads or reuses public datasets
- cleans and merges the policy and mobility data
- builds a county-day panel
- estimates the main Difference-in-Differences model
- runs robustness checks
- exports a results table
- generates a visualization
- creates a report-ready output set

## Main Findings

From the current run:

- the main workplace mobility estimate is about **-50.17 percentage points**
- a stricter treatment definition gives a similar, slightly larger negative estimate
- residential mobility moves in the opposite direction, which is consistent with the policy mechanism

These results suggest that mandatory stay-at-home orders were associated with a large and statistically significant behavioral response.

## How to Run

Install dependencies:

```bash
pip install pandas numpy matplotlib scipy
