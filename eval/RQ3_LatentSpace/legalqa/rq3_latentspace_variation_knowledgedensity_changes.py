#!/usr/bin/env python3
"""
plot_signed_angles.py
=====================
Signed angular distance analysis between method feature vectors.

Overview
--------
For the composite feature vector (activation norm, activation std,
perplexity, confidence, token log-prob std), this module:

  1. Loads per-query neural pathway records from JSON result files.
  2. Constructs normalised feature vectors for each (model, method,
     query, domain) combination.
  3. Computes the signed angular displacement of each challenger method
     relative to the NLQ baseline, per query.
  4. Aggregates per-query angles into per-(model x domain x challenger)
     summary statistics including 95% bootstrap confidence intervals and
     Wilcoxon signed-rank significance tests against zero displacement.
  5. Produces a diverging dot plot showing mean signed angle with CI
     lines, per domain panel, with significance encoded visually.
  6. Writes summary statistics to CSV and LaTeX (booktabs) formats.

Feature vector
--------------
Composite only:
    activation_norm, activation_std, sequence_perplexity,
    confidence_score, token_logprob_std

Challenger methods
------------------
  sparql      -- SPARQL_NLQ (primary FLQ challenger)
  nlq_cot     -- NLQ_CoT (CoT reformulation)
  sparql_cot  -- SPARQL_CoT (hybrid CoT)
  Baseline    -- NLQ (reference; not shown as a challenger)

Signed angle convention
-----------------------
  Positive -- challenger feature vector is angularly displaced in the
              direction of increasing activation norm relative to NLQ.
  Negative -- displacement in the direction of decreasing norm.
  For single-feature vectors the signed difference replaces the
  angular calculation directly.

Statistical significance
------------------------
  Wilcoxon signed-rank test (two-sided) against the null hypothesis
  that the population of per-query signed angles has median zero.
  Minimum sample size: 10 non-zero observations.
  Significance thresholds:
    ***  p < 0.001
    **   p < 0.01
    *    p < 0.05
    ns   not significant
  95% bootstrap confidence intervals (2 000 resamples, seed 42).
  Cohen's d (mean / SD) reported as an effect-size estimate.

Outputs
-------
Written to ``--output_dir`` (default: ``./signed_angle_output/``):

  signed_angles_composite.csv     -- per-query signed angles
  signed_angle_report.csv         -- per-(model x domain x challenger)
                                     summary statistics
  signed_angle_report.tex         -- LaTeX table (booktabs)
  plots/signed_angle_composite_dot.png/.pdf
                                  -- diverging dot plot

Usage
-----
  python plot_signed_angles.py --input_dir ./json_results/ \\
                                --output_dir ./output/

  python plot_signed_angles.py --files a.json b.json \\
                                --output_dir ./output/

Dependencies
------------
  numpy, pandas, matplotlib, scipy
"""

from __future__ import annotations

import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Metadata for each challenger method: display label, colour, marker.
CHALLENGER_META: Dict[str, Dict] = {
    "sparql": {
        "display": "SPARQL_NLQ",
        "colour":  "#E07B39",
        "marker":  "o",
    },
    "nlq_cot": {
        "display": "NLQ_CoT",
        "colour":  "#2A9D8F",
        "marker":  "s",
    },
    "sparql_cot": {
        "display": "SPARQL_CoT",
        "colour":  "#8E6BBF",
        "marker":  "D",
    },
}

CHALLENGERS = list(CHALLENGER_META.keys())

#: The NLQ method is the reference throughout.
REFERENCE = "nlq"

#: Number of bootstrap resamples for CI estimation.
N_BOOT = 2000

#: Bootstrap confidence interval percentage.
BOOT_CI = 95.0

#: Output resolution in dots per inch.
DPI = 300

#: Clipping range for token log-probability values.
LOGPROB_RANGE = (-8.0, 0.0)

#: Canonical domain display order.
DOMAIN_ORDER = ["finance", "legal", "telco"]

#: Human-readable domain labels.
DOMAIN_LABELS = {
    "finance": "FiQA",
    "legal":   "LegalQA",
    "telco":   "TelcoQA",
}

#: The single feature set used in this analysis.
FEATURE_SET_ID       = "composite"
FEATURE_SET_NAME     = "Composite Feature Vector"
FEATURE_SET_SUBTITLE = (
    "Features: activation norm, activation std, perplexity, "
    "confidence, token log-prob std"
)
FEATURE_COLS = [
    "activation_norm",
    "activation_std",
    "sequence_perplexity",
    "confidence_score",
    "token_logprob_std",
]

#: Canonical method name aliases from various input spellings.
_METHOD_ALIASES: Dict[str, str] = {
    "nlq":                     "nlq",
    "natural_language":        "nlq",
    "nl":                      "nlq",
    "sparql":                  "sparql",
    "sparql_nlq":              "sparql",
    "nlq_cot":                 "nlq_cot",
    "nlq_chain":               "nlq_cot",
    "nlq_chain_of_thought":    "nlq_cot",
    "sparql_cot":              "sparql_cot",
    "sparql_nlq_cot":          "sparql_cot",
    "sparql_chain":            "sparql_cot",
    "sparql_chain_of_thought": "sparql_cot",
}

#: Accepted values of the ``analysis_type`` metadata field.
VALID_ANALYSIS_TYPES = {
    "neural_pathway_analyzer_v3_method_b_enabled",
    "legal_neural_pathway_analyzer_v3_method_b_enabled",
    "neural_pathway_analyzer_v3",
    "prompt_engineering_neural_pathway_analyzer_v3_method_b",
    "neural_pathway_analyzer",
}

ALL_METHODS = ["nlq", "nlq_cot", "sparql", "sparql_cot"]

#: Mapping from canonical feature name to source JSON field.
#: ``__computed__`` indicates the value is derived at load time.
SOURCE_FIELDS: Dict[str, str] = {
    "activation_norm":     "method_b_activation_norm",
    "activation_std":      "method_b_activation_std",
    "sequence_perplexity": "method_b_sequence_perplexity",
    "confidence_score":    "confidence_score",
    "token_logprob_std":   "__computed__",
}

#: Matplotlib rcParams applied to all figures.
_RC: Dict = {
    "figure.facecolor":  "white",
    "figure.dpi":        DPI,
    "axes.facecolor":    "#F9F9F9",
    "axes.edgecolor":    "#3A3A3A",
    "axes.linewidth":    1.2,
    "axes.axisbelow":    True,
    "axes.grid":         True,
    "grid.color":        "#D5D5D5",
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
    "grid.alpha":        0.28,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica",
                          "DejaVu Sans", "Arial"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#CCCCCC",
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _clean_name(s: str) -> str:
    """Return the canonical lowercase method key for *s*."""
    name = str(s).lower().replace("-", "_").replace(" ", "_").strip()
    for alias in sorted(_METHOD_ALIASES, key=len, reverse=True):
        if name == alias:
            return _METHOD_ALIASES[alias]
    return name


def _safe_float(v) -> float:
    """Convert *v* to float; return NaN on failure or non-finite value."""
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _norm_qid(v) -> str:
    """Normalise a query ID to a zero-padded six-digit string."""
    try:
        return f"{int(float(str(v).strip())):06d}"
    except Exception:
        return str(v).strip()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between vectors *a* and *b*."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if n < 1e-12 else float(np.dot(a, b) / n)


def _angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return the angular distance in degrees between vectors *a* and *b*."""
    return float(np.degrees(
        np.arccos(np.clip(_cosine_sim(a, b), -1.0, 1.0))
    ))


def _signed_angle(
    va: np.ndarray,
    vb: np.ndarray,
    delta_norm: float,
) -> float:
    """
    Compute the signed angular displacement of *vb* relative to *va*.

    For multi-feature vectors the unsigned angular distance is computed
    and the sign is determined by the direction of the activation norm
    change (``delta_norm``).  For single-feature vectors the signed
    scalar difference is returned directly.

    Parameters
    ----------
    va:
        Reference (NLQ) feature vector.
    vb:
        Challenger feature vector.
    delta_norm:
        Challenger activation norm minus NLQ activation norm.
        Used for sign determination only.
    """
    if len(va) == 1:
        return float(vb[0] - va[0])
    abs_angle = _angular_distance(va, vb)
    sign = 1.0 if (not np.isfinite(delta_norm) or delta_norm >= 0) else -1.0
    return sign * abs_angle


def _mask_padding(logprobs: list) -> np.ndarray:
    """
    Filter and clip a token log-probability list.

    Zero values (padding tokens) and non-finite values are removed;
    remaining values are clipped to LOGPROB_RANGE.
    """
    arr = np.array([_safe_float(x) for x in logprobs], dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr != 0.0]
    return np.clip(arr, LOGPROB_RANGE[0], LOGPROB_RANGE[1])


def _infer_domain(filepath: str) -> str:
    """Infer the domain label from the JSON file path."""
    p = str(filepath).lower()
    if any(k in p for k in ["fiqa", "finance", "fin_"]):
        return "finance"
    if any(k in p for k in ["legal", "legalqa", "law"]):
        return "legal"
    if any(k in p for k in ["telco", "telecom", "tele_"]):
        return "telco"
    return "unknown"


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    arr: np.ndarray,
    n_boot: int = N_BOOT,
    ci: float = BOOT_CI,
) -> Tuple[float, float]:
    """
    Return a bootstrap confidence interval for the mean of *arr*.

    Parameters
    ----------
    arr:
        1-D array of observed values.
    n_boot:
        Number of bootstrap resamples.
    ci:
        Confidence level as a percentage (e.g. 95.0).

    Returns
    -------
    tuple of (lower_bound, upper_bound).
    If *arr* has fewer than three elements the interval collapses to the
    sample mean.
    """
    if len(arr) < 3:
        m = float(np.mean(arr))
        return m, m
    rng   = np.random.default_rng(42)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo    = float(np.percentile(boots, (100.0 - ci) / 2))
    hi    = float(np.percentile(boots, 100.0 - (100.0 - ci) / 2))
    return lo, hi


def _cohens_d(arr: np.ndarray) -> float:
    """
    Return Cohen's d as a standardised effect-size estimate.

    Computed as mean(arr) / SD(arr, ddof=1).  Returns NaN if the array
    has fewer than two elements or has zero standard deviation.
    """
    if len(arr) < 2:
        return np.nan
    sd = np.std(arr, ddof=1)
    return float(np.mean(arr) / sd) if sd > 1e-12 else np.nan


def _wilcoxon_test(arr: np.ndarray) -> Tuple[float, float]:
    """
    Two-sided Wilcoxon signed-rank test against zero displacement.

    Only non-zero observations are used (zero differences carry no
    rank information under the standard test).  Returns (NaN, NaN) if
    fewer than 10 non-zero observations are available.

    Parameters
    ----------
    arr:
        1-D array of per-query signed angles.

    Returns
    -------
    (statistic, p_value) as floats.
    """
    arr_nz = arr[arr != 0.0]
    if len(arr_nz) < 10:
        return np.nan, np.nan
    try:
        stat, p = scipy_stats.wilcoxon(arr_nz, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def _sig_stars(p: float) -> str:
    """
    Return a significance string for *p*.

    Thresholds:
        ***  p < 0.001
        **   p < 0.01
        *    p < 0.05
        ns   not significant
    Returns an empty string if *p* is not finite.
    """
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    """Apply the shared rcParams to the current Matplotlib session."""
    plt.rcParams.update(_RC)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save *fig* as both PNG (300 dpi) and PDF to *path*."""
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), format="pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved: %s  |  %s", path.name, pdf_path.name)


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def discover_files(input_dir: str) -> List[str]:
    """Return all JSON files found recursively under *input_dir*."""
    return sorted(str(f) for f in Path(input_dir).glob("**/*.json"))


def _load_raw(filepath: str) -> Optional[Dict]:
    """
    Load and validate a single JSON result file.

    Returns the parsed dictionary if the file passes schema validation,
    or ``None`` if it cannot be read or fails validation.

    Validation checks:
    - File must be valid JSON.
    - Top-level keys ``metadata`` and ``results`` must be present.
    - ``metadata.analysis_type`` must be in VALID_ANALYSIS_TYPES.
    - ``metadata.query_mode`` must resolve to a known method via
      :func:`_clean_name`.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.error("Cannot read %s: %s", filepath, e)
        return None

    if "metadata" not in data or "results" not in data:
        return None

    meta  = data["metadata"]
    atype = meta.get("analysis_type", "")
    mode  = _clean_name(meta.get("query_mode", ""))

    if atype not in VALID_ANALYSIS_TYPES:
        log.warning(
            "Skipping %s: analysis_type='%s'", Path(filepath).name, atype
        )
        return None
    if mode not in ALL_METHODS:
        log.warning(
            "Skipping %s: query_mode='%s'", Path(filepath).name, mode
        )
        return None

    return data


def _extract_record(record: Dict) -> Optional[Dict]:
    """
    Extract feature values from a single result record.

    Returns a feature dict, or ``None`` if more than half the features
    are missing.  Missing individual features are imputed to zero after
    the majority-NaN check.

    Parameters
    ----------
    record:
        A single element from the ``results`` list in a JSON file.
    """
    feats: Dict = {}
    for feat, src in SOURCE_FIELDS.items():
        if src == "__computed__":
            arr        = _mask_padding(record.get("method_b_token_log_probs", []))
            feats[feat] = float(np.std(arr)) if len(arr) > 1 else np.nan
        else:
            feats[feat] = _safe_float(record.get(src))

    nan_count = sum(1 for v in feats.values() if np.isnan(v))
    if nan_count > len(feats) // 2:
        return None

    for k in feats:
        if np.isnan(feats[k]):
            feats[k] = 0.0

    return feats


def build_feature_frame(json_files: List[str]) -> pd.DataFrame:
    """
    Load all JSON result files and construct a feature data frame.

    Each row represents one (model, method, query_id, domain) observation
    with one column per feature.  Where multiple records share the same
    (model, method, query_id, domain) key the first is retained.

    Parameters
    ----------
    json_files:
        List of paths to JSON result files.

    Returns
    -------
    pandas.DataFrame
        Columns: model, method, query_id, domain, <feature columns>.

    Exits
    -----
    Calls ``sys.exit(1)`` if no valid records are found across all files.
    """
    rows: List[Dict] = []

    for fp in json_files:
        data = _load_raw(fp)
        if data is None:
            continue

        meta   = data["metadata"]
        model  = _clean_name(
            meta.get("model_name", Path(fp).stem.split("_")[0])
        )
        method = _clean_name(meta.get("query_mode", "unknown"))
        domain = _infer_domain(fp)
        n_ok   = 0

        for r in data["results"]:
            if r.get("Generated_Response", "").startswith("ERROR"):
                continue
            feats = _extract_record(r)
            if feats is None:
                continue
            qid = _norm_qid(r.get("Query_ID", -1))
            row = {
                "model":    model,
                "method":   method,
                "query_id": qid,
                "domain":   domain,
            }
            row.update(feats)
            rows.append(row)
            n_ok += 1

        if n_ok > 0:
            log.info(
                "%s: %d records  model=%s  method=%s  domain=%s",
                Path(fp).name, n_ok, model, method, domain,
            )

    if not rows:
        log.error("No valid records found.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.groupby(
        ["model", "method", "query_id", "domain"], as_index=False
    ).first()
    log.info(
        "Feature frame: %d rows  models=%s  domains=%s",
        len(df),
        sorted(df["model"].unique()),
        sorted(df["domain"].unique()),
    )
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def zscore_normalise(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    Apply global z-score normalisation to *feat_cols* in *df*.

    The mean and standard deviation are computed across all rows (i.e.
    globally, not per-method or per-domain) so that the feature scale is
    consistent across the comparison.

    Parameters
    ----------
    df:
        Input data frame containing *feat_cols*.
    feat_cols:
        Names of columns to normalise.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with *feat_cols* replaced by their z-scored values.
    """
    df = df.copy()
    X  = df[feat_cols].values.astype(float)
    mu = np.nanmean(X, axis=0)
    sd = np.where(np.nanstd(X, axis=0) < 1e-12, 1.0, np.nanstd(X, axis=0))
    df[feat_cols] = (X - mu) / sd
    return df


# ---------------------------------------------------------------------------
# Signed angle computation
# ---------------------------------------------------------------------------

def compute_signed_angles(
    feat_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute per-query signed angular displacements for all challengers
    relative to the NLQ baseline.

    For each (model, domain) group and each query ID present in both the
    NLQ and challenger subsets, the function:

    1. Retrieves the z-scored NLQ feature vector ``va``.
    2. Retrieves the z-scored challenger feature vector ``vb``.
    3. Computes :func:`_signed_angle(va, vb, delta_norm)`.

    Parameters
    ----------
    feat_df:
        Feature data frame as returned by :func:`build_feature_frame`.
    feature_cols:
        Feature columns to include in the angular calculation.

    Returns
    -------
    pandas.DataFrame
        Columns: model, domain, challenger, query_id, signed_angle.
    """
    norm_df  = zscore_normalise(feat_df, feature_cols)
    rows     = []
    grp_cols = ["model", "domain"]

    for keys, grp in norm_df.groupby(grp_cols):
        model, domain = keys

        nlq_sub = grp[grp["method"] == REFERENCE].set_index("query_id")
        if nlq_sub.empty:
            continue

        for ch in CHALLENGERS:
            ch_sub = grp[grp["method"] == ch].set_index("query_id")
            if ch_sub.empty:
                continue

            common = nlq_sub.index.intersection(ch_sub.index)
            if len(common) == 0:
                continue

            for qid in common:
                va = nlq_sub.loc[qid, feature_cols].values.astype(float)
                vb = ch_sub.loc[qid,  feature_cols].values.astype(float)

                norm_nlq = _safe_float(
                    nlq_sub.loc[qid, "activation_norm"]
                    if "activation_norm" in nlq_sub.columns else va[0]
                )
                norm_ch = _safe_float(
                    ch_sub.loc[qid, "activation_norm"]
                    if "activation_norm" in ch_sub.columns else vb[0]
                )
                delta_norm = (
                    norm_ch - norm_nlq
                    if (np.isfinite(norm_ch) and np.isfinite(norm_nlq))
                    else np.nan
                )

                rows.append({
                    "model":        model,
                    "domain":       domain,
                    "challenger":   ch,
                    "query_id":     qid,
                    "signed_angle": _signed_angle(va, vb, delta_norm),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_stats(angle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-query signed angles into per-(model x domain x
    challenger) summary statistics.

    For each group the following are computed:

    * Sample size (n).
    * Mean and median signed angle.
    * Sample standard deviation.
    * 95% bootstrap confidence interval for the mean
      (:func:`_bootstrap_ci`).
    * Wilcoxon signed-rank test statistic and p-value against zero
      displacement (:func:`_wilcoxon_test`).
    * Significance code (:func:`_sig_stars`).
    * Cohen's d effect size (:func:`_cohens_d`).
    * Percentage of queries with positive / negative displacement.

    Parameters
    ----------
    angle_df:
        Per-query signed angles as returned by
        :func:`compute_signed_angles`.

    Returns
    -------
    pandas.DataFrame
        One row per (model x domain x challenger).
    """
    rows = []

    for (model, domain, ch), sub in angle_df.groupby(
        ["model", "domain", "challenger"]
    ):
        arr = sub["signed_angle"].dropna().values.astype(float)
        if len(arr) == 0:
            continue

        mean_v   = float(np.mean(arr))
        median_v = float(np.median(arr))
        std_v    = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci_lo, ci_hi = _bootstrap_ci(arr)
        w_stat, w_p  = _wilcoxon_test(arr)
        d            = _cohens_d(arr)

        rows.append({
            "feature_set":         FEATURE_SET_ID,
            "feature_set_name":    FEATURE_SET_NAME,
            "model":               model,
            "domain":              domain,
            "challenger":          ch,
            "challenger_display":  CHALLENGER_META[ch]["display"],
            "n":                   len(arr),
            "mean_signed_angle":   round(mean_v,   4),
            "median_signed_angle": round(median_v, 4),
            "std_signed_angle":    round(std_v,    4),
            "ci95_lo":             round(ci_lo,    4),
            "ci95_hi":             round(ci_hi,    4),
            "wilcoxon_stat":       round(w_stat, 4) if np.isfinite(w_stat) else np.nan,
            "wilcoxon_p":          round(w_p,    6) if np.isfinite(w_p)    else np.nan,
            "wilcoxon_sig":        _sig_stars(w_p),
            "cohens_d":            round(d, 4) if np.isfinite(d) else np.nan,
            "pct_positive":        round(100.0 * (arr > 0).mean(), 2),
            "pct_negative":        round(100.0 * (arr < 0).mean(), 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

def _get_domains(stats_df: pd.DataFrame) -> List[str]:
    """Return domains in canonical order, with unknown domains appended."""
    if "domain" not in stats_df.columns:
        return ["all"]
    present = stats_df["domain"].unique()
    ordered = [d for d in DOMAIN_ORDER if d in present]
    rest    = [d for d in present if d not in DOMAIN_ORDER]
    return ordered + rest


def _ylim_from_df(
    stats_df: pd.DataFrame,
    pad_frac: float = 0.22,
) -> Tuple[float, float]:
    """
    Return symmetric y-axis limits derived from the CI bounds in
    *stats_df*, with a proportional padding margin.
    """
    lo = stats_df["ci95_lo"].dropna()
    hi = stats_df["ci95_hi"].dropna()
    if lo.empty:
        return -30.0, 30.0
    sym = max(abs(lo.min()), abs(hi.max())) * (1 + pad_frac) + 3.0
    return -sym, sym


# ---------------------------------------------------------------------------
# Diverging dot plot
# ---------------------------------------------------------------------------

def plot_dot(
    stats_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Diverging dot plot of mean signed angle with 95% bootstrap CI.

    Layout
    ------
    One panel per domain (column); models on the Y-axis; three dots per
    model (one per challenger) with CI lines.  Filled symbol = Wilcoxon
    significant; open symbol = not significant.  A legend strip is placed
    below all panels, entirely outside the data axes.

    Design notes
    ------------
    * X range is computed from all data including CI bounds with a fixed
      additional margin so that significance annotations are never clipped.
    * Model labels are drawn in figure coordinates to the left of the
      panels, avoiding tick-label space compression.
    * Alternating pale model bands aid row tracking without competing with
      the data.
    * CI end caps are drawn as explicit short vertical ticks.

    Parameters
    ----------
    stats_df:
        Summary statistics as returned by :func:`compute_stats`.
    out_dir:
        Output directory; the figure is written to out_dir/plots/.
    """
    _apply_style()

    if stats_df.empty:
        log.warning("plot_dot: stats_df is empty -- skipped.")
        return

    domains = _get_domains(stats_df)
    models  = sorted(stats_df["model"].unique())
    n_m     = len(models)
    n_dom   = len(domains)
    n_ch    = len(CHALLENGERS)
    has_dom = domains != ["all"]

    # ------------------------------------------------------------------
    # Y-axis layout: each model group = n_ch rows at ROW_H spacing
    # plus a gap between groups.
    # ------------------------------------------------------------------
    ROW_H    = 1.0    # data units between challengers within a group
    GRP_GAP  = 2.2    # data units between model groups
    GRP_SPAN = (n_ch - 1) * ROW_H

    ch_row: Dict[str, float] = {
        ch: i * ROW_H for i, ch in enumerate(CHALLENGERS)
    }

    # model_y: Y coordinate of the group centre (middle challenger).
    # Reversed so the first model appears at the top.
    model_y: Dict[str, float] = {}
    y_cur = 0.0
    for model in reversed(models):
        model_y[model] = y_cur + GRP_SPAN / 2.0
        y_cur += GRP_SPAN + GRP_GAP

    y_min = -GRP_GAP * 0.6
    y_max = y_cur - GRP_GAP + GRP_SPAN + GRP_GAP * 0.6

    # ------------------------------------------------------------------
    # X-axis range: covers all CI bounds plus annotation margin.
    # ------------------------------------------------------------------
    all_x = pd.concat([
        stats_df["mean_signed_angle"].dropna(),
        stats_df["ci95_lo"].dropna(),
        stats_df["ci95_hi"].dropna(),
    ])
    x_data_abs   = max(abs(all_x.min()), abs(all_x.max()))
    STAR_MARGIN  = max(8.0, x_data_abs * 0.14)
    X_PAD        = max(6.0, x_data_abs * 0.10)
    x_lo = -(x_data_abs + X_PAD)
    x_hi =   x_data_abs + X_PAD + STAR_MARGIN

    # ------------------------------------------------------------------
    # Figure geometry.
    # ------------------------------------------------------------------
    LABEL_W = 1.30    # inches -- model name column
    PANEL_W = 3.20    # inches -- each domain panel
    LEG_H   = 1.20    # inches -- legend strip below panels
    TOP_H   = 0.60    # inches -- title space above panels
    DATA_H  = max(4.0, n_m * (GRP_SPAN + GRP_GAP) * 0.18 + 1.5)

    fig_w = LABEL_W + n_dom * PANEL_W + 0.30
    fig_h = TOP_H + DATA_H + LEG_H + 0.20

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    left_frac   = LABEL_W / fig_w
    right_frac  = 1.0 - 0.18 / fig_w
    bottom_frac = (LEG_H + 0.10) / fig_h
    top_frac    = 1.0 - (TOP_H / fig_h)

    gs = gridspec.GridSpec(
        1, n_dom,
        left=left_frac, right=right_frac,
        bottom=bottom_frac, top=top_frac,
        wspace=0.0,
    )
    axes = [fig.add_subplot(gs[0, di]) for di in range(n_dom)]

    # Share Y axis across all panels.
    for ax in axes[1:]:
        ax.sharey(axes[0])

    # ------------------------------------------------------------------
    # Draw each domain panel.
    # ------------------------------------------------------------------
    for di, domain in enumerate(domains):
        ax      = axes[di]
        dom_lbl = DOMAIN_LABELS.get(domain, domain.upper())
        sub_dom = stats_df[stats_df["domain"] == domain] if has_dom else stats_df

        # Zero-degree reference line.
        ax.axvline(0, color="#3A3A3A", lw=1.3, ls="--", zorder=2, alpha=0.70)

        # Alternating pale model band shading for row tracking.
        for mi, model in enumerate(reversed(models)):
            yc  = model_y[model]
            bnd = GRP_SPAN / 2.0 + GRP_GAP * 0.35
            if mi % 2 == 0:
                ax.axhspan(
                    yc - bnd, yc + bnd,
                    colour="#F4F4F4", alpha=0.55,
                    linewidth=0, zorder=0,
                )

        # Data: CI lines, end caps, symbols, and significance annotations.
        for model in models:
            yc = model_y[model]
            for ci_idx, ch in enumerate(CHALLENGERS):
                row = sub_dom[
                    (sub_dom["model"] == model) &
                    (sub_dom["challenger"] == ch)
                ]
                if row.empty:
                    continue
                r      = row.iloc[0]
                mean_v = float(r["mean_signed_angle"])
                ci_lo  = float(r["ci95_lo"])
                ci_hi  = float(r["ci95_hi"])
                sig    = str(r["wilcoxon_sig"])
                colour = CHALLENGER_META[ch]["colour"]
                marker = CHALLENGER_META[ch]["marker"]
                y_pos  = yc - GRP_SPAN / 2.0 + ci_idx * ROW_H
                is_sig = sig not in ("ns", "")

                # 95% CI line.
                ax.plot(
                    [ci_lo, ci_hi], [y_pos, y_pos],
                    color=colour, lw=2.0, alpha=0.80,
                    zorder=3, solid_capstyle="butt",
                )

                # Explicit CI end caps (short vertical ticks).
                cap_h = ROW_H * 0.22
                for cap_x in [ci_lo, ci_hi]:
                    ax.plot(
                        [cap_x, cap_x],
                        [y_pos - cap_h, y_pos + cap_h],
                        color=colour, lw=1.6, alpha=0.80, zorder=3,
                    )

                # Symbol: filled = Wilcoxon significant, open = not significant.
                ax.scatter(
                    mean_v, y_pos,
                    c=colour if is_sig else "white",
                    edgecolors=colour,
                    linewidths=2.0,
                    s=55 if is_sig else 44,
                    marker=marker,
                    zorder=6,
                )

                # Significance annotation to the right of the CI upper bound.
                if sig and sig not in ("ns", ""):
                    star_x = ci_hi + STAR_MARGIN * 0.18
                    ax.text(
                        star_x, y_pos, sig,
                        fontsize=8.5, colour=colour,
                        va="center", ha="left",
                        fontweight="bold", zorder=7,
                        clip_on=False,
                    )

        # Axes formatting.
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Signed angle vs NLQ (degrees)", fontsize=11, labelpad=5)
        ax.set_title(dom_lbl, fontsize=13, fontweight="bold",
                     pad=8, colour="#111111")

        x_tick_step = max(10, int(np.ceil(x_data_abs / 4 / 10) * 10))
        x_ticks = np.arange(
            int(np.ceil(x_lo / x_tick_step)) * x_tick_step,
            int(np.floor((x_data_abs + X_PAD) / x_tick_step)) * x_tick_step + 1,
            x_tick_step,
        )
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="x", labelsize=10)

        ax.tick_params(axis="y", length=0, labelsize=0)
        ax.set_yticks([model_y[m] for m in models])
        ax.set_yticklabels([""] * n_m)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if di > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

        # Light horizontal guide lines at each model centre.
        for model in models:
            ax.axhline(
                model_y[model],
                colour="#DDDDDD", lw=0.5, zorder=1,
            )

    # ------------------------------------------------------------------
    # Model name labels drawn in figure coordinates.
    # ------------------------------------------------------------------
    for model in models:
        yc     = model_y[model]
        y_disp = axes[0].transData.transform((0, yc))[1]
        y_fig  = fig.transFigure.inverted().transform((0, y_disp))[1]
        fig.text(
            left_frac - 0.012, y_fig,
            model.replace("_coder", "\ncoder").replace("_", " "),
            ha="right", va="center",
            fontsize=11, fontweight="bold",
            colour="#1A1A1A",
        )

    # ------------------------------------------------------------------
    # Legend strip below all panels.
    # ------------------------------------------------------------------
    leg_bottom = 0.02 / fig_h
    leg_top    = (LEG_H - 0.10) / fig_h

    # Row 0: method symbols.
    n_r0   = len(CHALLENGERS)
    x_step = (right_frac - left_frac) / (n_r0 + 0.5)
    for i, ch in enumerate(CHALLENGERS):
        m   = CHALLENGER_META[ch]
        xf  = left_frac + (i + 0.5) * x_step
        yf  = leg_bottom + (leg_top - leg_bottom) * 0.72
        fig.text(
            xf + 0.012, yf, m["display"],
            ha="left", va="center",
            fontsize=10, colour=m["colour"], fontweight="bold",
            transform=fig.transFigure,
        )

    # Row 1: significance key.
    sig_labels = [
        "Filled symbol = Wilcoxon p < 0.05",
        "Open symbol = not significant",
        "Error bar = 95% bootstrap CI",
        "-- = 0 degree reference",
    ]
    n_r1    = len(sig_labels)
    x_step1 = (right_frac - left_frac) / (n_r1 + 0.2)
    for i, lbl in enumerate(sig_labels):
        xf = left_frac + (i + 0.3) * x_step1
        yf = leg_bottom + (leg_top - leg_bottom) * 0.22
        fig.text(
            xf, yf, lbl,
            ha="left", va="center",
            fontsize=9, colour="#444444",
            transform=fig.transFigure,
            style="italic",
        )

    # Thin separator line above legend strip.
    sep_y = bottom_frac - 0.005
    fig.add_artist(plt.Line2D(
        [left_frac, right_frac], [sep_y, sep_y],
        transform=fig.transFigure,
        colour="#CCCCCC", lw=0.8, zorder=10,
    ))

    # ------------------------------------------------------------------
    # Figure title.
    # ------------------------------------------------------------------
    title_y = top_frac + (1.0 - top_frac) * 0.60
    fig.text(
        0.5, title_y,
        f"Signed Angular Displacement vs NLQ -- {FEATURE_SET_NAME}",
        ha="center", va="center",
        fontsize=14, fontweight="bold", colour="#111111",
    )
    fig.text(
        0.5, top_frac + (1.0 - top_frac) * 0.18,
        FEATURE_SET_SUBTITLE,
        ha="center", va="center",
        fontsize=9.5, colour="#666666", style="italic",
    )

    # Save.
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, plots_dir / f"signed_angle_{FEATURE_SET_ID}_dot.png")


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def write_csv_report(stats_df: pd.DataFrame, out_path: Path) -> None:
    """
    Write *stats_df* to a CSV file at *out_path*.

    Columns are written in a canonical order; any columns not in the
    canonical list are appended after it.
    """
    cols_ordered = [
        "feature_set", "feature_set_name", "model", "domain",
        "challenger", "challenger_display",
        "n", "mean_signed_angle", "median_signed_angle",
        "std_signed_angle", "ci95_lo", "ci95_hi",
        "wilcoxon_stat", "wilcoxon_p", "wilcoxon_sig",
        "cohens_d", "pct_positive", "pct_negative",
    ]
    cols = [c for c in cols_ordered if c in stats_df.columns]
    stats_df[cols].to_csv(str(out_path), index=False)
    log.info("Saved CSV: %s", out_path.name)


# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------

def write_latex_report(stats_df: pd.DataFrame, out_path: Path) -> None:
    """
    Write a LaTeX booktabs table and narrative comments to *out_path*.

    Table structure
    ---------------
    Rows: one per model.
    Column groups: one per domain (domain name as multi-column header).
    Sub-columns: SPL (SPARQL_NLQ), NLC (NLQ_CoT), SPC (SPARQL_CoT).
    Cell content: mean +/- SD with Wilcoxon significance superscript.

    The file also contains commented narrative statistics (direction
    flips and maximum displacement cells) which are intended as a
    drafting aid rather than for direct inclusion.

    Parameters
    ----------
    stats_df:
        Summary statistics as returned by :func:`compute_stats`.
    out_path:
        Full path to the output .tex file.
    """
    lines: List[str] = [
        "% Signed Angular Displacement Report",
        "% One table for the composite feature vector.",
        "% Cell entries: mean +/- SD with Wilcoxon significance superscript.",
        "% Significance vs zero displacement (Wilcoxon signed-rank, two-sided).",
        "",
    ]

    domains = _get_domains(stats_df)
    models  = sorted(stats_df["model"].unique())
    n_ch    = len(CHALLENGERS)
    has_dom = domains != ["all"]

    cap = (
        "Signed angular displacement (degrees) vs NLQ baseline -- "
        f"\\textbf{{{FEATURE_SET_NAME}}}. "
        "Entries: mean $\\pm$ SD. "
        "Wilcoxon signed-rank vs 0\\textdegree{}: "
        "\\textsuperscript{***}$p{<}.001$, "
        "\\textsuperscript{**}$p{<}.01$, "
        "\\textsuperscript{*}$p{<}.05$. "
        "SPL\\,=\\,SPARQL\\_NLQ; NLC\\,=\\,NLQ\\_CoT; "
        "SPC\\,=\\,SPARQL\\_CoT."
    )

    col_spec = "l" + "rrr" * len(domains)
    dom_hdrs = " ".join(
        f"& \\multicolumn{{{n_ch}}}{{c}}"
        f"{{\\textbf{{{DOMAIN_LABELS.get(d, d.upper())}}}}}"
        for d in domains
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + di * n_ch}--{1 + (di + 1) * n_ch}}}"
        for di in range(len(domains))
    )
    ch_hdrs = " ".join(
        "& \\textbf{SPL} & \\textbf{NLC} & \\textbf{SPC}"
        for _ in domains
    )

    lines += [
        r"\begin{table}[ht]",
        r"\centering\small",
        f"\\caption{{{cap}}}",
        f"\\label{{tab:signed_angle_{FEATURE_SET_ID}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"\\textbf{{Model}} {dom_hdrs} \\\\",
        cmidrules,
        f"\\textbf{{}} {ch_hdrs} \\\\",
        r"\midrule",
    ]

    for model in models:
        cells = [model.replace("_", r"\_")]
        for domain in domains:
            sub_dom = (
                stats_df[stats_df["domain"] == domain]
                if has_dom
                else stats_df
            )
            for ch in CHALLENGERS:
                rr = sub_dom[
                    (sub_dom["model"] == model) &
                    (sub_dom["challenger"] == ch)
                ]
                if rr.empty:
                    cells.append("---")
                    continue
                r    = rr.iloc[0]
                mean = float(r["mean_signed_angle"])
                std  = float(r["std_signed_angle"])
                sig  = str(r["wilcoxon_sig"])
                sup  = (
                    f"\\textsuperscript{{{sig}}}"
                    if sig and sig != "ns"
                    else r"\textsuperscript{\textit{ns}}"
                )
                cells.append(f"${mean:+.1f}\\!\\pm\\!{std:.1f}${sup}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # Narrative statistics as comments (drafting aid only).
    lines += [
        "% -- Narrative statistics (not for direct inclusion) --",
        "",
        f"% Feature set: {FEATURE_SET_NAME}",
    ]

    sparql_rows = stats_df[stats_df["challenger"] == "sparql"].copy()
    if not sparql_rows.empty:
        top = sparql_rows.loc[
            sparql_rows["mean_signed_angle"].abs().idxmax()
        ]
        dom_str = f"  domain={top['domain']}" if has_dom else ""
        lines.append(
            f"%   Largest SPARQL displacement: {top['model']}"
            f"  mean={top['mean_signed_angle']:+.2f} degrees"
            f"  ({top['wilcoxon_sig']}){dom_str}"
        )

    for model in sorted(stats_df["model"].unique()):
        sm = stats_df[stats_df["model"] == model]
        for domain in domains:
            sm_dom = sm[sm["domain"] == domain] if has_dom else sm
            signs: Dict[str, float] = {}
            for ch in CHALLENGERS:
                rr = sm_dom[sm_dom["challenger"] == ch]
                if not rr.empty:
                    signs[ch] = np.sign(rr.iloc[0]["mean_signed_angle"])
            if len(signs) == 3 and len(set(signs.values())) > 1:
                flip_parts = "; ".join(
                    f"{CHALLENGER_META[k]['display']}={v:+.0f}"
                    for k, v in signs.items()
                )
                dom_str = f" [{domain}]" if has_dom else ""
                lines.append(
                    f"%   {model}{dom_str}: direction flip ({flip_parts})"
                )
    lines.append("%")

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Saved LaTeX: %s", out_path.name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(json_files: List[str], output_dir: str) -> None:
    """
    Execute the full signed angular distance analysis pipeline.

    Steps:

    1. Load and validate all JSON result files via
       :func:`build_feature_frame`.
    2. Check that all required composite feature columns are present.
    3. Compute per-query signed angles via :func:`compute_signed_angles`.
    4. Aggregate to summary statistics via :func:`compute_stats`.
    5. Write per-query angles to CSV.
    6. Generate the diverging dot plot via :func:`plot_dot`.
    7. Write summary statistics to CSV and LaTeX.

    Parameters
    ----------
    json_files:
        List of paths to JSON result files.
    output_dir:
        Root directory for all output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("Signed Angular Distance Analysis -- Composite Feature Vector")
    log.info("Input JSON files: %d", len(json_files))
    log.info("=" * 65)

    feat_df   = build_feature_frame(json_files)
    available = [c for c in FEATURE_COLS if c in feat_df.columns]

    if not available:
        log.error(
            "None of the required feature columns are present in the data. "
            "Expected: %s", FEATURE_COLS,
        )
        return

    missing = [c for c in FEATURE_COLS if c not in feat_df.columns]
    if missing:
        log.warning(
            "The following feature columns are absent and will be excluded: %s",
            missing,
        )

    log.info("Computing signed angles (features: %s) ...", available)
    angle_df = compute_signed_angles(feat_df, available)

    if angle_df.empty:
        log.error("No signed angles could be computed -- check input data.")
        return

    angle_df.to_csv(
        out / f"signed_angles_{FEATURE_SET_ID}.csv", index=False
    )
    log.info("Per-query angles written to signed_angles_%s.csv", FEATURE_SET_ID)

    log.info("Aggregating summary statistics ...")
    stats_df = compute_stats(angle_df)

    # Log primary findings.
    log.info("")
    log.info("-- SUMMARY: SPARQL (primary challenger) --")
    sparql_rows = stats_df[stats_df["challenger"] == "sparql"]
    for _, r in sparql_rows.sort_values(["domain", "model"]).iterrows():
        log.info(
            "Model=%-12s  Domain=%-8s  mean=%+.2f deg  "
            "CI=[%+.2f, %+.2f]  %s  Cohen's d=%s",
            r["model"], r["domain"],
            r["mean_signed_angle"],
            r["ci95_lo"], r["ci95_hi"],
            r["wilcoxon_sig"],
            f"{r['cohens_d']:.3f}" if np.isfinite(r["cohens_d"]) else "nan",
        )
    log.info("")

    log.info("Generating diverging dot plot ...")
    plot_dot(stats_df, out)

    write_csv_report(stats_df, out / "signed_angle_report.csv")
    write_latex_report(stats_df, out / "signed_angle_report.tex")

    log.info("=" * 65)
    log.info("Analysis complete.  Output directory: %s", out)
    log.info("  signed_angles_%s.csv", FEATURE_SET_ID)
    log.info("  signed_angle_report.csv")
    log.info("  signed_angle_report.tex")
    log.info("  plots/signed_angle_%s_dot.png/.pdf", FEATURE_SET_ID)
    log.info("=" * 65)


def main() -> None:
    """Parse command-line arguments and invoke :func:`run`."""
    parser = argparse.ArgumentParser(
        description=(
            "Signed angular distance analysis for the composite feature "
            "vector.  Produces a diverging dot plot and summary statistics."
        )
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing JSON result files (searched recursively).",
    )
    grp.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="Explicit list of JSON result file paths.",
    )
    parser.add_argument(
        "--output_dir",
        default="./signed_angle_output/",
        help="Output directory (default: ./signed_angle_output/).",
    )
    args = parser.parse_args()

    json_files = (
        discover_files(args.input_dir) if args.input_dir else args.files
    )
    if not json_files:
        parser.error("No JSON files found.")

    run(json_files, args.output_dir)


if __name__ == "__main__":
    main()
