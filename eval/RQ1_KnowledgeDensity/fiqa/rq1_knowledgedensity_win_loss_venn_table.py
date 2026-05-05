#!/usr/bin/env python3
"""
rq1_venn_table.py
=================
Generates two visualisation artefacts for RQ1 KD_TKF analysis:

  1. Per-model Venn diagrams showing the overlap of winning query IDs
     across the three challenger methods (SPARQL, NLQ_CoT, SPARQL_CoT)
     relative to the NLQ baseline.

  2. A win-rate summary table (PNG render and LaTeX source) reporting,
     per model, the percentage of query IDs where each challenger
     outperforms the NLQ baseline on the KD_TKF metric.

Input
-----
A Microsoft Excel workbook containing one sheet per model, each named
``QueryQuality_<model>``.  Required columns (case-insensitive):

  query_id      -- unique identifier for each query
  exp_method    -- experimental method label (or ``method`` as fallback)
  kd_tkf        -- numeric KD_TKF score
  ext_method    -- extraction method label (used to filter rows)
  domain        -- knowledge domain label

Output
------
Written to ``--output_dir`` (default: ``./rq1_output/``):

  fig_venn_wins_per_model.png   -- tiled per-model Venn diagrams
  fig_winrate_table.png         -- win-rate summary table (PNG)
  tab_winrates.tex              -- win-rate summary table (LaTeX / booktabs)
  rq1_win_rate_table.csv        -- intermediate win-rate data

Usage
-----
  python rq1_venn_table.py \\
      --quality_xlsx /path/to/query_quality.xlsx \\
      --output_dir   ./rq1_output/ \\
      [--extractor   rule_based]

Dependencies
------------
  numpy, pandas, matplotlib, scipy, openpyxl, matplotlib-venn
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
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

#: The reference (baseline) method against which challengers are compared.
REFERENCE = "nlq"

#: A challenger is considered to win if its delta exceeds this threshold.
IMPROVEMENT_EPS = 0

#: Output resolution in dots per inch.
DPI = 300

#: Canonical lowercase method names, normalised from various input spellings.
METHOD_ALIASES: dict = {
    "nlq":         "nlq",
    "sparql":      "sparql",
    "nlq_cot":     "nlq_cot",
    "sparql_cot":  "sparql_cot",
    "NLQ":         "nlq",
    "SPARQL":      "sparql",
    "NLQ_CoT":     "nlq_cot",
    "SPARQL_CoT":  "sparql_cot",
}

#: Ordered list of challenger method keys.
CHALLENGERS = ["sparql", "nlq_cot", "sparql_cot"]

#: Short column abbreviations used in table headers.
COL_LABELS = ["SPL", "NLC", "SPC"]

#: Human-readable display names for each method key.
DISPLAY: dict = {
    "sparql":     "SPARQL",
    "nlq_cot":    "NLQ_CoT",
    "sparql_cot": "SPARQL_CoT",
    "nlq":        "NLQ",
}

#: Mapping from challenger key to its Wilcoxon p-value column name.
WILCOXON_COLS: dict = {
    "sparql":     "wilcoxon_p_sp",
    "nlq_cot":    "wilcoxon_p_nl",
    "sparql_cot": "wilcoxon_p_sc",
}

# Colour palette (Okabe-Ito, accessible).
_C_SP  = "#D55E00"   # SPARQL
_C_NL  = "#009E73"   # NLQ_CoT
_C_SC  = "#CC79A7"   # SPARQL_CoT
_C_BLK = "#222222"

METHOD_COLOURS: dict = {
    "sparql":     _C_SP,
    "nlq_cot":    _C_NL,
    "sparql_cot": _C_SC,
}


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------

def _configure_matplotlib() -> None:
    """Apply global rcParams for consistent plot styling."""
    plt.rcParams.update({
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
        "axes.edgecolor":        "#222222",
        "axes.linewidth":        2.0,
        "font.family":           "DejaVu Sans",
        "font.size":             17,
        "axes.titlesize":        24,
        "axes.labelsize":        20,
        "xtick.labelsize":       17,
        "ytick.labelsize":       17,
        "legend.fontsize":       16,
        "legend.title_fontsize": 17,
        "savefig.dpi":           DPI,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     "white",
        "grid.color":            "#dddddd",
        "grid.linewidth":        0.9,
        "grid.linestyle":        "--",
        "axes.axisbelow":        True,
    })


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save *fig* to *path* at DPI resolution and close it."""
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Saved -> %s", path.name)


# ---------------------------------------------------------------------------
# Data loading and normalisation
# ---------------------------------------------------------------------------

def _normalise_method(raw: str) -> str:
    """Return the canonical lowercase method key for *raw*."""
    name = str(raw).lower().replace("-", "_").replace(" ", "_").strip()
    return METHOD_ALIASES.get(name, METHOD_ALIASES.get(raw, name))


def _normalise_query_id(value) -> str:
    """Return a zero-padded six-digit string for *value* where possible."""
    try:
        return f"{int(float(str(value).strip())):06d}"
    except (ValueError, TypeError):
        return str(value).strip()


def _log10_safe(series: pd.Series) -> pd.Series:
    """Return log10 of *series*, substituting NaN for non-positive values."""
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.where(numeric > 0, np.log10(numeric), np.nan),
        index=series.index,
    )


def load_quality_data(xlsx_path: str, extractor: str = "rule_based") -> pd.DataFrame:
    """
    Load and concatenate all ``QueryQuality_<model>`` sheets from *xlsx_path*.

    Parameters
    ----------
    xlsx_path:
        Path to the Excel workbook.
    extractor:
        Value of ``ext_method`` to retain.  Rows with other values are
        discarded.

    Returns
    -------
    pandas.DataFrame
        Combined data frame with normalised ``method``, ``model``, and
        ``query_id`` columns.

    Raises
    ------
    RuntimeError
        If no ``QueryQuality_`` sheets are found in the workbook.
    """
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    frames = []

    for sheet in [s for s in xl.sheet_names if s.startswith("QueryQuality_")]:
        try:
            df = pd.read_excel(xl, sheet_name=sheet)
            df.columns = [c.lower().strip() for c in df.columns]

            if "ext_method" in df.columns:
                df = df[df["ext_method"].str.lower().str.strip()
                        == extractor.lower()]

            method_col = "exp_method" if "exp_method" in df.columns else "method"
            df["method"] = df[method_col].apply(_normalise_method)
            df["model"]  = _normalise_method(sheet.replace("QueryQuality_", ""))

            for col in ["kd_tkf", "triple_count_c"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "query_id" not in df.columns:
                log.warning("Sheet '%s': no query_id column -- skipping.", sheet)
                continue

            df["query_id"] = df["query_id"].apply(_normalise_query_id)
            frames.append(df)
            log.info("Loaded '%s': %d rows.", sheet, len(df))

        except Exception as exc:
            log.warning("Sheet '%s' failed: %s", sheet, exc)

    if not frames:
        raise RuntimeError("No QueryQuality_ sheets found in the workbook.")

    combined = pd.concat(frames, ignore_index=True)
    log.info("Total rows loaded: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Paired alignment
# ---------------------------------------------------------------------------

def _align_paired(
    df_model: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric_col: str,
    use_log: bool = False,
) -> tuple:
    """
    Return paired (a, b) series aligned on ``query_id``.

    For each query ID the per-query median of *metric_col* is used, so that
    duplicate rows (e.g. multiple extraction runs) do not inflate the sample.
    Only query IDs present in both methods with non-NaN values are retained.

    Parameters
    ----------
    df_model:
        Subset of the quality data frame for a single model (and domain).
    method_a, method_b:
        Canonical method keys to compare.
    metric_col:
        Column name of the metric to compare.
    use_log:
        If ``True``, apply a safe log10 transform before returning.
    """
    a = (df_model[df_model["method"] == method_a]
         .groupby("query_id")[metric_col].median())
    b = (df_model[df_model["method"] == method_b]
         .groupby("query_id")[metric_col].median())

    if use_log:
        a = _log10_safe(a)
        b = _log10_safe(b)

    common = a.dropna().index.intersection(b.dropna().index)
    return a.loc[common], b.loc[common]


# ---------------------------------------------------------------------------
# Win-rate table builder
# ---------------------------------------------------------------------------

def build_win_rate_table(quality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(domain, model) win rates and Wilcoxon p-values.

    For each domain-model combination the function aligns each challenger
    with the NLQ baseline on ``kd_tkf`` (log10-transformed, paired by
    ``query_id``) and computes:

    * **Win rate** -- the percentage of aligned query IDs where the challenger
      exceeds the baseline by more than :data:`IMPROVEMENT_EPS`.
    * **Wilcoxon p-value** -- two-sided Wilcoxon signed-rank test (requires
      at least five paired observations; otherwise ``NaN``).

    Parameters
    ----------
    quality_df:
        Combined quality data frame as returned by :func:`load_quality_data`.

    Returns
    -------
    pandas.DataFrame
        One row per (domain, model) with columns:

        ``domain``, ``model``,
        ``sp_wr``, ``nl_wr``, ``sc_wr``         -- win rates (0-100 %),
        ``wilcoxon_p_sp``, ``wilcoxon_p_nl``, ``wilcoxon_p_sc``,
        ``sp_bold``                               -- bool, SPARQL beats both CoT variants.
    """
    domain_col = "domain" if "domain" in quality_df.columns else "model"
    domains    = sorted(quality_df[domain_col].unique())
    models     = sorted(quality_df["model"].unique())
    rows       = []

    for domain in domains:
        df_domain = quality_df[quality_df[domain_col] == domain]

        for model in models:
            df_model = df_domain[df_domain["model"] == model]
            if df_model.empty:
                continue

            record = {"domain": domain, "model": model}
            win_rates: dict = {}

            for challenger in CHALLENGERS:
                a, b  = _align_paired(df_model, REFERENCE, challenger,
                                      "kd_tkf", use_log=True)
                n     = len(a)
                p_col = WILCOXON_COLS[challenger]

                if n == 0:
                    win_rates[challenger] = np.nan
                    record[p_col]         = np.nan
                    continue

                delta = b - a
                win_rates[challenger] = round(
                    float(100.0 * (delta > IMPROVEMENT_EPS).sum() / n), 1
                )

                if n >= 5:
                    try:
                        _, p_val      = scipy_stats.wilcoxon(
                            a.values, b.values, alternative="two-sided"
                        )
                        record[p_col] = float(p_val)
                    except Exception:
                        record[p_col] = np.nan
                else:
                    record[p_col] = np.nan

            record["sp_wr"] = win_rates.get("sparql",     np.nan)
            record["nl_wr"] = win_rates.get("nlq_cot",    np.nan)
            record["sc_wr"] = win_rates.get("sparql_cot", np.nan)
            record["sp_bold"] = bool(
                np.isfinite(record["sp_wr"])
                and np.isfinite(record["nl_wr"])
                and np.isfinite(record["sc_wr"])
                and record["sp_wr"] > record["nl_wr"]
                and record["sp_wr"] > record["sc_wr"]
            )
            rows.append(record)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _significance_label(p_value: float) -> str:
    """Return a significance superscript string for *p_value*."""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "+"
    if p_value < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Figure -- per-model Venn diagrams
# ---------------------------------------------------------------------------

def plot_venn_per_model(
    quality_df: pd.DataFrame,
    out: Path,
    extractor: str = "rule_based",
) -> None:
    """
    Produce a tiled figure of three-circle Venn diagrams, one per model.

    Each Venn diagram shows the sets of query IDs for which each challenger
    method (SPARQL, NLQ_CoT, SPARQL_CoT) outperforms the NLQ baseline on
    ``kd_tkf`` (log10-transformed, paired by ``query_id``).  Intersections
    represent query IDs won by multiple challengers simultaneously.

    The figure is saved to ``fig_venn_wins_per_model.png`` in *out*.

    Parameters
    ----------
    quality_df:
        Combined quality data frame as returned by :func:`load_quality_data`.
    out:
        Output directory path.
    extractor:
        Retained for interface consistency; row filtering has already been
        applied by :func:`load_quality_data`.

    Notes
    -----
    Requires the ``matplotlib-venn`` package.  If it is not installed the
    function logs an error message and returns without writing any file.
    """
    try:
        from matplotlib_venn import venn3, venn3_circles
    except ImportError:
        log.error(
            "matplotlib-venn is not installed.  "
            "Install it with:  pip install matplotlib-venn"
        )
        return

    _configure_matplotlib()

    VENN_ALPHA   = 0.50
    VENN_COLOURS = [_C_SP, _C_NL, _C_SC]

    models   = sorted(quality_df["model"].unique())
    n_models = len(models)
    n_cols   = min(4, n_models)
    n_rows   = int(np.ceil(n_models / n_cols))

    cell_w, cell_h = 4.6, 4.2
    fig_w = cell_w * n_cols + 1.2
    fig_h = cell_h * n_rows + 2.8

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI, facecolor="white")

    legend_h_frac = 2.0 / fig_h
    gs = GridSpec(
        n_rows, n_cols,
        figure=fig,
        left=0.03, right=0.97,
        top=0.91,
        bottom=legend_h_frac + 0.02,
        hspace=0.45, wspace=0.18,
    )

    for mi, model in enumerate(models):
        row_idx, col_idx = divmod(mi, n_cols)
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.set_aspect("equal")
        ax.axis("off")

        df_model = quality_df[quality_df["model"] == model]

        winning_sets: dict = {}
        for challenger in CHALLENGERS:
            a, b  = _align_paired(df_model, REFERENCE, challenger,
                                   "kd_tkf", use_log=True)
            delta = b - a
            winning_sets[challenger] = set(
                delta.index[delta > IMPROVEMENT_EPS].tolist()
            )

        s_sparql     = winning_sets["sparql"]
        s_nlq_cot    = winning_sets["nlq_cot"]
        s_sparql_cot = winning_sets["sparql_cot"]

        total_winning = len(s_sparql | s_nlq_cot | s_sparql_cot)

        subsets = (
            len(s_sparql - s_nlq_cot - s_sparql_cot),
            len(s_nlq_cot - s_sparql - s_sparql_cot),
            len(s_sparql & s_nlq_cot - s_sparql_cot),
            len(s_sparql_cot - s_sparql - s_nlq_cot),
            len(s_sparql & s_sparql_cot - s_nlq_cot),
            len(s_nlq_cot & s_sparql_cot - s_sparql),
            len(s_sparql & s_nlq_cot & s_sparql_cot),
        )

        venn_obj = venn3(
            subsets=subsets,
            set_labels=("", "", ""),
            set_colors=VENN_COLOURS,
            alpha=VENN_ALPHA,
            ax=ax,
        )

        venn3_circles(
            subsets=subsets,
            linestyle="solid",
            linewidth=2.0,
            color="#333333",
            ax=ax,
        )

        region_ids = ("100", "010", "110", "001", "101", "011", "111")
        for patch_id, count in zip(region_ids, subsets):
            label = venn_obj.get_label_by_id(patch_id)
            if label is not None:
                if count == 0:
                    label.set_text("")
                else:
                    label.set_text(str(count))
                    label.set_fontsize(17)
                    label.set_fontweight("bold")
                    label.set_color("#111111")

        label_positions = [
            (s_sparql,     _C_SP, (-0.70,  0.52), "SPARQL"),
            (s_nlq_cot,    _C_NL, ( 0.70,  0.52), "NLQ_CoT"),
            (s_sparql_cot, _C_SC, ( 0.00, -0.72), "SPARQL_CoT"),
        ]
        for s_set, colour, (tx, ty), label_name in label_positions:
            ax.text(
                tx, ty,
                f"{label_name}\nn={len(s_set)}",
                ha="center", va="center",
                fontsize=15, fontweight="bold",
                color=colour,
                transform=ax.transData,
            )

        ax.set_title(
            f"{model}\n(total winning QIDs = {total_winning})",
            fontsize=18, fontweight="bold",
            color="#1F3864", pad=6,
        )

    # Hide unused subplot cells.
    for mi in range(n_models, n_rows * n_cols):
        row_idx, col_idx = divmod(mi, n_cols)
        fig.add_subplot(gs[row_idx, col_idx]).axis("off")

    # Legend strip at the bottom of the figure.
    leg_ax = fig.add_axes([0.03, 0.00, 0.94, legend_h_frac])
    leg_ax.axis("off")
    handles = [
        mpatches.Patch(color=_C_SP, alpha=VENN_ALPHA, label="SPARQL  (SPL)"),
        mpatches.Patch(color=_C_NL, alpha=VENN_ALPHA, label="NLQ_CoT  (NLC)"),
        mpatches.Patch(color=_C_SC, alpha=VENN_ALPHA, label="SPARQL_CoT  (SPC)"),
        mpatches.Patch(color="none", label="Numbers = query IDs in each region"),
        mpatches.Patch(color="none", label="n = total winning QIDs per method"),
        mpatches.Patch(color="none",
                       label=f"Win criterion: delta_log10(KD_TKF) > {IMPROVEMENT_EPS}"),
        mpatches.Patch(color="none", label="Baseline = NLQ (not shown)"),
    ]
    leg_ax.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=True,
        framealpha=0.97,
        edgecolor="#aaaaaa",
        fontsize=16,
        handlelength=1.8,
        handleheight=1.6,
        columnspacing=1.6,
        borderpad=1.0,
    )

    fig.suptitle(
        "Overlap of Winning Query IDs per Model\n"
        "Each circle = queries where challenger beats NLQ baseline on KD_TKF  "
        "|  Intersections = queries won by multiple challengers",
        fontsize=22, fontweight="bold", y=0.995,
    )

    _save_figure(fig, out / "fig_venn_wins_per_model.png")


# ---------------------------------------------------------------------------
# Figure -- win-rate summary table
# ---------------------------------------------------------------------------

def plot_winrate_table(win_df: pd.DataFrame, out: Path) -> None:
    """
    Render a win-rate summary table and write its LaTeX source.

    The table reports, for each model, the win rate (%) of each challenger
    over the NLQ baseline together with Wilcoxon significance markers.
    The best-performing challenger per model is highlighted.

    Two files are written to *out*:

    * ``fig_winrate_table.png``  -- matplotlib render.
    * ``tab_winrates.tex``       -- standalone LaTeX (booktabs).

    Columns
    -------
    Model | SPL | NLC | SPC | Best | Delta-Best

    where Delta-Best is the best win rate minus the runner-up (percentage
    points).

    Notes
    -----
    This function is designed for a single-domain run.  When ``win_df``
    contains exactly one domain the domain label is embedded in the table
    caption; when multiple domains are present their names are listed.

    Parameters
    ----------
    win_df:
        Win-rate data frame as returned by :func:`build_win_rate_table`.
    out:
        Output directory path.
    """
    _configure_matplotlib()

    models       = sorted(win_df["model"].unique())
    domain_vals  = win_df["domain"].unique()
    domain_label = (domain_vals[0]
                    if len(domain_vals) == 1
                    else ", ".join(sorted(domain_vals)))

    # Map challenger key -> (win-rate column, p-value column, abbreviation).
    COL_MAP = {
        "sparql":     ("sp_wr", "wilcoxon_p_sp", "SPL"),
        "nlq_cot":    ("nl_wr", "wilcoxon_p_nl", "NLC"),
        "sparql_cot": ("sc_wr", "wilcoxon_p_sc", "SPC"),
    }

    rows_data = []

    for model in models:
        df_model = win_df[win_df["model"] == model]
        record   = {"model": model}

        for challenger, (wr_col, p_col, abbr) in COL_MAP.items():
            vals   = df_model[wr_col].dropna().values
            val    = float(vals[0]) if len(vals) > 0 else np.nan
            p_vals = (df_model[p_col].dropna().values
                      if p_col in df_model.columns else [])
            best_p = float(np.min(p_vals)) if len(p_vals) > 0 else np.nan

            record[f"{abbr}_val"] = val
            record[f"{abbr}_sig"] = _significance_label(best_p)
            record[f"{abbr}_p"]   = best_p

        vals_map = {
            abbr: record[f"{abbr}_val"]
            for abbr in ["SPL", "NLC", "SPC"]
            if np.isfinite(record[f"{abbr}_val"])
        }
        if vals_map:
            best_abbr = max(vals_map, key=vals_map.get)
            best_val  = vals_map[best_abbr]
            others    = [v for k, v in vals_map.items() if k != best_abbr]
            runner_up = max(others) if others else np.nan
            delta     = (best_val - runner_up
                         if np.isfinite(runner_up) else np.nan)
        else:
            best_abbr, best_val, delta = "--", np.nan, np.nan

        record["best"]  = best_abbr
        record["delta"] = delta
        rows_data.append(record)

    # ------------------------------------------------------------------
    # LaTeX source
    # ------------------------------------------------------------------

    def _cell_tex(val: float, sig: str, bold: bool = False) -> str:
        if not np.isfinite(val):
            return "--"
        text = f"{val:.1f}{sig}"
        return f"\\textbf{{{text}}}" if bold else text

    def _delta_tex(d: float) -> str:
        return f"{d:+.1f}" if np.isfinite(d) else "--"

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        (
            rf"\caption{{KD\_TKF win rates (\%) over the NLQ baseline "
            rf"-- domain: \textit{{{domain_label}}}. "
            r"Each value is the fraction of query IDs where the challenger "
            r"outperforms NLQ. "
            r"Bold\,=\,highest win rate per model. "
            r"$\Delta$-Best\,=\,best minus runner-up (pp). "
            r"Wilcoxon vs NLQ: $^{*}p{<}0.05$,\ $^{+}p{<}0.001$.}}"
        ),
        r"\label{tab:rq1_winrates}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrc r}",
        r"\toprule",
        (
            r"\textbf{Model} & "
            r"\textbf{SPL} & "
            r"\textbf{NLC} & "
            r"\textbf{SPC} & "
            r"\textbf{Best} & "
            r"$\boldsymbol{\Delta}$\textbf{-Best} \\"
        ),
        r"\midrule",
    ]

    for rec in rows_data:
        best  = rec["best"]
        cells = [
            _cell_tex(rec[f"{a}_val"], rec[f"{a}_sig"], bold=(a == best))
            for a in ["SPL", "NLC", "SPC"]
        ]
        tex_lines.append(
            f"{rec['model']} & "
            f"{cells[0]} & {cells[1]} & {cells[2]} & "
            f"{best} & {_delta_tex(rec['delta'])} \\\\"
        )

    tex_lines += [
        r"\bottomrule",
        (
            r"\multicolumn{6}{l}{\footnotesize "
            r"SPL\,=\,SPARQL;\ NLC\,=\,NLQ\_CoT;\ SPC\,=\,SPARQL\_CoT.\ "
            r"Baseline\,=\,NLQ (not shown).\ "
            r"Win\,=\,challenger\,$\Delta\log_{10}(\text{KD\_TKF}) > 0$\,vs\,NLQ.} \\"
        ),
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = out / "tab_winrates.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    log.info("Saved -> tab_winrates.tex")

    # ------------------------------------------------------------------
    # Matplotlib render
    # ------------------------------------------------------------------

    col_headers = ["Model", "SPL", "NLC", "SPC", "Best", "D-Best"]
    n_cols_tbl  = len(col_headers)
    n_rows_data = len(rows_data)

    def _cell_plain(val: float, sig: str) -> str:
        return f"{val:.1f}{sig}" if np.isfinite(val) else "--"

    def _delta_plain(d: float) -> str:
        return f"{d:+.1f}" if np.isfinite(d) else "--"

    cell_matrix = []
    bold_mask   = []

    for rec in rows_data:
        best  = rec["best"]
        row_t = [rec["model"]]
        row_b = [True]
        for abbr in ["SPL", "NLC", "SPC"]:
            row_t.append(_cell_plain(rec[f"{abbr}_val"], rec[f"{abbr}_sig"]))
            row_b.append(abbr == best)
        row_t.append(best)
        row_b.append(False)
        row_t.append(_delta_plain(rec["delta"]))
        row_b.append(False)
        cell_matrix.append(row_t)
        bold_mask.append(row_b)

    # Fractional column widths (must sum to <= 1.0).
    COL_WIDTHS = [0.16, 0.165, 0.165, 0.165, 0.08, 0.09]
    assert len(COL_WIDTHS) == n_cols_tbl

    ROW_H  = 0.72   # inches per data row
    HEAD_H = 0.80   # inches for header row
    FOOT_H = 0.60   # inches for footnote row
    LEG_H  = 0.70   # inches for legend strip
    FIG_W  = 14.0
    FIG_H  = HEAD_H + n_rows_data * ROW_H + FOOT_H + LEG_H + 1.4

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")

    HDR_FC    = "#1F3864"
    HDR_TC    = "white"
    ALT_FC    = ["#f0f4fa", "#dce4f0"]
    BEST_FC   = "#fff3cd"
    BEST_EC   = "#e6a817"
    BODY_EC   = "#bbbbbb"
    METHOD_FC = {"SPL": "#fce8df", "NLC": "#d6f2ea", "SPC": "#f6e8f2"}
    METHOD_TC = {"SPL": _C_SP,     "NLC": _C_NL,     "SPC": _C_SC}

    content_h = HEAD_H + n_rows_data * ROW_H + FOOT_H
    ax_bottom = (LEG_H + 0.40) / FIG_H
    ax_height = content_h / FIG_H
    ax        = fig.add_axes([0.01, ax_bottom, 0.98, ax_height])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    row_h_frac  = ROW_H  / content_h
    head_h_frac = HEAD_H / content_h
    foot_h_frac = FOOT_H / content_h

    def _x0(ci: int) -> float:
        return sum(COL_WIDTHS[:ci])

    def _rect(axis, x0, y0, w, h, fc, ec=BODY_EC, lw=0.7, zorder=1):
        axis.add_patch(plt.Rectangle(
            (x0, y0), w, h,
            transform=axis.transAxes,
            facecolor=fc, edgecolor=ec, linewidth=lw,
            clip_on=False, zorder=zorder,
        ))

    def _text(axis, x, y, s, ha="center", va="center",
              fs=17, fw="normal", color="#111111", zorder=2):
        axis.text(
            x, y, s, ha=ha, va=va,
            fontsize=fs, fontweight=fw, color=color,
            transform=axis.transAxes, clip_on=False, zorder=zorder,
        )

    # Header row.
    y_head = 1.0 - head_h_frac
    for ci, (hdr, cw) in enumerate(zip(col_headers, COL_WIDTHS)):
        x0 = _x0(ci)
        if hdr in METHOD_TC:
            hfc, htc, hec = METHOD_FC[hdr], METHOD_TC[hdr], METHOD_TC[hdr]
        else:
            hfc, htc, hec = HDR_FC, HDR_TC, "#ffffff"
        _rect(ax, x0, y_head, cw, head_h_frac, fc=hfc, ec=hec, lw=1.6, zorder=3)
        _text(ax, x0 + cw / 2, y_head + head_h_frac / 2,
              hdr, fs=19, fw="bold", color=htc)

    # Data rows.
    for ri, (rec, row_t, row_b) in enumerate(
            zip(rows_data, cell_matrix, bold_mask)):
        best  = rec["best"]
        y_row = y_head - (ri + 1) * row_h_frac

        for ci, (txt, cw, is_bold) in enumerate(
                zip(row_t, COL_WIDTHS, row_b)):
            x0  = _x0(ci)
            hdr = col_headers[ci]

            if ci == 0:
                fc, tc = "#edf1f8", "#1F3864"
            elif hdr in METHOD_FC:
                fc = BEST_FC if hdr == best else METHOD_FC[hdr]
                tc = METHOD_TC[hdr]
            else:
                fc, tc = ALT_FC[ri % 2], "#111111"

            ec = BEST_EC if (hdr in METHOD_FC and hdr == best) else BODY_EC
            lw = 1.4     if (hdr in METHOD_FC and hdr == best) else 0.7

            _rect(ax, x0, y_row, cw, row_h_frac, fc=fc, ec=ec, lw=lw, zorder=2)
            _text(ax, x0 + cw / 2, y_row + row_h_frac / 2, txt,
                  fs=17, fw="bold" if is_bold else "normal", color=tc)

    # Footnote row.
    y_foot   = y_head - (n_rows_data + 1) * row_h_frac
    footnote = (
        f"Domain: {domain_label}  |  "
        "SPL = SPARQL  |  NLC = NLQ_CoT  |  SPC = SPARQL_CoT  |  "
        "Baseline = NLQ (not shown)  |  "
        "Win = challenger delta_log10(KD_TKF) > 0 vs NLQ  |  "
        "* p < 0.05   + p < 0.001  (Wilcoxon vs NLQ)"
    )
    _rect(ax, 0, y_foot, 1.0, foot_h_frac, fc="#f7f7f7", ec="#aaaaaa", lw=0.8)
    _text(ax, 0.5, y_foot + foot_h_frac / 2, footnote,
          fs=13, fw="normal", color="#444444")

    # Outer border.
    total_box_h = head_h_frac + n_rows_data * row_h_frac + foot_h_frac
    ax.add_patch(plt.Rectangle(
        (0, y_foot), 1.0, total_box_h,
        transform=ax.transAxes,
        facecolor="none", edgecolor="#333333",
        linewidth=2.2, clip_on=False, zorder=5,
    ))

    # Legend strip.
    leg_ax = fig.add_axes([0.01, 0.01, 0.98, (LEG_H - 0.10) / FIG_H])
    leg_ax.axis("off")
    handles = [
        mpatches.Patch(color=METHOD_FC["SPL"], edgecolor=_C_SP,
                       linewidth=1.6, label="SPL = SPARQL"),
        mpatches.Patch(color=METHOD_FC["NLC"], edgecolor=_C_NL,
                       linewidth=1.6, label="NLC = NLQ_CoT"),
        mpatches.Patch(color=METHOD_FC["SPC"], edgecolor=_C_SC,
                       linewidth=1.6, label="SPC = SPARQL_CoT"),
        mpatches.Patch(color=BEST_FC, edgecolor=BEST_EC,
                       linewidth=1.6, label="Amber = best challenger for that model"),
        mpatches.Patch(color="none",
                       label="D-Best = best win rate minus runner-up (pp)"),
    ]
    leg_ax.legend(
        handles=handles,
        loc="center", ncol=5,
        frameon=True, framealpha=0.97,
        edgecolor="#aaaaaa",
        fontsize=16,
        handlelength=1.8, handleheight=1.5,
        columnspacing=1.6, borderpad=0.9,
    )

    fig.suptitle(
        f"KD_TKF Win Rates (%) over NLQ Baseline -- Domain: {domain_label}\n"
        "Win rate = % of query IDs where challenger beats NLQ on KD_TKF  |  "
        "bold = highest win rate per model",
        fontsize=20, fontweight="bold", y=0.995,
    )

    _save_figure(fig, out / "fig_winrate_table.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(quality_xlsx: str, output_dir: str, extractor: str = "rule_based") -> None:
    """
    Execute the full RQ1 visualisation pipeline.

    Parameters
    ----------
    quality_xlsx:
        Path to the query-quality Excel workbook.
    output_dir:
        Directory to which all output files are written.
    extractor:
        Extraction method filter (``ext_method`` column value to retain).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading quality data from: %s", quality_xlsx)
    quality_df = load_quality_data(quality_xlsx, extractor)

    log.info("Building win-rate table ...")
    win_df = build_win_rate_table(quality_df)
    win_df.to_csv(out / "rq1_win_rate_table.csv", index=False)
    log.info("  %d rows written to rq1_win_rate_table.csv", len(win_df))

    log.info("Plotting per-model Venn diagrams ...")
    plot_venn_per_model(quality_df, out, extractor)

    log.info("Plotting win-rate summary table ...")
    plot_winrate_table(win_df, out)

    log.info("All outputs written to: %s", out)


def main() -> None:
    """Parse command-line arguments and invoke :func:`run`."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate RQ1 KD_TKF visualisations: "
            "per-model Venn diagrams and win-rate summary table."
        )
    )
    parser.add_argument(
        "--quality_xlsx",
        required=True,
        help="Path to the query-quality workbook (sheets: QueryQuality_<model>).",
    )
    parser.add_argument(
        "--output_dir",
        default="./rq1_output/",
        help="Directory for output files (default: ./rq1_output/).",
    )
    parser.add_argument(
        "--extractor",
        default="rule_based",
        help="Value of ext_method to retain (default: rule_based).",
    )
    args = parser.parse_args()
    run(args.quality_xlsx, args.output_dir, args.extractor)


if __name__ == "__main__":
    main()
