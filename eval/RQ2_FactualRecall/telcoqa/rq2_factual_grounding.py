#!/usr/bin/env python3
"""
rq2_factual_grounding.py
========================
Factual Coverage Analysis: Claim Recall on Recall-Winning Queries (RQ2).

Research Question
-----------------
To what extent do FLQ (SPARQL) and CoT reformulations improve propositional
grounding over the NLQ baseline?

Evaluation Method
-----------------
Claim Recall (CR):
    For each candidate response h, sentences of five or more words are
    extracted as discrete claims.  Each claim c is classified as entailed or
    not entailed by the ground-truth reference g* via a cross-encoder NLI
    model:

        CR = |{c in C(h) : g* |= c}| / |C(h)|

    CR = 1.0 -- every extractable claim is grounded in g*.
    CR = 0.0 -- no claims are grounded in g*.

Coverage Gap (delta-CR):
    Signed coverage gap of challenger CH against the NLQ baseline:

        delta-CR = CR_CH - CR_NLQ

    delta-CR > 0 -- challenger claims are more grounded than NLQ.
    delta-CR <= 0 -- NLQ claims are equally or more grounded.

    delta-CR is computed restricted to recall-winning queries: those where
    the challenger already exceeds NLQ on BERTScore Recall.  A positive
    delta-CR on recall-winning queries confirms that the recall gain is
    accompanied by genuine factual grounding rather than verbose generation.

Methods
-------
  NLQ          -- baseline (natural language query response).
  SPARQL       -- primary challenger (FLQ: formal language query response).
                  Also referred to as SPARQL_NLQ in column names.
  NLQ_CoT      -- CoT reformulation challenger.
  SPARQL_CoT   -- hybrid CoT challenger.

  Primary analysis : SPARQL vs NLQ.
  Secondary        : NLQ_CoT vs NLQ, SPARQL_CoT vs NLQ.

Co-directionality (CD):
    On recall-winning queries, is delta-CR positive (same direction as the
    recall gain)?
    Up arrow   = co-directional (delta-CR > 0): recall gain accompanied by
                 grounding gain.
    Down arrow = non-co-directional (delta-CR <= 0): recall gain not
                 accompanied by grounding gain.

Outputs
-------
CSVs (output_dir/):
    per_query_metrics.csv             -- full per-query metric table (long).
    recall_win_anchored_delta_cr.csv  -- delta-CR restricted to recall-winning
                                        queries, per (challenger x model x
                                        domain).
    challenger_summary.csv            -- aggregated summary with win-rates,
                                        mean delta-CR, co-directionality.
    sparql_primary_summary.csv        -- SPARQL-only summary (primary focus).

LaTeX (output_dir/):
    tab_recall_win_rates.tex  -- SPARQL-centric recall win-rate comparison:
                                 per (model x domain), SPARQL, NLQ_CoT, and
                                 SPARQL_CoT win-rates side-by-side with
                                 absolute counts and delta-pp advantage
                                 columns.

Plots (output_dir/plots/):
    fig_cr_absolute_boxplot.pdf/.png  -- box/violin plot of absolute CR
                                        distributions on recall-winning
                                        queries, faceted by domain.
                                        NLQ deduplication applied: each
                                        (model x domain) cell contributes
                                        exactly one NLQ value.
                                        Log-scale y-axis activated
                                        automatically when the dynamic range
                                        of finite CR values exceeds
                                        LOG_SCALE_RATIO_THRESHOLD.

Usage
-----
  python rq2_factual_grounding.py \\
      --responses_xlsx path/to/data.xlsx \\
      --output_dir     ./rq2_output/

  Add --no_semantic to skip BERTScore and Claim Recall computation (useful
  for schema validation without model inference).
  Add --domain_col COLUMN_NAME if the domain label is stored in a column
  rather than being inferred from the sheet name.

Dependencies
------------
  numpy, pandas, matplotlib, scipy, openpyxl, bert-score, transformers,
  torch, nltk
"""

import re
import argparse
import logging
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema configuration
# ---------------------------------------------------------------------------

GT_SHEET    = "ground_truth"
GT_TEXT_COL = "ground_truth"
GT_QID_COL  = "Query_ID"

#: The NLQ column is the baseline throughout RQ2.
BASELINE_COL   = "nlq_response"
BASELINE_LABEL = "NLQ"

#: SPARQL is the primary challenger; CoT variants are secondary.
#: Keys are DataFrame column names; values are canonical display labels.
CHALLENGERS: dict = {
    "sparql_nlq_response": "SPARQL",       # primary FLQ challenger
    "nlq_cot_response":    "NLQ_CoT",      # CoT reformulation challenger
    "sparql_cot_response": "SPARQL_CoT",   # hybrid CoT challenger
}

#: Primary challenger column (the FLQ / SPARQL method).
PRIMARY_CHALLENGER_COL   = "sparql_nlq_response"
PRIMARY_CHALLENGER_LABEL = "SPARQL"

#: Column aliases for older workbook variants.
COLUMN_ALIASES: dict = {
    "sparql_response": "sparql_nlq_response",
    "cot_response":    "nlq_cot_response",
}

#: All columns evaluated: baseline plus each challenger.
ALL_CANDIDATE_COLS: list = [BASELINE_COL] + list(CHALLENGERS.keys())


# ---------------------------------------------------------------------------
# Plot style constants
# ---------------------------------------------------------------------------

#: Method colour palette, fixed across all figures.
#: Uses a colourblind-accessible scheme.
METHOD_COLOURS: dict = {
    "SPARQL":     "#1B4F72",   # deep navy   -- primary FLQ challenger
    "NLQ_CoT":    "#B7950B",   # dark gold   -- CoT reformulation
    "SPARQL_CoT": "#7B241C",   # dark red    -- hybrid CoT
    "NLQ":        "#616A6B",   # neutral grey -- baseline
}

METHOD_HATCHES: dict = {
    "SPARQL":     "",
    "NLQ_CoT":    "///",
    "SPARQL_CoT": "xxx",
    "NLQ":        "...",
}

METHOD_DISPLAY: dict = {
    "SPARQL":     "SPARQL (FLQ)",
    "NLQ_CoT":    "NLQ+CoT",
    "SPARQL_CoT": "SPARQL+CoT",
    "NLQ":        "NLQ (baseline)",
}

#: Canonical domain ordering used in all figures and tables.
KNOWN_DOMAIN_ORDER = ["TelcoQA", "FiQA", "LegalQA"]

#: Shared rcParams applied to all figures.
_RC: dict = {
    "font.family":           "DejaVu Sans",
    "font.size":             14,
    "axes.titlesize":        16,
    "axes.labelsize":        15,
    "xtick.labelsize":       13,
    "ytick.labelsize":       13,
    "legend.fontsize":       13,
    "legend.title_fontsize": 14,
    "axes.linewidth":        1.4,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "xtick.minor.visible":   False,
    "ytick.minor.visible":   False,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.15,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
}

#: Activate log y-axis in the CR boxplot when max(CR)/min(CR) exceeds this.
#: Set to np.inf to always use linear scale.
LOG_SCALE_RATIO_THRESHOLD = 20.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ordered_domains(domains_present) -> list:
    """Return domains in canonical order, with any unknown domains appended."""
    return (
        [d for d in KNOWN_DOMAIN_ORDER if d in domains_present]
        + [d for d in sorted(domains_present) if d not in KNOWN_DOMAIN_ORDER]
    )


def _save_figure(fig, out_dir: Path, stem: str) -> None:
    """Save *fig* as both PNG (300 dpi) and PDF under *out_dir*/plots/."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(str(plots_dir / f"{stem}.{ext}"))
    log.info("Saved: plots/%s.{png,pdf}", stem)


# ---------------------------------------------------------------------------
# Figure -- absolute CR distributions (box + strip, faceted by domain)
# ---------------------------------------------------------------------------

def plot_cr_absolute_boxplot(delta_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Box-and-strip plot of absolute Claim Recall distributions by method,
    faceted by domain, on recall-winning queries.

    Each dot represents the mean CR for one (model x domain) cell on that
    cell's recall-winning queries.  The plot has one panel per domain so
    that cross-domain patterns are immediately visible.

    NLQ deduplication
    -----------------
    delta_df has one row per (challenger x model x domain).  The NLQ CR
    value (nlq_cr_mean_on_rw) is identical across all challenger rows for
    the same (model, domain) cell because it derives from the same NLQ
    column.  To avoid inflating the NLQ distribution by a factor of
    n_challengers, NLQ entries are collected in a separate pass that visits
    each (model, domain) cell exactly once (using the first row encountered
    as the source of nlq_cr_mean_on_rw).  Challenger CR entries are
    collected independently with no change to their count logic.

    Y-axis scaling
    --------------
    A log y-axis is activated automatically when the ratio
    max(CR) / max(0.01, min(CR)) across all finite values exceeds
    LOG_SCALE_RATIO_THRESHOLD.  This preserves perceptual separation of
    tightly clustered values without distorting comparisons.

    Parameters
    ----------
    delta_df:
        Output of :func:`compute_delta_cr_recall_wins`.
    out_dir:
        Root output directory; the figure is written to out_dir/plots/.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update(_RC)

    domains         = _ordered_domains(delta_df["domain"].unique())
    present_methods = [m for m in ["NLQ", "SPARQL", "NLQ_CoT", "SPARQL_CoT"]]

    # ------------------------------------------------------------------
    # Collect per-method, per-domain CR arrays with NLQ deduplication.
    # ------------------------------------------------------------------
    def _collect_domain_arrays(
        sub_df: pd.DataFrame,
    ) -> dict:
        """
        Return {method: list[float]} for *sub_df* (a single-domain slice).
        NLQ is collected once per (model, domain) cell; challengers are
        collected once per (challenger x model x domain) row.
        """
        ch_lists: dict = {m: [] for m in CHALLENGERS.values()}
        nlq_seen: set  = set()
        nlq_list: list = []

        for _, row in sub_df.iterrows():
            method = row["method"]
            ch_cr  = row.get("ch_cr_mean_on_rw", np.nan)
            nlq_cr = row.get("nlq_cr_mean_on_rw", np.nan)

            if np.isfinite(float(ch_cr)):
                ch_lists[method].append(float(ch_cr))

            cell_key = (row["model"], row["domain"])
            if cell_key not in nlq_seen and np.isfinite(float(nlq_cr)):
                nlq_list.append(float(nlq_cr))
                nlq_seen.add(cell_key)

        result = {"NLQ": np.array(nlq_list)}
        for m, lst in ch_lists.items():
            result[m] = np.array(lst)
        return result

    # ------------------------------------------------------------------
    # Determine whether to use log scale (global decision across panels).
    # ------------------------------------------------------------------
    all_finite: list = []
    for domain in domains:
        arrays = _collect_domain_arrays(delta_df[delta_df["domain"] == domain])
        for arr in arrays.values():
            all_finite.extend(arr[np.isfinite(arr) & (arr > 0)].tolist())

    use_log = False
    if len(all_finite) > 1:
        ratio   = float(np.max(all_finite)) / max(float(np.min(all_finite)), 1e-6)
        use_log = ratio > LOG_SCALE_RATIO_THRESHOLD

    # ------------------------------------------------------------------
    # Build figure: one panel per domain.
    # ------------------------------------------------------------------
    n_domains = len(domains)
    fig, axes = plt.subplots(
        1, n_domains,
        figsize=(max(7, 2.8 * len(present_methods)) * n_domains, 6.5),
        sharey=True,
    )
    if n_domains == 1:
        axes = [axes]

    positions = np.arange(len(present_methods))

    for ax, domain in zip(axes, domains):
        dom_df  = delta_df[delta_df["domain"] == domain]
        arrays  = _collect_domain_arrays(dom_df)
        methods = [m for m in present_methods if m in arrays and len(arrays[m]) > 0]

        for pi, method in enumerate(methods):
            vals = arrays[method]
            if len(vals) == 0:
                continue

            is_primary = (method == "SPARQL")

            bp = ax.boxplot(
                vals,
                positions=[pi],
                widths=0.45,
                patch_artist=True,
                notch=False,
                vert=True,
                showfliers=False,
                medianprops=dict(color="#000000", linewidth=2.5),
                boxprops=dict(
                    facecolor=METHOD_COLOURS.get(method, "#AAAAAA"),
                    alpha=0.55,
                    linewidth=2.8 if is_primary else 1.2,
                    edgecolor=METHOD_COLOURS.get(method, "#333333"),
                ),
                whiskerprops=dict(linewidth=1.4, linestyle="--"),
                capprops=dict(linewidth=2.0),
            )

            if is_primary:
                for patch in bp["boxes"]:
                    patch.set_linewidth(2.8)

            # Jittered strip overlay -- fixed seed for reproducibility.
            rng    = np.random.default_rng(42)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                pi + jitter, vals,
                color=METHOD_COLOURS.get(method, "#888888"),
                alpha=0.55, s=40, zorder=4,
                edgecolors="#333333", linewidths=0.5,
            )

            # Mean diamond.
            ax.scatter(
                pi, float(np.mean(vals)),
                color="#FFFFFF", marker="D", s=80, zorder=5,
                edgecolors=METHOD_COLOURS.get(method, "#333333"), linewidths=2.0,
            )

            # Annotate N (number of model x domain cells represented).
            y_annot = (
                min(all_finite) * 0.85
                if (use_log and all_finite)
                else ax.get_ylim()[0]
            )
            ax.text(
                pi, y_annot,
                f"N={len(vals)}",
                ha="center", va="top",
                fontsize=9, color="#555555",
            )

        ax.set_xticks(positions[:len(methods)])
        ax.set_xticklabels(
            [METHOD_DISPLAY.get(m, m) for m in methods],
            fontsize=12, fontweight="bold",
        )
        ax.set_title(domain, fontsize=15, fontweight="bold", pad=8)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, max(len(methods) - 0.4, 0.4))

    if use_log:
        axes[0].set_yscale("log")
        y_label = (
            "Absolute Claim Recall (CR) -- log scale\n"
            "on Recall-Winning Queries"
        )
    else:
        y_label = "Absolute Claim Recall (CR)\non Recall-Winning Queries"

    axes[0].set_ylabel(y_label, fontsize=14, fontweight="bold")

    # Shared colour-swatch legend on the last panel.
    from matplotlib.patches import Patch
    patches = [
        Patch(
            facecolor=METHOD_COLOURS.get(m, "#AAA"),
            edgecolor=METHOD_COLOURS.get(m, "#333"),
            label=METHOD_DISPLAY.get(m, m) + (" (primary)" if m == "SPARQL" else ""),
            linewidth=2.5 if m == "SPARQL" else 1.2,
            alpha=0.7,
        )
        for m in [m for m in present_methods if m in
                  {mm for d in domains for mm in
                   _collect_domain_arrays(delta_df[delta_df["domain"] == d]).keys()}]
    ]
    legend = axes[-1].legend(
        handles=patches,
        title="Method",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True, framealpha=0.95,
        edgecolor="#AAAAAA",
    )
    legend.get_title().set_fontweight("bold")

    fig.suptitle(
        "Absolute Claim Recall by Method -- Recall-Winning Queries, Faceted by Domain\n"
        "(each dot = one model x domain mean; diamond = grand mean; box = IQR)",
        fontsize=15, y=1.01,
    )
    fig.tight_layout()
    _save_figure(fig, out_dir, "fig_cr_absolute_boxplot")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX table -- recall win-rate comparison (SPARQL-centric delta-pp)
# ---------------------------------------------------------------------------

def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters in plain text."""
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_",  "\\_")
        .replace("%",  "\\%")
        .replace("&",  "\\&")
        .replace("#",  "\\#")
        .replace("{",  "\\{")
        .replace("}",  "\\}")
    )


def _winrate_delta_cell(v: float, precision: int = 1) -> str:
    """
    Render a signed win-rate delta (percentage points) as a LaTeX
    colour-coded cell.  Green = SPARQL leads; red = CoT leads.
    """
    if not np.isfinite(v):
        return "--"
    colour = "posval" if v > 0 else "negval"
    sign   = "+" if v > 0 else ""
    fmt    = f"{sign}{v:.{precision}f}"
    return f"\\textcolor{{{colour}}}{{{fmt}}}"


def _latex_preamble(landscape: bool = True) -> list:
    """Return the standard LaTeX document preamble lines."""
    geometry = (
        "a4paper, margin=1.5cm, landscape"
        if landscape
        else "a4paper, margin=2cm"
    )
    return [
        r"\documentclass[10pt]{article}",
        f"\\usepackage[{geometry}]{{geometry}}",
        r"\usepackage{booktabs}",
        r"\usepackage{colortbl}",
        r"\usepackage{xcolor}",
        r"\usepackage{multirow}",
        r"\usepackage{array}",
        r"\usepackage{caption}",
        r"\usepackage{makecell}",
        r"\usepackage{amsmath}",
        r"\definecolor{posval}{rgb}{0.05,0.40,0.10}",
        r"\definecolor{negval}{rgb}{0.70,0.06,0.06}",
        r"\definecolor{rowalt}{rgb}{0.97,0.97,0.97}",
        r"\definecolor{sparqlrow}{rgb}{0.88,0.96,0.88}",
        r"\definecolor{cot1row}{rgb}{0.96,0.93,0.80}",
        r"\definecolor{cot2row}{rgb}{0.95,0.88,0.88}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{4pt}",
    ]


def write_recall_win_rate_table(
    delta_df: pd.DataFrame,
    out_path: str,
) -> None:
    """
    Write the SPARQL-centric recall win-rate comparison table to a LaTeX file.

    The table provides an exact numerical companion to the CR boxplot figure
    by reporting:

    * BERTScore Recall win-rate (%) for SPARQL, NLQ_CoT, and SPARQL_CoT
      against the NLQ baseline, per (model x domain) cell.
    * Absolute win counts (n_rw / n_total) for reproducibility.
    * Signed win-rate advantage columns (delta-pp):
        - delta_1 = SPARQL win-rate minus NLQ_CoT win-rate.
        - delta_2 = SPARQL win-rate minus SPARQL_CoT win-rate.
    * A footer row reporting column-wise means across models per domain.

    Colour coding: delta-pp > 0 rendered in green (SPARQL leads);
    delta-pp <= 0 rendered in red (CoT leads).

    Parameters
    ----------
    delta_df:
        Output of :func:`compute_delta_cr_recall_wins`.
    out_path:
        Full path to the output .tex file.
    """
    all_domains = sorted(delta_df["domain"].unique())
    domains     = _ordered_domains(all_domains)
    models      = sorted(delta_df["model"].unique())

    # Fast lookup: (method, model, domain) -> row dict.
    lk: dict = {}
    for _, r in delta_df.iterrows():
        lk[(r["method"], r["model"], r["domain"])] = r.to_dict()

    # Five sub-columns per domain:
    #   SPARQL Win%  |  NLQ_CoT Win%  |  SPARQL_CoT Win%
    #   |  delta(SPARQL-NLQ_CoT)  |  delta(SPARQL-SPARQL_CoT)
    n_domains  = len(domains)
    sub_cols   = 5
    total_cols = 1 + sub_cols * n_domains

    lines = _latex_preamble(landscape=True)
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{%",
        (
            r"  BERTScore Recall win-rate by model, domain, and method. "
            r"  Win-rate: percentage of queries where the challenger's "
            r"  BERTScore Recall exceeds the NLQ baseline. "
            r"  Counts in parentheses: winning queries over total evaluated. "
            r"  $\Delta_1$: SPARQL win-rate minus NLQ\_CoT win-rate "
            r"  (positive = SPARQL leads). "
            r"  $\Delta_2$: SPARQL win-rate minus SPARQL\_CoT win-rate. "
            r"  \colorbox{sparqlrow}{\strut Shaded}: SPARQL win-rates. "
            r"  \textcolor{posval}{Green}\,/\,\textcolor{negval}{Red}: "
            r"  SPARQL advantage\,/\,deficit."
        ),
        r"\label{tab:recall_win_rates}",
    ]

    col_spec = "@{}l" + "rrrrr" * n_domains + "@{}"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: domain group spans.
    dom_header = [r"\multirow{3}{*}{\textbf{Model}}"]
    for d in domains:
        dom_header.append(
            f"\\multicolumn{{{sub_cols}}}{{c}}{{\\textbf{{{_latex_escape(d)}}}}}"
        )
    lines.append(" & ".join(dom_header) + r" \\")

    # Cmidrule per domain group.
    cmidrules = []
    for di in range(n_domains):
        sc = 2 + sub_cols * di
        ec = sc + sub_cols - 1
        cmidrules.append(f"\\cmidrule(lr){{{sc}-{ec}}}")
    lines.append(" ".join(cmidrules))

    # Header row 2: method span and advantage span.
    method_span_header = [""]
    for _ in domains:
        method_span_header.append(
            r"\multicolumn{3}{c}{\textit{Win-Rate (\%)}}"
        )
        method_span_header.append(
            r"\multicolumn{2}{c}{\textit{SPARQL Advantage ($\Delta$pp)}}"
        )
    lines.append(" & ".join(method_span_header) + r" \\")

    # Cmidrule for win-rate and advantage sub-groups.
    cmidrules2 = []
    for di in range(n_domains):
        sc = 2 + sub_cols * di
        cmidrules2.append(f"\\cmidrule(lr){{{sc}-{sc + 2}}}")
        cmidrules2.append(f"\\cmidrule(lr){{{sc + 3}-{sc + 4}}}")
    lines.append(" ".join(cmidrules2))

    # Header row 3: individual sub-column labels.
    subheader = [""]
    for _ in domains:
        subheader.append(r"\makecell{SPARQL\\(FLQ)}")
        subheader.append(r"\makecell{NLQ\\+CoT}")
        subheader.append(r"\makecell{SPARQL\\+CoT}")
        subheader.append(r"$\Delta_1$")
        subheader.append(r"$\Delta_2$")
    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\midrule")

    # Accumulator for footer means.
    footer_accum: dict = {
        d: {"sp": [], "cot1": [], "cot2": [], "d1": [], "d2": []}
        for d in domains
    }

    # Data rows.
    for mi, model in enumerate(models):
        row_color = "sparqlrow" if mi % 2 == 0 else "rowalt"
        cells = [f"\\textbf{{{_latex_escape(model)}}}"]

        for domain in domains:
            sp_row   = lk.get(("SPARQL",     model, domain))
            cot1_row = lk.get(("NLQ_CoT",    model, domain))
            cot2_row = lk.get(("SPARQL_CoT", model, domain))

            def _wr_cell(r, shade: bool = False) -> str:
                """Format win-rate with count; optionally apply cell shading."""
                if r is None:
                    return "--"
                wr  = r["recall_win_rate"]
                nrw = r["recall_win_count"]
                nt  = r["n_queries_total"]
                if not np.isfinite(wr):
                    return "--"
                val_str = f"{wr:.1f} ({int(nrw)}/{int(nt)})"
                return f"\\cellcolor{{sparqlrow}}{val_str}" if shade else val_str

            def _adv(sp_r, cmp_r) -> float:
                """Return signed win-rate advantage (SPARQL minus comparator)."""
                if sp_r is None or cmp_r is None:
                    return float("nan")
                sv = sp_r["recall_win_rate"]
                cv = cmp_r["recall_win_rate"]
                if not (np.isfinite(sv) and np.isfinite(cv)):
                    return float("nan")
                return sv - cv

            d1 = _adv(sp_row, cot1_row)
            d2 = _adv(sp_row, cot2_row)

            cells.append(_wr_cell(sp_row, shade=True))
            cells.append(_wr_cell(cot1_row))
            cells.append(_wr_cell(cot2_row))
            cells.append(_winrate_delta_cell(d1))
            cells.append(_winrate_delta_cell(d2))

            # Accumulate for footer.
            if sp_row   is not None and np.isfinite(sp_row["recall_win_rate"]):
                footer_accum[domain]["sp"].append(sp_row["recall_win_rate"])
            if cot1_row is not None and np.isfinite(cot1_row["recall_win_rate"]):
                footer_accum[domain]["cot1"].append(cot1_row["recall_win_rate"])
            if cot2_row is not None and np.isfinite(cot2_row["recall_win_rate"]):
                footer_accum[domain]["cot2"].append(cot2_row["recall_win_rate"])
            if np.isfinite(d1):
                footer_accum[domain]["d1"].append(d1)
            if np.isfinite(d2):
                footer_accum[domain]["d2"].append(d2)

        lines.append(f"\\rowcolor{{{row_color}}}" + " & ".join(cells) + r" \\")

    # Footer: column-wise means.
    lines.append(r"\midrule")
    footer_cells = [r"\textit{Mean}"]
    for domain in domains:
        acc    = footer_accum[domain]
        sp_m   = float(np.mean(acc["sp"]))   if acc["sp"]   else float("nan")
        cot1_m = float(np.mean(acc["cot1"])) if acc["cot1"] else float("nan")
        cot2_m = float(np.mean(acc["cot2"])) if acc["cot2"] else float("nan")
        d1_m   = float(np.mean(acc["d1"]))   if acc["d1"]   else float("nan")
        d2_m   = float(np.mean(acc["d2"]))   if acc["d2"]   else float("nan")

        footer_cells.append(
            f"\\cellcolor{{sparqlrow}}\\textit{{{sp_m:.1f}}}"
            if np.isfinite(sp_m) else "--"
        )
        footer_cells.append(
            f"\\textit{{{cot1_m:.1f}}}" if np.isfinite(cot1_m) else "--"
        )
        footer_cells.append(
            f"\\textit{{{cot2_m:.1f}}}" if np.isfinite(cot2_m) else "--"
        )
        footer_cells.append(
            _winrate_delta_cell(d1_m) if np.isfinite(d1_m) else "--"
        )
        footer_cells.append(
            _winrate_delta_cell(d2_m) if np.isfinite(d2_m) else "--"
        )

    lines.append(r"\rowcolor{white}" + " & ".join(footer_cells) + r" \\")

    lines += [
        r"\bottomrule",
        f"\\multicolumn{{{total_cols}}}{{l}}{{%",
        r"  \footnotesize",
        (
            r"  Win-Rate (\%): fraction of queries where challenger "
            r"  BERTScore Recall $>$ NLQ BERTScore Recall. "
            r"  Counts: winning queries / total evaluated per cell. "
            r"  $\Delta$pp = percentage-point difference. "
            r"  SPARQL = primary FLQ challenger; "
            r"  CoT1 = NLQ\_CoT; CoT2 = SPARQL\_CoT. "
            r"  NLQ = natural language query baseline (not shown; "
            r"  win-rate is defined relative to it)."
        ),
        r"} \\",
        r"\end{tabular}",
        r"\end{table}",
        r"\end{document}",
    ]

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    log.info("LaTeX recall win-rate table written: %s", out_path)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalise and strip a text string; return '[EMPTY]' if blank."""
    if not text or not str(text).strip():
        return "[EMPTY]"
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def _norm_qid(v) -> str:
    """Normalise a Query_ID to a zero-padded six-digit string."""
    try:
        return f"{int(float(str(v).strip())):06d}"
    except Exception:
        return str(v).strip()


def _extract_claims(text: str) -> list:
    """
    Extract discrete claims from a response.

    Sentences of five or more words are retained as claims; shorter
    fragments are excluded as they rarely carry verifiable propositional
    content.
    """
    sentences = _sentence_tokenise(text)
    return [s for s in sentences if len(s.split()) >= 5]


def _sentence_tokenise(text: str) -> list:
    """Tokenise *text* into sentences; falls back to regex split if NLTK
    is unavailable."""
    try:
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_NLI_PIPELINE = None


def _get_nli_pipeline():
    """
    Lazy-load the cross-encoder NLI pipeline.

    Model: cross-encoder/nli-deberta-v3-base.
    Loaded once and cached in the module-level variable ``_NLI_PIPELINE``.
    """
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        from transformers import pipeline as hf_pipeline
        device_id = 0 if _device() == "cuda" else -1
        _NLI_PIPELINE = hf_pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-base",
            device=device_id,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        log.info("NLI pipeline loaded (cross-encoder/nli-deberta-v3-base).")
    return _NLI_PIPELINE


def compute_claim_recall(hyp: str, ref: str) -> float:
    """
    Compute Claim Recall for a single (hypothesis, reference) pair.

        CR = |{c in C(h) : g* |= c}| / |C(h)|

    where C(h) is the set of sentences in h with five or more words.

    Parameters
    ----------
    hyp:
        Candidate response.
    ref:
        Ground-truth reference text.

    Returns
    -------
    float
        CR in [0, 1], or NaN if no qualifying claims exist or ref is empty.
    """
    claims = _extract_claims(hyp)
    if not claims or ref == "[EMPTY]":
        return float("nan")

    nli       = _get_nli_pipeline()
    n_entailed = 0
    for claim in claims:
        try:
            result = nli(f"{ref[:256]} [SEP] {claim[:256]}")
            best   = max(result[0], key=lambda x: x["score"])
            if best["label"].upper() == "ENTAILMENT":
                n_entailed += 1
        except Exception as exc:
            log.debug("NLI inference error (claim skipped): %s", exc)
    return n_entailed / len(claims)


def compute_bertscore_recall_batch(
    hyps: list,
    refs: list,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute BERTScore Recall for a batch of (hypothesis, reference) pairs.

    Parameters
    ----------
    hyps:
        List of candidate response strings.
    refs:
        List of reference strings (same length as *hyps*).
    device:
        Torch device string ('cpu' or 'cuda').

    Returns
    -------
    numpy.ndarray
        Float array of recall scores, length ``len(hyps)``.
    """
    from bert_score import score as bs_score
    _, r_tensor, _ = bs_score(
        hyps, refs,
        lang="en",
        device=device,
        verbose=False,
        rescale_with_baseline=False,
    )
    return r_tensor.numpy()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(xl: pd.ExcelFile) -> dict:
    """
    Load the ground_truth sheet and return a ``{query_id: text}`` mapping.

    Parameters
    ----------
    xl:
        Open ExcelFile handle.

    Returns
    -------
    dict
        Mapping from normalised query ID strings to cleaned ground-truth
        text strings.

    Raises
    ------
    ValueError
        If the ground_truth sheet or its required columns are absent.
    """
    sheet_map = {s.lower(): s for s in xl.sheet_names}
    gt_sheet  = sheet_map.get(GT_SHEET.lower())
    if gt_sheet is None:
        raise ValueError(
            f"Workbook must contain a '{GT_SHEET}' sheet. "
            f"Found: {xl.sheet_names}"
        )

    gt = xl.parse(gt_sheet, dtype=str)
    gt.columns = [c.strip() for c in gt.columns]
    col_lower  = {c.lower(): c for c in gt.columns}

    qid_col = col_lower.get(GT_QID_COL.lower()) or col_lower.get("query_id")
    txt_col = col_lower.get(GT_TEXT_COL.lower()) or col_lower.get("ground_truth")

    if qid_col is None or txt_col is None:
        raise ValueError(
            f"Ground-truth sheet requires '{GT_QID_COL}' and '{GT_TEXT_COL}'. "
            f"Found columns: {list(gt.columns)}"
        )

    return {
        _norm_qid(qid): _clean(str(txt))
        for qid, txt in zip(
            gt[qid_col].values,
            gt[txt_col].fillna("").values,
        )
    }


def _resolve_domain(
    sheet_name: str,
    df: pd.DataFrame,
    domain_col: str | None,
) -> str:
    """
    Determine the domain label for a sheet.

    If *domain_col* is given and present in *df*, its first unique non-null
    value is returned.  Otherwise the sheet name itself is used.
    """
    if domain_col and domain_col in df.columns:
        vals = df[domain_col].dropna().unique()
        if len(vals) > 0:
            return str(vals[0]).strip()
    return sheet_name.strip()


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column aliases and normalise the Query_ID column name.

    Parameters
    ----------
    df:
        Raw sheet data frame.

    Returns
    -------
    pandas.DataFrame
        Data frame with canonical column names.

    Raises
    ------
    ValueError
        If no Query_ID column (under any case variant) is found.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    col_lower  = {c.lower(): c for c in df.columns}
    qid_actual = col_lower.get("query_id")
    if qid_actual is None:
        raise ValueError(
            f"Sheet has no 'Query_ID' column. Columns found: {list(df.columns)}"
        )
    if qid_actual != "Query_ID":
        df.rename(columns={qid_actual: "Query_ID"}, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Per-sheet evaluation
# ---------------------------------------------------------------------------

def evaluate_sheet(
    df: pd.DataFrame,
    gt_lookup: dict,
    model_name: str,
    domain: str,
    run_full: bool,
) -> pd.DataFrame:
    """
    Compute per-query BERTScore Recall and Claim Recall for every candidate
    column present in *df*.

    Parameters
    ----------
    df:
        Normalised sheet data frame.
    gt_lookup:
        Mapping from query ID to ground-truth text.
    model_name:
        Label identifying the model (used in output rows).
    domain:
        Domain label (used in output rows).
    run_full:
        If ``True``, run BERTScore and NLI inference; otherwise fill with NaN.

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns: model, domain, candidate_col,
        method_label, query_id, bertscore_recall, claim_recall.
    """
    qids = [_norm_qid(v) for v in df["Query_ID"].values]
    refs = [_clean(gt_lookup.get(q, "[EMPTY]")) for q in qids]
    dev  = _device()
    rows: list = []

    for col in ALL_CANDIDATE_COLS:
        if col not in df.columns:
            log.warning(
                "Model='%s' domain='%s': column '%s' missing -- skipped.",
                model_name, domain, col,
            )
            continue

        hyps = [_clean(str(v)) for v in df[col].fillna("").values]

        # BERTScore Recall.
        if run_full:
            log.info(
                "[BERTScore R] model=%s domain=%s col=%s n=%d",
                model_name, domain, col, len(hyps),
            )
            bs_r = compute_bertscore_recall_batch(hyps, refs, dev)
        else:
            bs_r = np.full(len(hyps), float("nan"))

        # Claim Recall (per query).
        cr_vals: list = []
        if run_full:
            log.info(
                "[Claim Recall] model=%s domain=%s col=%s",
                model_name, domain, col,
            )
            for hyp, ref in zip(hyps, refs):
                cr_vals.append(compute_claim_recall(hyp, ref))
        else:
            cr_vals = [float("nan")] * len(hyps)

        method_label = CHALLENGERS.get(
            col,
            BASELINE_LABEL if col == BASELINE_COL else col,
        )

        for i, qid in enumerate(qids):
            rows.append({
                "model":            model_name,
                "domain":           domain,
                "candidate_col":    col,
                "method_label":     method_label,
                "query_id":         qid,
                "bertscore_recall": (float(bs_r[i])
                                     if np.isfinite(bs_r[i]) else float("nan")),
                "claim_recall":     (float(cr_vals[i])
                                     if np.isfinite(cr_vals[i]) else float("nan")),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core RQ2 analysis: delta-CR on recall-winning queries
# ---------------------------------------------------------------------------

def compute_delta_cr_recall_wins(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (challenger x model x domain), compute delta-CR restricted to
    recall-winning queries.

    Step 1 -- Recall-winning queries:
        Queries where the challenger's BERTScore Recall exceeds the NLQ
        baseline's BERTScore Recall.

    Step 2 -- delta-CR on recall-winning queries:
        delta-CR = CR_CH - CR_NLQ  (computed on recall-winning queries only).

        delta-CR > 0  and co-directional (up):   recall gain accompanied by
                                                  factual grounding gain.
        delta-CR <= 0 and non-co-directional (down): recall gain carries no
                                                     factual grounding benefit.

    Step 3 -- Output metadata per (challenger x model x domain):
        recall_win_rate      -- % of queries where challenger > NLQ recall.
        recall_win_count     -- absolute count of recall-winning queries.
        n_queries_total      -- total comparable query count.
        delta_cr_on_rw       -- mean delta-CR on recall-winning queries.
        ch_cr_mean_on_rw     -- mean CR of challenger on recall-winning queries.
        nlq_cr_mean_on_rw    -- mean CR of NLQ on recall-winning queries.
        delta_recall_on_rw   -- mean recall gain on recall-winning queries.
        co_directional       -- True if delta-CR > 0.
        cd_arrow             -- 'up' or 'down' string indicator.

    Parameters
    ----------
    all_df:
        Concatenated per-query metrics data frame from
        :func:`evaluate_sheet`.

    Returns
    -------
    pandas.DataFrame
        One row per (challenger x model x domain).
    """
    baseline_df = (
        all_df[all_df["candidate_col"] == BASELINE_COL]
        .copy()
        .set_index(["model", "domain", "query_id"])
        [["bertscore_recall", "claim_recall"]]
        .rename(columns={
            "bertscore_recall": "nlq_recall",
            "claim_recall":     "nlq_cr",
        })
    )

    result_rows: list = []

    for ch_col, ch_label in CHALLENGERS.items():
        ch_df = (
            all_df[all_df["candidate_col"] == ch_col]
            .copy()
            .set_index(["model", "domain", "query_id"])
            [["bertscore_recall", "claim_recall"]]
            .rename(columns={
                "bertscore_recall": "ch_recall",
                "claim_recall":     "ch_cr",
            })
        )

        merged = ch_df.join(baseline_df, how="inner").reset_index()

        for (model, domain), grp in merged.groupby(["model", "domain"]):
            g = grp.copy()

            # Identify recall-winning queries.
            has_recall      = np.isfinite(g["ch_recall"]) & np.isfinite(g["nlq_recall"])
            recall_win_mask = has_recall & (g["ch_recall"] > g["nlq_recall"])

            n_total  = int(has_recall.sum())
            n_rw     = int(recall_win_mask.sum())
            win_rate = (
                100.0 * n_rw / n_total if n_total > 0 else float("nan")
            )

            # Restrict to recall-winning queries.
            rw = g[recall_win_mask].copy()

            # Delta-recall on recall-winning queries.
            r_mask = np.isfinite(rw["ch_recall"]) & np.isfinite(rw["nlq_recall"])
            if r_mask.sum() > 0:
                delta_recall_rw = float(np.mean(
                    rw.loc[r_mask, "ch_recall"].values
                    - rw.loc[r_mask, "nlq_recall"].values
                ))
            else:
                delta_recall_rw = float("nan")

            # Delta-CR on recall-winning queries (primary RQ2 metric).
            cr_mask = np.isfinite(rw["ch_cr"]) & np.isfinite(rw["nlq_cr"])
            n_rw_cr = int(cr_mask.sum())
            if n_rw_cr > 0:
                ch_cr_vals  = rw.loc[cr_mask, "ch_cr"].values
                nlq_cr_vals = rw.loc[cr_mask, "nlq_cr"].values
                delta_cr_rw    = float(np.mean(ch_cr_vals - nlq_cr_vals))
                ch_cr_mean_rw  = float(np.mean(ch_cr_vals))
                nlq_cr_mean_rw = float(np.mean(nlq_cr_vals))
            else:
                delta_cr_rw = ch_cr_mean_rw = nlq_cr_mean_rw = float("nan")

            # Co-directionality: delta-CR > 0 on recall-winning queries.
            co_directional = bool(
                np.isfinite(delta_cr_rw) and delta_cr_rw > 0
            )

            is_primary = (ch_col == PRIMARY_CHALLENGER_COL)

            result_rows.append({
                "challenger_col":        ch_col,
                "method":                ch_label,
                "is_primary_sparql":     is_primary,
                "model":                 model,
                "domain":                domain,
                "n_queries_total":       n_total,
                "recall_win_count":      n_rw,
                "recall_win_rate":       (
                    round(win_rate, 2) if np.isfinite(win_rate) else float("nan")
                ),
                "delta_recall_on_rw":    (
                    round(delta_recall_rw, 4)
                    if np.isfinite(delta_recall_rw) else float("nan")
                ),
                "n_rw_with_cr":          n_rw_cr,
                "ch_cr_mean_on_rw":      (
                    round(ch_cr_mean_rw, 4)
                    if np.isfinite(ch_cr_mean_rw) else float("nan")
                ),
                "nlq_cr_mean_on_rw":     (
                    round(nlq_cr_mean_rw, 4)
                    if np.isfinite(nlq_cr_mean_rw) else float("nan")
                ),
                "delta_cr_on_rw":        (
                    round(delta_cr_rw, 4)
                    if np.isfinite(delta_cr_rw) else float("nan")
                ),
                "co_directional":        co_directional,
                "cd_arrow":              "up" if co_directional else "down",
            })

    return pd.DataFrame(result_rows)


def compute_challenger_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate delta-CR statistics across models and domains per challenger.

    Returns a data frame with one row per challenger method, including:
    mean recall win-rate, mean delta-CR, co-directionality rate, and
    per-domain mean delta-CR columns.

    Parameters
    ----------
    delta_df:
        Output of :func:`compute_delta_cr_recall_wins`.
    """
    rows: list = []

    for ch_label, grp in delta_df.groupby("method"):
        n_cells    = len(grp)
        is_primary = bool(grp["is_primary_sparql"].iloc[0])

        mean_win_rate = float(np.nanmean(grp["recall_win_rate"].values))
        mean_delta_cr = float(np.nanmean(grp["delta_cr_on_rw"].values))
        n_codir       = int(grp["co_directional"].sum())
        pct_codir     = (
            100.0 * n_codir / n_cells if n_cells > 0 else float("nan")
        )

        per_domain: dict = {}
        for domain, dgrp in grp.groupby("domain"):
            per_domain[f"mean_delta_cr_{domain}"] = float(
                np.nanmean(dgrp["delta_cr_on_rw"].values)
            )

        row = {
            "method":                ch_label,
            "is_primary_sparql":     is_primary,
            "n_model_domain_cells":  n_cells,
            "mean_recall_win_rate":  (
                round(mean_win_rate, 2)
                if np.isfinite(mean_win_rate) else float("nan")
            ),
            "mean_delta_cr_on_rw":   (
                round(mean_delta_cr, 4)
                if np.isfinite(mean_delta_cr) else float("nan")
            ),
            "n_codir":               n_codir,
            "pct_codir":             (
                round(pct_codir, 1)
                if np.isfinite(pct_codir) else float("nan")
            ),
        }
        row.update(per_domain)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["is_primary_sparql", "mean_recall_win_rate"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def compute_sparql_primary_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    """
    SPARQL-centric summary: per (model x domain) delta-CR with comparison
    to CoT variants.

    Parameters
    ----------
    delta_df:
        Output of :func:`compute_delta_cr_recall_wins`.

    Returns
    -------
    pandas.DataFrame
        Wide-format table pivoted on method, one row per (model x domain).
    """
    pivot = delta_df.pivot_table(
        index=["model", "domain"],
        columns="method",
        values=[
            "delta_cr_on_rw", "delta_recall_on_rw",
            "recall_win_rate", "recall_win_count",
            "ch_cr_mean_on_rw", "nlq_cr_mean_on_rw", "co_directional",
        ],
        aggfunc="first",
    )
    pivot.columns = [
        "_".join(str(c) for c in col).strip()
        for col in pivot.columns
    ]
    pivot = pivot.reset_index()

    sparql_col   = f"delta_cr_on_rw_{PRIMARY_CHALLENGER_LABEL}"
    nlq_cot_col  = "delta_cr_on_rw_NLQ_CoT"

    for c in [sparql_col, nlq_cot_col]:
        if c not in pivot.columns:
            pivot[c] = float("nan")

    def _sparql_vs_cot(row) -> float:
        sv  = row.get(sparql_col, float("nan"))
        ncv = row.get(nlq_cot_col, float("nan"))
        if not (np.isfinite(sv) and np.isfinite(ncv)):
            return float("nan")
        return sv - ncv

    pivot["sparql_delta_cr_vs_nlq_cot"] = pivot.apply(_sparql_vs_cot, axis=1)

    float_cols = pivot.select_dtypes(include=[float]).columns
    pivot[float_cols] = pivot[float_cols].round(4)

    return pivot.sort_values(["model", "domain"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    responses_xlsx: str,
    output_dir: str,
    run_full: bool = True,
    domain_col: str | None = None,
) -> None:
    """
    Execute the full RQ2 factual grounding analysis pipeline.

    Parameters
    ----------
    responses_xlsx:
        Path to the multi-sheet Excel workbook.  Must contain a
        'ground_truth' sheet and at least one model sheet.
    output_dir:
        Directory for all output files (created if absent).
    run_full:
        If ``False``, skip BERTScore and NLI inference; produces structural
        output only (useful for schema validation).
    domain_col:
        Optional column name identifying the domain within each model sheet.
        If ``None``, the sheet name is used as the domain label.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 72)
    log.info("RQ2 -- Factual Coverage Analysis")
    log.info("Primary metric    : Claim Recall (CR)")
    log.info("Primary output    : delta-CR on recall-winning queries")
    log.info("Primary challenger: SPARQL (FLQ)")
    log.info("Baseline          : NLQ")
    log.info("=" * 72)

    xl = pd.ExcelFile(responses_xlsx, engine="openpyxl")
    log.info("Loading ground truth ...")
    gt_lookup = load_ground_truth(xl)
    log.info("Ground truth entries loaded: %d", len(gt_lookup))

    # Evaluate each model sheet.
    model_sheets = [s for s in xl.sheet_names if s.lower() != GT_SHEET.lower()]
    if not model_sheets:
        raise ValueError(
            "No model sheets found.  The workbook must contain at least one "
            f"sheet other than '{GT_SHEET}'."
        )

    all_frames: list = []

    for sheet_name in model_sheets:
        log.info("Processing sheet: '%s'", sheet_name)
        try:
            raw_df     = xl.parse(sheet_name, dtype=str)
            df         = _normalise_columns(raw_df)
            domain     = _resolve_domain(sheet_name, df, domain_col)
            model_name = sheet_name

            for known_domain in ["TelcoQA", "FiQA", "LegalQA"]:
                if known_domain.lower() in sheet_name.lower():
                    model_name = (
                        sheet_name.lower()
                        .replace(known_domain.lower(), "")
                        .strip("_- ")
                        .title()
                    )
                    domain = known_domain
                    break

            sheet_df = evaluate_sheet(
                df=df,
                gt_lookup=gt_lookup,
                model_name=model_name,
                domain=domain,
                run_full=run_full,
            )
            all_frames.append(sheet_df)
            log.info(
                "  -> %d per-query rows for model='%s' domain='%s'",
                len(sheet_df), model_name, domain,
            )

        except ValueError as ve:
            log.error("Sheet '%s' skipped: %s", sheet_name, ve)
            continue
        except Exception:
            log.exception("Sheet '%s': unexpected error -- skipped.", sheet_name)
            continue

    if not all_frames:
        raise RuntimeError(
            "No data produced.  Check that column names match the expected "
            f"schema: {ALL_CANDIDATE_COLS}"
        )

    all_df  = pd.concat(all_frames, ignore_index=True)
    models  = sorted(all_df["model"].unique())
    domains = sorted(all_df["domain"].unique())
    log.info("Models : %s", models)
    log.info("Domains: %s", domains)

    # Core RQ2 computation.
    log.info("Computing delta-CR on recall-winning queries ...")
    delta_df       = compute_delta_cr_recall_wins(all_df)
    summary        = compute_challenger_summary(delta_df)
    sparql_summary = compute_sparql_primary_summary(delta_df)

    # Log primary findings.
    log.info("")
    log.info("-- PRIMARY FINDINGS: SPARQL (FLQ) --")
    sparql_rows = delta_df[delta_df["method"] == PRIMARY_CHALLENGER_LABEL]
    for _, r in sparql_rows.sort_values(["domain", "model"]).iterrows():
        log.info(
            "Model=%-12s  Domain=%-10s  WinRate=%5.1f%%  "
            "delta-CR=%+.4f  %s",
            r["model"], r["domain"],
            r["recall_win_rate"]  if np.isfinite(r["recall_win_rate"])  else float("nan"),
            r["delta_cr_on_rw"]   if np.isfinite(r["delta_cr_on_rw"])   else float("nan"),
            r["cd_arrow"],
        )
    log.info("")
    log.info("-- CHALLENGER SUMMARY --")
    for _, r in summary.iterrows():
        log.info(
            "%-12s  WinRate=%5.1f%%  Mean-delta-CR=%+.4f  "
            "Co-dir=%4.1f%%  %s",
            r["method"],
            r["mean_recall_win_rate"] if np.isfinite(r["mean_recall_win_rate"]) else float("nan"),
            r["mean_delta_cr_on_rw"]  if np.isfinite(r["mean_delta_cr_on_rw"])  else float("nan"),
            r["pct_codir"]            if np.isfinite(r["pct_codir"])            else float("nan"),
            "(PRIMARY FLQ)" if r["is_primary_sparql"] else "",
        )
    log.info("")

    # Save CSVs.
    all_df.to_csv(         out_dir / "per_query_metrics.csv",            index=False)
    delta_df.to_csv(       out_dir / "recall_win_anchored_delta_cr.csv", index=False)
    summary.to_csv(        out_dir / "challenger_summary.csv",           index=False)
    sparql_summary.to_csv( out_dir / "sparql_primary_summary.csv",       index=False)
    log.info("CSVs written to: %s", out_dir)

    # LaTeX recall win-rate table.
    log.info("Writing LaTeX recall win-rate table ...")
    write_recall_win_rate_table(
        delta_df=delta_df,
        out_path=str(out_dir / "tab_recall_win_rates.tex"),
    )

    # CR absolute boxplot (faceted by domain).
    log.info("Generating CR absolute boxplot ...")
    plot_cr_absolute_boxplot(delta_df=delta_df, out_dir=out_dir)

    log.info("=" * 72)
    log.info("RQ2 analysis complete.  Outputs in: %s", out_dir)
    log.info("=" * 72)


def main() -> None:
    """Parse command-line arguments and invoke :func:`run`."""
    parser = argparse.ArgumentParser(
        description=(
            "RQ2 Factual Coverage Analysis: Claim Recall (CR) and delta-CR "
            "on recall-winning queries.  SPARQL (FLQ) is the primary "
            "challenger; NLQ is the baseline."
        )
    )
    parser.add_argument(
        "--responses_xlsx",
        required=True,
        help=(
            "Path to the multi-sheet Excel workbook.  Must contain a "
            "'ground_truth' sheet and one sheet per model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./rq2_output/",
        help="Output directory for CSVs, LaTeX, and plots (default: ./rq2_output/).",
    )
    parser.add_argument(
        "--no_semantic",
        action="store_true",
        help=(
            "Skip BERTScore Recall and Claim Recall computation.  "
            "Produces structural output only -- useful for schema validation."
        ),
    )
    parser.add_argument(
        "--domain_col",
        default=None,
        help=(
            "Column name in model sheets identifying the domain.  "
            "If not specified, the sheet name is used as the domain label."
        ),
    )
    args = parser.parse_args()
    run(
        responses_xlsx=args.responses_xlsx,
        output_dir=args.output_dir,
        run_full=not args.no_semantic,
        domain_col=args.domain_col,
    )


if __name__ == "__main__":
    main()
