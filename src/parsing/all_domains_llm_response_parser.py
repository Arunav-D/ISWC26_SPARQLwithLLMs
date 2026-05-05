"""
model_output_parser.py
======================
Parses JSON output files produced by the QAS evaluation harness for each
model under test and writes per-model and cross-model Excel workbooks.

Directory layout expected under the base directory
-------------------------------------------------
    <model>_NLQ/         <model>_NLQ.json
    <model>_NLQ_COT/     <model>_NLQ_COT.json
    <model>_SPARQL/      <model>_SPARQL.json
    <model>_SPARQL_COT/  <model>_SPARQL_COT.json

Output (written to <base_dir>/analysis/ by default)
-----------------------------------------------------
    <model>_metrics.xlsx       per Query_ID: input_length, generated_length,
                               generation_time_seconds, confidence_score
                               for each inference mode
    <model>_responses.xlsx     per Query_ID: Question, SPARQL_Query,
                               and the generated response for each mode
    ALL_MODELS_metrics.xlsx    one sheet per model (cross-model comparison)
    ALL_MODELS_responses.xlsx  one sheet per model (cross-model comparison)

Usage
-----
    # default: base directory = directory containing this script
    python model_output_parser.py

    # explicit paths
    python model_output_parser.py --base-dir /path/to/data --out-dir /path/to/output

    # override model list
    python model_output_parser.py --models llama3 mistral qwen
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model list — override at runtime with --models
# ---------------------------------------------------------------------------

DEFAULT_MODELS: List[str] = [
    "llama3",
    "mistral",
    "qwen",
    "qwen-coder",
    "codellama",
    "gemma",
]

# Inference modes: internal key -> folder/file suffix
MODES: Dict[str, str] = {
    "nlq":        "NLQ",
    "nlq_cot":    "NLQ_COT",
    "sparql":     "SPARQL",
    "sparql_cot": "SPARQL_COT",
}

# Numeric fields extracted from each result record
METRIC_FIELDS: List[str] = [
    "input_length",
    "generated_length",
    "generation_time_seconds",
    "confidence_score",
]

# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON results file and return its contents, or None on failure."""
    if not path.exists():
        logger.warning("Not found: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded %d records  ←  %s", len(data.get("results", [])), path.name)
        return data
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------

# Matches common "ANSWER:" header variants produced by instruction-tuned models,
# including markdown decoration (**, ##, -, *) and case variation.
_ANSWER_TAG = re.compile(r"(?i)(?:[*#\->\s]*)answer\s*[*]*\s*:")


def clean_response(text: str) -> str:
    """
    Strip all "ANSWER:" metatag variants from a generated response string and
    return only the substantive answer text.

    The regex is applied in a loop so nested or repeated tags are all removed.
    Leading non-alphanumeric characters are then stripped from the result.
    If no tag is found the original text is returned unchanged.
    """
    current = text
    while True:
        match = _ANSWER_TAG.search(current)
        if not match:
            break
        current = current[match.end():]
    return re.sub(r"^[^\w(]+", "", current).strip()


# ---------------------------------------------------------------------------
# Result indexing
# ---------------------------------------------------------------------------

def index_results(data: Dict[str, Any], mode_key: str) -> Dict[int, Dict[str, Any]]:
    """
    Convert a flat list of result records into a dict keyed by Query_ID (int).

    Each entry stores the cleaned response, the original question and SPARQL
    query, plus all numeric metric fields prefixed with the mode key.
    """
    indexed: Dict[int, Dict[str, Any]] = {}
    for record in data.get("results", []):
        qid = record.get("Query_ID")
        if qid is None:
            continue
        qid = int(qid)
        entry: Dict[str, Any] = {
            f"{mode_key}_response": clean_response(record.get("Generated_Response", "")),
            "_question":            record.get("Question", ""),
            "_sparql_query":        record.get("SPARQL_Query", ""),
        }
        for field in METRIC_FIELDS:
            entry[f"{mode_key}_{field}"] = record.get(field)
        indexed[qid] = entry
    return indexed


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def build_dataframes(model: str, base_dir: Path) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load all four inference-mode JSON files for *model* and return two
    DataFrames: one for numeric metrics and one for text responses.

    Parameters
    ----------
    model    : model name, used to construct subdirectory and file names
    base_dir : root directory containing the <model>_<MODE>/ subdirectories

    Returns None if no Query_IDs are found across any mode.
    """
    mode_data: Dict[str, Dict[int, Dict]] = {}
    for mode_key, suffix in MODES.items():
        path = base_dir / f"{model}_{suffix}" / f"{model}_{suffix}.json"
        raw  = load_json(path)
        mode_data[mode_key] = index_results(raw, mode_key) if raw else {}

    all_qids = sorted(set().union(*(set(d.keys()) for d in mode_data.values())))
    if not all_qids:
        logger.error("No Query_IDs found for model '%s' — skipping.", model)
        return None

    active_modes = [k for k, v in mode_data.items() if v]
    logger.info("%d Query_IDs found | active modes: %s", len(all_qids), active_modes)

    metrics_rows:  List[Dict] = []
    response_rows: List[Dict] = []

    for qid in all_qids:
        # Pull shared text fields from whichever mode has them
        question = sparql_query = ""
        for mode_key in MODES:
            rec = mode_data[mode_key].get(qid, {})
            if not question     and rec.get("_question"):     question     = rec["_question"]
            if not sparql_query and rec.get("_sparql_query"): sparql_query = rec["_sparql_query"]

        # Metrics row: one numeric column per mode × metric field
        m_row: Dict[str, Any] = {"Query_ID": qid}
        for mode_key in MODES:
            rec = mode_data[mode_key].get(qid, {})
            for field in METRIC_FIELDS:
                m_row[f"{mode_key}_{field}"] = rec.get(f"{mode_key}_{field}")
        metrics_rows.append(m_row)

        # Responses row: cleaned generated text per mode
        r_row: Dict[str, Any] = {
            "Query_ID":    qid,
            "Question":    question,
            "SPARQL_Query": sparql_query,
        }
        for mode_key in MODES:
            rec = mode_data[mode_key].get(qid, {})
            r_row[f"{mode_key}_response"] = rec.get(f"{mode_key}_response", "")
        response_rows.append(r_row)

    # Enforce canonical column ordering
    metric_cols = ["Query_ID"] + [
        f"{mk}_{f}" for mk in MODES for f in METRIC_FIELDS
    ]
    response_cols = ["Query_ID", "Question", "SPARQL_Query"] + [
        f"{mk}_response" for mk in MODES
    ]

    return {
        "metrics":   pd.DataFrame(metrics_rows).reindex(columns=metric_cols),
        "responses": pd.DataFrame(response_rows).reindex(columns=response_cols),
    }


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------

def _autofit_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    """Set column widths to fit content (capped at 80 characters)."""
    try:
        ws = writer.sheets[sheet_name]
        for col_idx, col_name in enumerate(df.columns, start=1):
            col_letter = ws.cell(1, col_idx).column_letter
            max_len = max(
                len(str(col_name)),
                int(df[col_name].astype(str).str.len().max()) if len(df) else 0,
            )
            ws.column_dimensions[col_letter].width = min(max_len + 2, 80)
    except Exception:
        pass  # Non-fatal: cosmetic only


def write_workbook(df: pd.DataFrame, path: Path, sheet_name: str) -> None:
    """Write *df* to a single-sheet Excel workbook at *path*."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        safe_name = sheet_name[:31]
        df.to_excel(writer, index=False, sheet_name=safe_name)
        _autofit_columns(writer, safe_name, df)
    logger.info("Saved %d rows  →  %s", len(df), path.name)


def append_sheet(df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str) -> None:
    """Append *df* as a new sheet inside an already-open ExcelWriter."""
    safe_name = sheet_name[:31]
    df.to_excel(writer, index=False, sheet_name=safe_name)
    _autofit_columns(writer, safe_name, df)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_model_summary(model: str, metrics_df: pd.DataFrame) -> None:
    """Print a brief per-mode summary of average metrics to the console."""
    for mode_key in MODES:
        col_t = f"{mode_key}_generation_time_seconds"
        col_c = f"{mode_key}_confidence_score"
        col_g = f"{mode_key}_generated_length"
        if col_t in metrics_df.columns and metrics_df[col_t].notna().any():
            logger.info(
                "  %-14s  avg_time=%5.2fs  avg_conf=%.3f  avg_gen_len=%4.0f tok",
                mode_key.upper(),
                metrics_df[col_t].mean(),
                metrics_df[col_c].mean(),
                metrics_df[col_g].mean(),
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    # Default base directory is the folder that contains this script, so the
    # repository can be cloned anywhere without touching the source.
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Parse QAS model output JSON files and write Excel workbooks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=script_dir,
        help="Root directory containing <model>_<MODE>/ subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for Excel files. Defaults to <base-dir>/analysis/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help="Model names to process (must match subdirectory prefixes).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    base_dir: Path = args.base_dir.resolve()
    out_dir:  Path = (args.out_dir or base_dir / "analysis").resolve()
    models:   List[str] = args.models

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Base dir : %s", base_dir)
    logger.info("Out dir  : %s", out_dir)
    logger.info("Models   : %s", models)

    processed: Dict[str, Dict[str, pd.DataFrame]] = {}

    for model in models:
        logger.info("=" * 60)
        logger.info("Processing model: %s", model)
        logger.info("=" * 60)

        dfs = build_dataframes(model, base_dir)
        if dfs is None:
            continue

        write_workbook(dfs["metrics"],   out_dir / f"{model}_metrics.xlsx",   "metrics")
        write_workbook(dfs["responses"], out_dir / f"{model}_responses.xlsx", "responses")
        _log_model_summary(model, dfs["metrics"])

        processed[model] = dfs

    # Write combined cross-model workbooks (one sheet per model)
    if len(processed) > 1:
        metrics_path   = out_dir / "ALL_MODELS_metrics.xlsx"
        responses_path = out_dir / "ALL_MODELS_responses.xlsx"

        with (
            pd.ExcelWriter(metrics_path,   engine="openpyxl") as mw,
            pd.ExcelWriter(responses_path, engine="openpyxl") as rw,
        ):
            for model, dfs in processed.items():
                append_sheet(dfs["metrics"],   mw, model)
                append_sheet(dfs["responses"], rw, model)

        logger.info("Combined metrics    →  %s", metrics_path.name)
        logger.info("Combined responses  →  %s", responses_path.name)

    logger.info("Done. %d/%d models written to %s", len(processed), len(models), out_dir)


if __name__ == "__main__":
    main()
