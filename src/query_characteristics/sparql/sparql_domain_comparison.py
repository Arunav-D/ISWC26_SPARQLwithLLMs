"""
sparql_domain_comparison.py
============================
NLQ-to-SPARQL cross-domain structural comparison pipeline.

Generates all outputs required for Appendix A.3 of the paper.

OUTPUT FILES (written to OUTPUT_DIR)
─────────────────────────────────────
sparql_comparison_report.xlsx
sparql_comparison_summary.txt
csv/A3_1_domain_feature_profiles.csv
csv/A3_2_paired_nlq_sparql_examples.csv
csv/A3_3_statistical_tests.csv
csv/A3_4_structural_similarity.csv
csv/scalar_summary.csv
csv/operator_profile.csv
csv/compositionality.csv
csv/boolean_features.csv
csv/nary_patterns.csv
csv/query_form_distribution.csv
csv/subgroup_analysis.csv

DEPENDENCIES
------------
pip install pandas openpyxl scipy scikit-learn numpy xlsxwriter
"""

from __future__ import annotations

import re, os, warnings, math
from collections import Counter
from typing import Dict, List, Any
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — update paths before running
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_FILES: Dict[str, str] = {
    "Domain_A": "data/domain_a.xlsx",
    "Domain_B": "data/domain_b.xlsx",
    "Domain_C": "data/domain_c.xlsx",
}

SPARQL_COL = "SPARQL Query"
NLQ_COL    = "Natural Language Query"

OUTPUT_DIR = "results"
XLSX_OUT   = os.path.join(OUTPUT_DIR, "sparql_comparison_report.xlsx")
TXT_OUT    = os.path.join(OUTPUT_DIR, "sparql_comparison_summary.txt")
CSV_DIR    = os.path.join(OUTPUT_DIR, "csv")

# ─────────────────────────────────────────────────────────────────────────────
# 0. UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    os.makedirs(CSV_DIR, exist_ok=True)


def write_csv(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(CSV_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [CSV] {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, target: str) -> str | None:
    for c in df.columns:
        if c.strip().lower() == target.lower():
            return c
    for c in df.columns:
        if target.lower().split()[0] in c.lower():
            return c
    return None


def load_domain_data(files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for domain, path in files.items():
        if not os.path.exists(path):
            print(f"  [WARN] {domain}: '{path}' not found — skipping.")
            continue
        df = pd.read_excel(path)
        sparql_col = _find_col(df, SPARQL_COL)
        if sparql_col is None:
            print(f"  [WARN] {domain}: no SPARQL column found. Columns: {list(df.columns)}")
            continue
        nlq_col = _find_col(df, NLQ_COL)
        out = pd.DataFrame()
        out["sparql"] = df[sparql_col].dropna().astype(str)
        out["sparql"] = out["sparql"][out["sparql"].str.strip().str.len() > 10]
        out["nlq"]    = df[nlq_col].astype(str) if nlq_col else ""
        out = out.reset_index(drop=True)
        data[domain] = out
        print(f"  Loaded {domain:10s}: {len(out):4d} queries  "
              f"(SPARQL='{sparql_col}'"
              + (f", NLQ='{nlq_col}'" if nlq_col else ", NLQ column absent") + ")")
    return data

# ─────────────────────────────────────────────────────────────────────────────
# 2. SPARQL FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_query_form(sparql: str) -> str:
    body = re.sub(r'#.*', '', sparql).upper().strip()
    if re.search(r'\bCONSTRUCT\b', body): return "CONSTRUCT"
    if re.search(r'\bDESCRIBE\b',  body): return "DESCRIBE"
    if re.search(r'\bASK\b',       body): return "ASK"
    if re.search(r'\bSELECT\b',    body): return "SELECT"
    return "UNKNOWN"


def count_operators(sparql: str) -> Dict[str, int]:
    body = re.sub(r'#.*', '', sparql)
    up   = body.upper()
    return {
        "FILTER"           : len(re.findall(r'\bFILTER\b(?!\s+NOT)', up)),
        "FILTER_NOT_EXISTS": len(re.findall(r'\bFILTER\s+NOT\s+EXISTS\b', up)),
        "OPTIONAL"         : len(re.findall(r'\bOPTIONAL\b', up)),
        "UNION"            : len(re.findall(r'\bUNION\b', up)),
        "GROUP_BY"         : len(re.findall(r'\bGROUP\s+BY\b', up)),
        "ORDER_BY"         : len(re.findall(r'\bORDER\s+BY\b', up)),
        "LIMIT"            : len(re.findall(r'\bLIMIT\b', up)),
        "DISTINCT"         : len(re.findall(r'\bDISTINCT\b', up)),
        "HAVING"           : len(re.findall(r'\bHAVING\b', up)),
        "BIND"             : len(re.findall(r'\bBIND\b', up)),
        "VALUES"           : len(re.findall(r'\bVALUES\b', up)),
        "SUBQUERY"         : len(re.findall(r'\{[^{}]*SELECT[^{}]*\}', up)),
        "REGEX"            : len(re.findall(r'\bREGEX\b', up)),
        "STRSTARTS"        : len(re.findall(r'\bSTRSTARTS\b', up)),
        "COUNT"            : len(re.findall(r'\bCOUNT\s*\(', up)),
        "AGG_OTHER"        : len(re.findall(r'\b(SUM|AVG|MIN|MAX)\s*\(', up)),
    }


def complexity_score(ops: Dict[str, int]) -> float:
    weights = {
        "SUBQUERY": 4, "UNION": 3, "FILTER_NOT_EXISTS": 3,
        "GROUP_BY": 2, "HAVING": 2, "FILTER": 2, "BIND": 2,
        "OPTIONAL": 1.5, "DISTINCT": 1, "ORDER_BY": 1,
        "REGEX": 1, "STRSTARTS": 0.5, "COUNT": 1.5,
        "AGG_OTHER": 1.5, "VALUES": 1, "LIMIT": 0.5,
    }
    return sum(weights.get(k, 1) * v for k, v in ops.items())


def count_nary_patterns(sparql: str) -> int:
    body = re.sub(r'#[^\n]*', '', sparql)
    score = 0
    score += len(re.findall(r'_:[a-zA-Z0-9]+\s+\w+:(?:hasParticipant|involvesRole)', body))
    score += len(re.findall(
        r'#\s*n-ary\s+(PRECONDITION_CHAIN|PARTICIPANT_SET|QUANTIFIED_RELATION'
        r'|TEMPORAL_CONTEXT|SPATIAL_CONTEXT|CONDITIONAL_ROLE)', sparql))
    if len(re.findall(r'FILTER\s+EXISTS\s*\{', body.upper())) >= 2:
        score += 1
    return score


def compositional_depth(sparql: str, ops: Dict[str, int]) -> Dict[str, Any]:
    body         = re.sub(r'#.*', '', sparql)
    op_count     = sum(ops.values())
    op_diversity = sum(1 for v in ops.values() if v > 0)
    vars_in      = re.findall(r'\?\w+', body)
    var_freq     = Counter(vars_in)
    join_count   = sum(1 for v, cnt in var_freq.items() if cnt >= 2)
    depth = max_depth = 0
    for ch in body:
        if ch == '{':  depth += 1; max_depth = max(max_depth, depth)
        elif ch == '}': depth -= 1
    comp_score = op_diversity * 2 + join_count * 1.5 + max_depth * 1.0
    return {
        "operator_count"     : op_count,
        "operator_diversity" : op_diversity,
        "join_count"         : join_count,
        "nesting_depth"      : max_depth,
        "compositional_score": round(comp_score, 2),
    }


def count_entities_and_relations(sparql: str) -> Dict[str, int]:
    body       = re.sub(r'#.*', '', sparql)
    variables  = set(re.findall(r'\?\w+', body))
    skip_vars  = {"?result", "?answer", "?label", "?p", "?o", "?s"}
    entity_vars = variables - skip_vars
    class_count = len(re.findall(r'rdf:type\b', body))
    rel_count   = len(re.findall(
        r'(?:broadband:|landline:|legal:|case:|fi:|property:)\w+\s+\?', body))
    literals    = re.findall(r'"[^"]*"', body)
    namespaces  = set(re.findall(r'\bPREFIX\s+(\w+):', body, re.IGNORECASE))
    return {
        "entity_count"   : len(entity_vars),
        "class_count"    : class_count,
        "relation_count" : max(0, rel_count),
        "literal_count"  : len(literals),
        "namespace_count": len(namespaces),
    }


def lexical_diversity(sparql: str) -> Dict[str, float]:
    body = re.sub(r'#.*', '', sparql)
    body = re.sub(r'PREFIX\s+\w+:\s*<[^>]+>', '', body)
    body = re.sub(r'<[^>]+>', '', body)
    boilerplate = {
        "select","distinct","where","filter","optional","union","ask","describe",
        "construct","order","by","group","having","limit","offset","bind","values",
        "not","exists","rdf","type","rdfs","label","owl","xsd","string","true",
        "false","and","or","graph","service","minus","prefix","from","named","as",
    }
    tokens = [t for t in re.findall(r'[a-zA-Z_]\w*', body.lower())
              if t not in boilerplate and len(t) > 2]
    if not tokens:
        return {"ttr": 0.0, "vocab_tokens": 0, "type_count": 0}
    types = set(tokens)
    return {"ttr": round(len(types) / len(tokens), 4),
            "vocab_tokens": len(tokens), "type_count": len(types)}


def size_metrics(sparql: str) -> Dict[str, int]:
    lines  = sparql.strip().split("\n")
    body   = re.sub(r'#.*', '', sparql)
    tokens = re.findall(r'\S+', body)
    return {"line_count": len(lines), "token_count": len(tokens), "char_count": len(sparql)}


def has_negation(sparql: str) -> bool:
    return bool(re.search(r'\bFILTER\s+NOT\s+EXISTS\b|\bMINUS\b', sparql.upper()))


def has_nlq_annotation(sparql: str) -> bool:
    return bool(re.search(r'_:queryMeta\s+rdf:type', sparql) or
                re.search(r'rdfs:comment.*@en', sparql))


def featurise(sparql: str) -> Dict[str, Any]:
    ops  = count_operators(sparql)
    comp = compositional_depth(sparql, ops)
    er   = count_entities_and_relations(sparql)
    ld   = lexical_diversity(sparql)
    sz   = size_metrics(sparql)
    return {
        "query_form"      : detect_query_form(sparql),
        "complexity_score": complexity_score(ops),
        "nary_count"      : count_nary_patterns(sparql),
        "has_negation"    : int(has_negation(sparql)),
        "has_annotation"  : int(has_nlq_annotation(sparql)),
        **{f"op_{k}": v for k, v in ops.items()},
        **comp,
        **er,
        "ttr"             : ld["ttr"],
        "vocab_tokens"    : ld["vocab_tokens"],
        **sz,
    }


def featurise_domain(df: pd.DataFrame) -> pd.DataFrame:
    rows = [featurise(s) for s in df["sparql"]]
    feat = pd.DataFrame(rows)
    feat["sparql"] = df["sparql"].values
    feat["nlq"]    = df["nlq"].values
    return feat

# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2: return float("nan")
    pooled = math.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) /
                       (len(a)+len(b)-2))
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled


def kruskal_dunn(groups: Dict[str, np.ndarray], metric: str) -> Dict[str, Any]:
    arrays = [v for v in groups.values() if len(v) >= 3]
    names  = [k for k, v in groups.items() if len(v) >= 3]

    def _skip(reason: str) -> Dict[str, Any]:
        return {"metric": metric, "kw_H": None, "kw_p": None, "skipped": reason,
                "pairwise_p": {}, "pairwise_d": {},
                "means"  : {n: round(float(groups[n].mean()), 3) for n in names},
                "medians": {n: round(float(np.median(groups[n])), 3) for n in names},
                "stds"   : {n: round(float(groups[n].std(ddof=1)), 3)
                             if len(groups[n]) >= 2 else float("nan") for n in names}}

    if len(arrays) < 2:
        return _skip("fewer than 2 groups with n>=3")
    try:
        H, p = stats.kruskal(*arrays)
    except ValueError as exc:
        print(f"  [WARN] Kruskal skipped '{metric}': {exc}")
        return _skip(str(exc))
    except Exception as exc:
        print(f"  [WARN] Kruskal error '{metric}': {exc}")
        return _skip(str(exc))

    all_data  = np.concatenate(arrays)
    all_ranks = stats.rankdata(all_data)
    n_total   = len(all_data)
    gsizes    = [len(a) for a in arrays]
    bounds    = np.cumsum([0] + gsizes)
    granks    = [all_ranks[bounds[i]:bounds[i+1]] for i in range(len(arrays))]
    m         = len(arrays)
    n_comp    = m * (m-1) / 2
    pairwise: Dict[str, float] = {}
    cohen_ds: Dict[str, float] = {}

    for i, j in combinations(range(m), 2):
        se  = math.sqrt((n_total*(n_total+1)/12.0) * (1.0/gsizes[i] + 1.0/gsizes[j]))
        z   = abs(granks[i].mean() - granks[j].mean()) / se if se != 0 else 0.0
        adj = min(2*(1-stats.norm.cdf(z)) * n_comp, 1.0)
        key = f"{names[i]} vs {names[j]}"
        pairwise[key] = round(adj, 5)
        cohen_ds[key] = round(cohen_d(arrays[i], arrays[j]), 3)

    return {"metric": metric, "kw_H": round(H, 3), "kw_p": round(p, 5),
            "skipped": None, "pairwise_p": pairwise, "pairwise_d": cohen_ds,
            "means"  : {names[i]: round(arrays[i].mean(), 3) for i in range(len(names))},
            "medians": {names[i]: round(float(np.median(arrays[i])), 3) for i in range(len(names))},
            "stds"   : {names[i]: round(arrays[i].std(ddof=1), 3) for i in range(len(names))}}

# ─────────────────────────────────────────────────────────────────────────────
# 4. SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def jaccard_tokens(a: str, b: str) -> float:
    ta = set(re.findall(r'[a-zA-Z_]\w*', a.lower()))
    tb = set(re.findall(r'[a-zA-Z_]\w*', b.lower()))
    if not ta and not tb: return 1.0
    if not ta or  not tb: return 0.0
    return len(ta & tb) / len(ta | tb)


def intra_domain_similarity(series: pd.Series, sample_n: int = 200) -> float:
    queries = series.tolist()
    if len(queries) > sample_n:
        np.random.seed(42)
        queries = list(np.random.choice(queries, sample_n, replace=False))
    pairs = list(combinations(range(len(queries)), 2))
    if len(pairs) > 5000:
        np.random.seed(42)
        pairs = [pairs[i] for i in np.random.choice(len(pairs), 5000, replace=False)]
    total = sum(jaccard_tokens(queries[i], queries[j]) for i, j in pairs)
    return total / len(pairs) if pairs else 0.0


def inter_domain_tfidf_similarity(domain_sparql: Dict[str, pd.Series],
                                   sample_n: int = 100) -> Dict[str, float]:
    sampled = {}
    for d, series in domain_sparql.items():
        q = series.tolist()
        if len(q) > sample_n:
            np.random.seed(42)
            q = [q[i] for i in np.random.choice(len(q), sample_n, replace=False)]
        sampled[d] = q
    domains  = list(sampled.keys())
    all_docs = [" ".join(sampled[d]) for d in domains]
    if len(all_docs) < 2: return {}
    try:
        M = TfidfVectorizer(token_pattern=r'[a-zA-Z_]\w*',
                            min_df=1, max_features=5000).fit_transform(all_docs)
    except Exception:
        return {}
    return {f"{domains[i]} vs {domains[j]}": round(float(cosine_similarity(M[i], M[j])[0,0]), 4)
            for i, j in combinations(range(len(domains)), 2)}

# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSIS TABLES
# ─────────────────────────────────────────────────────────────────────────────

def query_form_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    forms = ["SELECT","ASK","DESCRIBE","CONSTRUCT","UNKNOWN"]
    rows  = []
    for domain, df in feat_dfs.items():
        counts = df["query_form"].value_counts()
        total  = len(df)
        row    = {"Domain": domain, "N_queries": total}
        for f in forms:
            n = counts.get(f, 0)
            row[f"{f}_n"]   = n
            row[f"{f}_pct"] = round(100*n/total, 2) if total else 0
        rows.append(row)
    return pd.DataFrame(rows)


def operator_profile_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    op_cols = [c for c in next(iter(feat_dfs.values())).columns if c.startswith("op_")]
    rows = []
    for domain, df in feat_dfs.items():
        row = {"Domain": domain, "N": len(df)}
        for col in op_cols:
            name = col[3:]
            row[f"mean_{name}"]        = round(df[col].mean(), 3)
            row[f"pct_nonzero_{name}"] = round(100*(df[col]>0).mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def scalar_summary_table(feat_dfs: Dict[str, pd.DataFrame],
                          metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for domain, df in feat_dfs.items():
            if metric not in df.columns: continue
            try:
                arr = df[metric].dropna().values.astype(float)
                rows.append({"Metric": metric, "Domain": domain, "N": len(arr),
                             "Mean"  : round(arr.mean(), 3),
                             "Median": round(float(np.median(arr)), 3),
                             "Std"   : round(arr.std(ddof=1), 3),
                             "Min"   : round(arr.min(), 3),
                             "Max"   : round(arr.max(), 3),
                             "P25"   : round(float(np.percentile(arr, 25)), 3),
                             "P75"   : round(float(np.percentile(arr, 75)), 3)})
            except Exception as exc:
                print(f"  [WARN] scalar_summary: {domain}/{metric}: {exc}")
    return pd.DataFrame(rows)


def statistical_tests_table(feat_dfs: Dict[str, pd.DataFrame],
                              metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        try:
            groups = {d: df[metric].dropna().values.astype(float)
                      for d, df in feat_dfs.items() if metric in df.columns}
            if len(groups) < 2: continue
            result = kruskal_dunn(groups, metric)
            kw_p   = result["kw_p"]
            sig    = ("Yes" if (kw_p is not None and kw_p < 0.05)
                      else ("Skipped" if result.get("skipped") else "No"))
            row: Dict[str, Any] = {"Metric": metric, "KW_H": result["kw_H"],
                                   "KW_p": kw_p, "Significant": sig,
                                   "Skip_reason": result.get("skipped") or ""}
            for dom, mn in result["means"].items():
                row[f"Mean_{dom}"] = mn
            for dom, md in result["medians"].items():
                row[f"Median_{dom}"] = md
            for dom, sd in result["stds"].items():
                row[f"Std_{dom}"] = sd
            for pair, pv in result["pairwise_p"].items():
                row[f"p_{pair}"] = pv
            for pair, dv in result["pairwise_d"].items():
                row[f"d_{pair}"] = dv
            rows.append(row)
        except Exception as exc:
            print(f"  [WARN] stats_table '{metric}': {exc}")
            rows.append({"Metric": metric, "KW_H": None, "KW_p": None,
                         "Significant": "Error", "Skip_reason": str(exc)})
    return pd.DataFrame(rows)


def nary_summary(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for domain, df in feat_dfs.items():
        arr = df["nary_count"].values
        rows.append({"Domain": domain, "N": len(arr),
                     "Queries_with_Nary": int((arr>0).sum()),
                     "Pct_Nary"         : round(100*(arr>0).mean(), 1),
                     "Mean_Nary"        : round(arr.mean(), 3),
                     "Max_Nary"         : int(arr.max())})
    return pd.DataFrame(rows)


def boolean_feature_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for domain, df in feat_dfs.items():
        row = {"Domain": domain, "N": len(df)}
        for feat in ["has_negation","has_annotation"]:
            if feat in df.columns:
                row[f"Pct_{feat}"] = round(100*df[feat].mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def compositionality_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = ["operator_diversity","join_count","nesting_depth","compositional_score"]
    rows = []
    for domain, df in feat_dfs.items():
        row = {"Domain": domain, "N": len(df)}
        for col in cols:
            if col in df.columns:
                arr = df[col].values.astype(float)
                row[f"Mean_{col}"]   = round(arr.mean(), 3)
                row[f"Median_{col}"] = round(float(np.median(arr)), 3)
                row[f"Std_{col}"]    = round(arr.std(ddof=1), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_analysis(feat_dfs: Dict[str, pd.DataFrame],
                       metrics: List[str]) -> pd.DataFrame:
    rows = []
    for domain, df in feat_dfs.items():
        for form, grp in df.groupby("query_form"):
            if len(grp) < 3: continue
            row = {"Domain": domain, "Query_Form": form, "N": len(grp)}
            for m in metrics:
                if m in grp.columns:
                    arr = grp[m].values.astype(float)
                    row[f"{m}_mean"]   = round(arr.mean(), 3)
                    row[f"{m}_median"] = round(float(np.median(arr)), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def similarity_table(domain_sparql: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for domain, series in domain_sparql.items():
        sim = intra_domain_similarity(series)
        rows.append({"Type": "Intra-domain", "Pair": f"{domain} internal",
                     "Jaccard_mean": round(sim, 4), "TF-IDF_cosine": None})
    inter = inter_domain_tfidf_similarity(domain_sparql)
    for pair, sim in inter.items():
        rows.append({"Type": "Inter-domain", "Pair": pair,
                     "Jaccard_mean": None, "TF-IDF_cosine": round(sim, 4)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 6. TABLE A3.2 — REAL PAIRED NLQ↔SPARQL EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY_LEVELS = [
    ("Simple",       0,    1),
    ("1-hop",        1,    3),
    ("Multi-hop",    3,    7),
    ("Aggregation",  None, None),
    ("Negation",     None, None),
]


def _level_mask(df: pd.DataFrame, level: str, lo: float, hi: float) -> pd.Series:
    if level == "Aggregation":
        return (df.get("op_GROUP_BY", pd.Series(0, index=df.index)) > 0) | \
               (df.get("op_COUNT",    pd.Series(0, index=df.index)) > 0)
    if level == "Negation":
        return df["has_negation"] == 1
    return (df["complexity_score"] >= lo) & (df["complexity_score"] < hi)


def _infer_nlq(row: pd.Series) -> str:
    if str(row.get("nlq","")).strip() not in ("","nan","None"):
        return str(row["nlq"]).strip()
    form  = row.get("query_form","SELECT")
    joins = int(row.get("join_count", 0))
    ops   = int(row.get("operator_diversity", 0))
    neg   = bool(row.get("has_negation", 0))
    agg   = (row.get("op_GROUP_BY",0) or row.get("op_COUNT",0))
    parts = [f"[Auto-inferred] {form} query"]
    if joins:   parts.append(f"with {joins} join(s)")
    if ops > 1: parts.append(f"and {ops} operator types")
    if neg:     parts.append("including negation (FILTER NOT EXISTS / MINUS)")
    if agg:     parts.append("with aggregation (GROUP BY / COUNT)")
    return " ".join(parts)


def paired_nlq_sparql_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for domain, df in feat_dfs.items():
        for (level, lo, hi) in COMPLEXITY_LEVELS:
            mask = _level_mask(df, level, lo, hi)
            sub  = df[mask].copy()
            if sub.empty:
                continue
            median_cs = sub["complexity_score"].median()
            idx       = (sub["complexity_score"] - median_cs).abs().idxmin()
            rep       = sub.loc[idx]
            rows.append({
                "Domain"            : domain,
                "Complexity_Level"  : level,
                "NLQ"               : _infer_nlq(rep),
                "SPARQL"            : rep["sparql"],
                "complexity_score"  : rep["complexity_score"],
                "operator_diversity": rep["operator_diversity"],
                "join_count"        : rep["join_count"],
                "nesting_depth"     : rep["nesting_depth"],
                "query_form"        : rep["query_form"],
                "has_negation"      : int(rep["has_negation"]),
                "has_annotation"    : int(rep["has_annotation"]),
                "line_count"        : rep["line_count"],
                "vocab_tokens"      : rep["vocab_tokens"],
                "ttr"               : rep["ttr"],
                "Selection_note"    : "Closest to median complexity_score within stratum",
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 7. AGGREGATE PROFILE TABLE (Table A3.1)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_profile_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for domain, df in feat_dfs.items():
        n  = len(df)
        qf = df["query_form"].value_counts()
        row = {
            "Domain"                  : domain,
            "N_queries"               : n,
            "SELECT_pct"              : round(100*qf.get("SELECT",0)/n, 1),
            "ASK_pct"                 : round(100*qf.get("ASK",0)/n, 1),
            "DESCRIBE_pct"            : round(100*qf.get("DESCRIBE",0)/n, 1),
            "CONSTRUCT_pct"           : round(100*qf.get("CONSTRUCT",0)/n, 1),
            "mean_complexity_score"   : round(df["complexity_score"].mean(), 2),
            "median_complexity_score" : round(df["complexity_score"].median(), 2),
            "std_complexity_score"    : round(df["complexity_score"].std(ddof=1), 2),
            "pct_FILTER"              : round(100*(df["op_FILTER"]>0).mean(), 1),
            "pct_OPTIONAL"            : round(100*(df["op_OPTIONAL"]>0).mean(), 1),
            "pct_UNION"               : round(100*(df["op_UNION"]>0).mean(), 1),
            "pct_FILTER_NOT_EXISTS"   : round(100*(df["op_FILTER_NOT_EXISTS"]>0).mean(), 1),
            "pct_GROUP_BY_or_COUNT"   : round(100*((df["op_GROUP_BY"]>0)|(df["op_COUNT"]>0)).mean(), 1),
            "pct_SUBQUERY"            : round(100*(df["op_SUBQUERY"]>0).mean(), 1),
            "mean_operator_diversity" : round(df["operator_diversity"].mean(), 2),
            "mean_join_count"         : round(df["join_count"].mean(), 2),
            "mean_nesting_depth"      : round(df["nesting_depth"].mean(), 2),
            "mean_compositional_score": round(df["compositional_score"].mean(), 2),
            "mean_entity_count"       : round(df["entity_count"].mean(), 2),
            "mean_class_count"        : round(df["class_count"].mean(), 2),
            "mean_literal_count"      : round(df["literal_count"].mean(), 2),
            "mean_ttr"                : round(df["ttr"].mean(), 4),
            "mean_vocab_tokens"       : round(df["vocab_tokens"].mean(), 1),
            "pct_nary"                : round(100*(df["nary_count"]>0).mean(), 1),
            "pct_negation"            : round(100*df["has_negation"].mean(), 1),
            "mean_line_count"         : round(df["line_count"].mean(), 1),
            "mean_token_count"        : round(df["token_count"].mean(), 1),
        }
        rows.append(row)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 8. NARRATIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def generate_narrative(domain_sparql, feat_dfs, qf_tbl, stats_tbl,
                        nary_tbl, sim_tbl, comp_tbl, bool_tbl,
                        profile_tbl) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("APPENDIX A.3 — NLQ-TO-SPARQL CROSS-DOMAIN FEATURE COMPARISON")
    lines.append("=" * 80)

    lines.append("\n§1  DATASET SIZES")
    lines.append("-" * 40)
    for _, row in profile_tbl.iterrows():
        lines.append(f"  {row['Domain']:10s}: {int(row['N_queries']):4d} queries")

    lines.append("\n§2  QUERY FORM DISTRIBUTION")
    lines.append("-" * 40)
    for _, row in profile_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}  SELECT={row['SELECT_pct']:5.1f}%  "
            f"ASK={row['ASK_pct']:5.1f}%  "
            f"DESCRIBE={row['DESCRIBE_pct']:5.1f}%  "
            f"CONSTRUCT={row['CONSTRUCT_pct']:5.1f}%")

    lines.append("\n§3  COMPLEXITY")
    lines.append("-" * 40)
    for _, row in profile_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: mean={row['mean_complexity_score']:.2f}  "
            f"median={row['median_complexity_score']:.2f}  "
            f"std={row['std_complexity_score']:.2f}")
    sig = stats_tbl[stats_tbl["Metric"] == "complexity_score"]
    if not sig.empty:
        r = sig.iloc[0]
        lines.append(f"  Kruskal-Wallis H={r.get('KW_H')}  p={r.get('KW_p')}  "
                     f"Significant={r.get('Significant')}")

    lines.append("\n§4  KEY OPERATOR USAGE (% of queries per domain)")
    lines.append("-" * 40)
    op_cols = ["pct_FILTER","pct_OPTIONAL","pct_UNION",
               "pct_FILTER_NOT_EXISTS","pct_GROUP_BY_or_COUNT","pct_SUBQUERY"]
    header  = f"  {'Domain':10s}  " + "  ".join(f"{c[4:]:>22s}" for c in op_cols)
    lines.append(header)
    for _, row in profile_tbl.iterrows():
        vals = "  ".join(f"{row[c]:>22.1f}" for c in op_cols)
        lines.append(f"  {row['Domain']:10s}  {vals}")

    lines.append("\n§5  COMPOSITIONALITY")
    lines.append("-" * 40)
    for _, row in profile_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: op_diversity={row['mean_operator_diversity']:.2f}  "
            f"joins={row['mean_join_count']:.2f}  "
            f"nesting={row['mean_nesting_depth']:.2f}  "
            f"comp_score={row['mean_compositional_score']:.2f}")

    lines.append("\n§6  SEMANTIC DENSITY")
    lines.append("-" * 40)
    for _, row in profile_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: entities={row['mean_entity_count']:.2f}  "
            f"classes={row['mean_class_count']:.2f}  "
            f"literals={row['mean_literal_count']:.2f}")

    lines.append("\n§7  LEXICAL RICHNESS")
    lines.append("-" * 40)
    for _, row in profile_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: TTR={row['mean_ttr']:.4f}  "
            f"vocab_tokens={row['mean_vocab_tokens']:.1f}")

    lines.append("\n§8  N-ARY PATTERNS & NEGATION")
    lines.append("-" * 40)
    for _, row in profile_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: n-ary={row['pct_nary']:.1f}%  "
            f"negation={row['pct_negation']:.1f}%")

    lines.append("\n§9  STRUCTURAL SIMILARITY (Jaccard / TF-IDF cosine)")
    lines.append("-" * 40)
    for _, row in sim_tbl.iterrows():
        if row["Type"] == "Intra-domain":
            lines.append(f"  {row['Pair']:30s}  Jaccard={row['Jaccard_mean']}")
        else:
            lines.append(f"  {row['Pair']:30s}  TF-IDF_cosine={row['TF-IDF_cosine']}")

    lines.append("\n§10 STATISTICAL SIGNIFICANCE (Bonferroni-corrected Dunn post-hoc)")
    lines.append("-" * 40)
    for _, row in stats_tbl.iterrows():
        sig = row.get("Significant")
        if sig == "Yes":
            lines.append(f"  '{row['Metric']}' significantly differs  "
                         f"(H={row['KW_H']}, p={row['KW_p']})")
            for col in row.index:
                if col.startswith("d_") and pd.notna(row[col]):
                    d   = row[col]
                    mag = "small" if abs(d)<0.5 else ("medium" if abs(d)<0.8 else "large")
                    lines.append(f"    {col[2:]}: d={d:.3f} ({mag})")
        elif sig in ("Skipped","Error"):
            lines.append(f"  '{row['Metric']}' — {row.get('Skip_reason','')}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# 9. SCALAR METRICS LIST
# ─────────────────────────────────────────────────────────────────────────────

SCALAR_METRICS = [
    "complexity_score","nary_count","compositional_score",
    "operator_count","operator_diversity","join_count","nesting_depth",
    "entity_count","class_count","relation_count","literal_count","namespace_count",
    "ttr","vocab_tokens","line_count","token_count","char_count",
    "has_negation","has_annotation",
    "op_FILTER","op_OPTIONAL","op_UNION","op_FILTER_NOT_EXISTS",
    "op_ORDER_BY","op_DISTINCT","op_GROUP_BY","op_COUNT","op_LIMIT",
]

SUBGROUP_METRICS = [
    "complexity_score","compositional_score","join_count",
    "entity_count","literal_count"
]

# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison() -> None:
    ensure_dirs()
    print("\nNLQ-to-SPARQL Cross-Domain Comparison Pipeline")
    print("=" * 60)

    print("\n[1] Loading data...")
    domain_data = load_domain_data(DOMAIN_FILES)
    if not domain_data:
        print("\n[ERROR] No data loaded. Check DOMAIN_FILES paths.")
        return

    print("\n[2] Featurising queries...")
    feat_dfs: Dict[str, pd.DataFrame] = {}
    for domain, df in domain_data.items():
        print(f"  {domain}...")
        feat_dfs[domain] = featurise_domain(df)

    domain_sparql = {d: feat_dfs[d]["sparql"] for d in feat_dfs}

    print("\n[3] Building analysis tables...")
    qf_tbl      = query_form_table(feat_dfs)
    op_tbl      = operator_profile_table(feat_dfs)
    scalar_tbl  = scalar_summary_table(feat_dfs, SCALAR_METRICS)
    nary_tbl    = nary_summary(feat_dfs)
    comp_tbl    = compositionality_table(feat_dfs)
    bool_tbl    = boolean_feature_table(feat_dfs)
    sub_tbl     = subgroup_analysis(feat_dfs, SUBGROUP_METRICS)
    profile_tbl = aggregate_profile_table(feat_dfs)

    print("\n[4] Running statistical tests (Kruskal-Wallis + Dunn)...")
    stats_tbl = statistical_tests_table(feat_dfs, SCALAR_METRICS)

    print("\n[5] Computing structural similarity...")
    sim_tbl = similarity_table(domain_sparql)

    print("\n[6] Extracting real paired NLQ-SPARQL examples (Table A3.2)...")
    paired_tbl = paired_nlq_sparql_table(feat_dfs)

    combined = pd.concat([df.drop(columns=["sparql","nlq"], errors="ignore")
                           .assign(Domain=domain)
                          for domain, df in feat_dfs.items()], ignore_index=True)

    print("\n[7] Writing CSV outputs...")
    write_csv(profile_tbl,  "A3_1_domain_feature_profiles.csv")
    write_csv(paired_tbl,   "A3_2_paired_nlq_sparql_examples.csv")
    write_csv(stats_tbl,    "A3_3_statistical_tests.csv")
    write_csv(sim_tbl,      "A3_4_structural_similarity.csv")
    write_csv(scalar_tbl,   "scalar_summary.csv")
    write_csv(op_tbl,       "operator_profile.csv")
    write_csv(comp_tbl,     "compositionality.csv")
    write_csv(bool_tbl,     "boolean_features.csv")
    write_csv(nary_tbl,     "nary_patterns.csv")
    write_csv(qf_tbl,       "query_form_distribution.csv")
    write_csv(sub_tbl,      "subgroup_analysis.csv")
    write_csv(combined,     "all_feature_vectors.csv")

    print(f"\n[8] Writing Excel workbook → {XLSX_OUT}")
    with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as writer:
        profile_tbl.to_excel(writer,  sheet_name="A3.1_DomainProfiles",    index=False)
        paired_tbl.to_excel(writer,   sheet_name="A3.2_NLQ_SPARQL_Pairs",  index=False)
        stats_tbl.to_excel(writer,    sheet_name="A3.3_StatisticalTests",   index=False)
        sim_tbl.to_excel(writer,      sheet_name="A3.4_Similarity",         index=False)
        qf_tbl.to_excel(writer,       sheet_name="QueryFormDistribution",   index=False)
        op_tbl.to_excel(writer,       sheet_name="OperatorProfile",         index=False)
        scalar_tbl.to_excel(writer,   sheet_name="ScalarMetrics",           index=False)
        nary_tbl.to_excel(writer,     sheet_name="NaryPatterns",            index=False)
        comp_tbl.to_excel(writer,     sheet_name="Compositionality",        index=False)
        bool_tbl.to_excel(writer,     sheet_name="BooleanFeatures",         index=False)
        sub_tbl.to_excel(writer,      sheet_name="SubgroupAnalysis",        index=False)
        combined.to_excel(writer,     sheet_name="AllFeatureVectors",       index=False)

        method_rows = [
            ["Dimension","Method","Implementation"],
            ["Query Form","Form taxonomy","Regex on SELECT/ASK/DESCRIBE/CONSTRUCT"],
            ["Complexity","Weighted operator sum","16 operators with expressive-power weights"],
            ["N-ary","Pattern detection","Blank-node sets, FILTER EXISTS pairs, pipeline comments"],
            ["Compositionality","Structural depth","op_diversity×2 + join_count×1.5 + nesting_depth"],
            ["Entities/Relations","Variable & triple count","?var count, rdf:type, predicate triples"],
            ["Lexical Diversity","Type-Token Ratio","TTR on non-boilerplate tokens"],
            ["Similarity","Jaccard + TF-IDF cosine","Token-set Jaccard (intra); TF-IDF centroid (inter)"],
            ["Statistics","Kruskal-Wallis + Dunn","Non-parametric; Bonferroni-corrected pairwise"],
            ["Effect Size","Cohen's d","Pooled-std; <0.5=small, 0.5-0.8=medium, >0.8=large"],
        ]
        pd.DataFrame(method_rows[1:], columns=method_rows[0]).to_excel(
            writer, sheet_name="Methodology", index=False)

    print(f"\n[9] Writing narrative summary → {TXT_OUT}")
    narrative = generate_narrative(
        domain_sparql, feat_dfs, qf_tbl, stats_tbl,
        nary_tbl, sim_tbl, comp_tbl, bool_tbl, profile_tbl)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(narrative)
    print("\n" + narrative)

    print(f"\n✓ Excel  : {XLSX_OUT}")
    print(f"✓ Summary: {TXT_OUT}")
    print(f"✓ CSVs   : {CSV_DIR}/")


if __name__ == "__main__":
    run_comparison()
