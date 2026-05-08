"""
nlq_domain_analysis.py
=======================
Comparative analysis of Natural Language Questions (NLQs) across
knowledge-graph QA domains.

NLQ analysis characterises:
  (a) Linguistic surface properties  — length, vocabulary, readability
  (b) Syntactic complexity           — dependency depth, clause structure,
                                       POS distributions
  (c) Semantic / pragmatic type      — question intent taxonomy
  (d) Lexical diversity              — TTR, MATTR, MTLD
  (e) Information density            — content-word ratio, named-entity density
  (f) Cross-domain separability      — TF-IDF + cosine, Wasserstein distance
  (g) Corpus-level diversity         — Self-BLEU (lower = more diverse)
  (h) Semantic similarity            — sentence-transformer embeddings
  (i) Statistical tests              — permutation test on mean embeddings,
                                       Kruskal-Wallis, effect sizes (Cohen's d, η²)

OUTPUTS
-------
  csv/NLQ_B1_surface_metrics.csv
  csv/NLQ_B2_question_type_distribution.csv
  csv/NLQ_B3_lexical_diversity.csv
  csv/NLQ_B4_pos_profile.csv
  csv/NLQ_B5_statistical_tests.csv
  csv/NLQ_B6_self_bleu.csv
  csv/NLQ_B7_embedding_similarity.csv
  csv/NLQ_B8_named_entity_density.csv
  csv/NLQ_B9_dependency_complexity.csv
  csv/NLQ_B10_combined_feature_vectors.csv
  nlq_analysis_report.xlsx
  nlq_analysis_summary.txt

DEPENDENCIES
------------
  pip install pandas openpyxl scipy scikit-learn numpy spacy nltk
              sentence-transformers textstat

  python -m spacy download en_core_web_sm
  python -m spacy download en_core_web_trf   # optional: transformer-based

  NOTE: sentence-transformers downloads ~90 MB model on first run.
        Set USE_SENTENCE_TRANSFORMERS = False to skip if offline.

CONFIGURATION
-------------
  Edit DOMAIN_FILES, OUTPUT_DIR, and the feature flags below.
"""

from __future__ import annotations

import os
import re
import math
import warnings
import itertools
import random
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_FILES: Dict[str, str] = {
    "Domain_A": "data/domain_a.xlsx",
    "Domain_B": "data/domain_b.xlsx",
    "Domain_C": "data/domain_c.xlsx",
}

NLQ_COL    = "Question"
OUTPUT_DIR = "results"
XLSX_OUT   = os.path.join(OUTPUT_DIR, "nlq_analysis_report.xlsx")
TXT_OUT    = os.path.join(OUTPUT_DIR, "nlq_analysis_summary.txt")
CSV_DIR    = os.path.join(OUTPUT_DIR, "csv")

# ── Feature flags ─────────────────────────────────────────────────────────────
USE_SPACY                 = True
USE_SENTENCE_TRANSFORMERS = True
USE_TEXTSTAT              = True
SPACY_MODEL               = "en_core_web_sm"
SBERT_MODEL               = "all-MiniLM-L6-v2"
SELF_BLEU_SAMPLE_N        = 500
EMBEDDING_SAMPLE_N        = 300

# ─────────────────────────────────────────────────────────────────────────────
# 0. UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    os.makedirs(CSV_DIR, exist_ok=True)


def write_csv(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(CSV_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [CSV] {path}")


def _find_col(df: pd.DataFrame, target: str) -> Optional[str]:
    for c in df.columns:
        if c.strip().lower() == target.lower():
            return c
    for c in df.columns:
        if target.lower() in c.lower():
            return c
    return None


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt(
        ((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) /
        (len(a) + len(b) - 2))
    return 0.0 if pooled == 0 else (np.mean(a) - np.mean(b)) / pooled


def eta_squared(H: float, k: int, N: int) -> float:
    """Epsilon-squared (unbiased η²) from Kruskal-Wallis H."""
    if N <= k:
        return float("nan")
    return (H - k + 1) / (N - k)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_nlq_data(files: Dict[str, str]) -> Dict[str, pd.Series]:
    data: Dict[str, pd.Series] = {}
    for domain, path in files.items():
        if not os.path.exists(path):
            print(f"  [WARN] {domain}: '{path}' not found — skipping.")
            continue
        df  = pd.read_excel(path)
        col = _find_col(df, NLQ_COL)
        if col is None:
            print(f"  [WARN] {domain}: column '{NLQ_COL}' not found. "
                  f"Available: {list(df.columns)}")
            continue
        series = (df[col].dropna().astype(str)
                  .str.strip()
                  .pipe(lambda s: s[s.str.len() > 3])
                  .reset_index(drop=True))
        data[domain] = series
        print(f"  Loaded {domain:10s}: {len(series):4d} NLQs  (col='{col}')")
    return data

# ─────────────────────────────────────────────────────────────────────────────
# 2. SURFACE / LEXICAL METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Question-word taxonomy
QTYPE_PATTERNS: List[Tuple[str, str]] = [
    ("Who",         r'^\s*who\b'),
    ("What",        r'^\s*what\b'),
    ("Which",       r'^\s*which\b'),
    ("When",        r'^\s*when\b'),
    ("Where",       r'^\s*where\b'),
    ("Why",         r'^\s*why\b'),
    ("How_many",    r'^\s*how\s+many\b'),
    ("How_much",    r'^\s*how\s+much\b'),
    ("How",         r'^\s*how\b'),
    ("Is_Are",      r'^\s*(is|are|was|were)\b'),
    ("Do_Does_Did", r'^\s*(do|does|did)\b'),
    ("Can_Could",   r'^\s*(can|could|may|might|shall|should|will|would)\b'),
    ("List",        r'^\s*(list|give|show|find|return|retrieve)\b'),
    ("Other",       r'.*'),
]


def classify_question_type(text: str) -> str:
    t = text.strip().lower()
    for label, pattern in QTYPE_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return label
    return "Other"


STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","shall","should","may","might",
    "must","can","could","of","in","on","at","to","for","with","by","from",
    "and","or","but","if","as","that","this","these","those","it","its",
    "what","which","who","when","where","why","how","all","any","both",
    "each","few","more","most","other","some","such","no","nor","not",
    "only","same","so","than","too","very","just","also","about","after",
}


def surface_features(text: str) -> Dict[str, Any]:
    text   = text.strip()
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    words  = [t for t in tokens if re.match(r"[a-z]", t)]

    char_len   = len(text)
    word_count = len(words)
    types      = set(words)
    ttr        = len(types) / len(words) if words else 0.0
    content    = [w for w in words if w not in STOP_WORDS]
    content_ratio = len(content) / len(words) if words else 0.0
    avg_word_len  = np.mean([len(w) for w in words]) if words else 0.0

    clause_markers = len(re.findall(
        r'\b(which|that|who|whom|whose|where|when|because|although|since|'
        r'unless|whether|if|as|while|whereas|after|before|until|once)\b',
        text, re.IGNORECASE))

    qtype = classify_question_type(text)

    has_comparative = int(bool(re.search(
        r'\b(more|less|fewer|higher|lower|greater|older|newer|better|worse'
        r'|most|least|highest|lowest|greatest|oldest|newest|best|worst)\b',
        text, re.IGNORECASE)))

    has_negation = int(bool(re.search(
        r"\b(not|no|never|neither|nor|without|except|exclude|n't)\b",
        text, re.IGNORECASE)))

    has_temporal = int(bool(re.search(
        r'\b(before|after|since|until|during|between|in\s+\d{4}|'
        r'by\s+\d{4}|as\s+of|current|latest|recent|historical|past|future)\b',
        text, re.IGNORECASE)))

    has_numeric = int(bool(re.search(r'\b\d+\b', text)))

    has_aggregation = int(bool(re.search(
        r'\b(how\s+many|how\s+much|total|sum|average|mean|count|number\s+of|'
        r'percentage|proportion|rate|most|least|maximum|minimum|highest|lowest)\b',
        text, re.IGNORECASE)))

    has_question_mark = int(text.strip().endswith("?"))

    return {
        "char_len"         : char_len,
        "word_count"       : word_count,
        "type_count"       : len(types),
        "ttr"              : round(ttr, 4),
        "content_ratio"    : round(content_ratio, 4),
        "avg_word_len"     : round(avg_word_len, 3),
        "clause_count"     : clause_markers,
        "question_type"    : qtype,
        "has_comparative"  : has_comparative,
        "has_negation"     : has_negation,
        "has_temporal"     : has_temporal,
        "has_numeric"      : has_numeric,
        "has_aggregation"  : has_aggregation,
        "has_question_mark": has_question_mark,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. READABILITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def readability_features(text: str) -> Dict[str, float]:
    if not USE_TEXTSTAT:
        return {"flesch_reading_ease": np.nan,
                "gunning_fog": np.nan,
                "flesch_kincaid_grade": np.nan,
                "smog_index": np.nan}
    try:
        import textstat
        return {
            "flesch_reading_ease"  : textstat.flesch_reading_ease(text),
            "gunning_fog"          : textstat.gunning_fog(text),
            "flesch_kincaid_grade" : textstat.flesch_kincaid_grade(text),
            "smog_index"           : textstat.smog_index(text),
        }
    except Exception:
        return {"flesch_reading_ease": np.nan, "gunning_fog": np.nan,
                "flesch_kincaid_grade": np.nan, "smog_index": np.nan}

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPACY: POS, DEPENDENCY DEPTH, NAMED ENTITIES
# ─────────────────────────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(SPACY_MODEL)
            print(f"  [spaCy] Loaded model: {SPACY_MODEL}")
        except Exception as e:
            print(f"  [WARN] spaCy load failed: {e}")
            _nlp = False
    return _nlp


def spacy_features(text: str, nlp) -> Dict[str, Any]:
    if nlp is False:
        return {}
    try:
        doc = nlp(text)
    except Exception:
        return {}

    tokens = [t for t in doc if not t.is_space]
    n      = len(tokens) if tokens else 1

    pos_c  = Counter(t.pos_ for t in tokens)
    pos_feats = {f"pos_{p}": round(100 * pos_c.get(p, 0) / n, 2)
                 for p in ["NOUN","VERB","ADJ","ADV","PROPN","NUM",
                            "ADP","DET","PRON","CCONJ","SCONJ","PUNCT"]}

    tag_c = Counter(t.tag_ for t in tokens)
    tense_feats = {
        "pct_VBZ_VBP": round(100 * (tag_c.get("VBZ",0)+tag_c.get("VBP",0))/n, 2),
        "pct_VBD"    : round(100 * tag_c.get("VBD",0)/n, 2),
        "pct_VBN"    : round(100 * tag_c.get("VBN",0)/n, 2),
    }

    def _tree_depth(token):
        children = list(token.children)
        if not children:
            return 0
        return 1 + max(_tree_depth(c) for c in children)

    roots     = [t for t in doc if t.dep_ == "ROOT"]
    dep_depth = max((_tree_depth(r) for r in roots), default=0)

    dep_labels  = [t.dep_ for t in tokens]
    has_passive = int("nsubjpass" in dep_labels or "auxpass" in dep_labels)

    ents       = doc.ents
    ne_count   = len(ents)
    ne_density = round(ne_count / n, 4)
    ne_types   = Counter(e.label_ for e in ents)
    ne_feats   = {f"ne_{et}": ne_types.get(et, 0)
                 for et in ["ORG","PERSON","GPE","DATE","MONEY","PERCENT",
                             "LAW","NORP","PRODUCT","EVENT","TIME","CARDINAL"]}

    return {
        "dep_tree_depth"  : dep_depth,
        "has_passive"     : has_passive,
        "ne_count"        : ne_count,
        "ne_density"      : ne_density,
        **pos_feats,
        **tense_feats,
        **ne_feats,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. LEXICAL DIVERSITY — MATTR & MTLD
# ─────────────────────────────────────────────────────────────────────────────

def mattr(tokens: List[str], window: int = 50) -> float:
    """Moving-Average Type-Token Ratio. Robust to text length."""
    if len(tokens) < window:
        types = set(tokens)
        return len(types) / len(tokens) if tokens else 0.0
    ttrs = []
    for i in range(len(tokens) - window + 1):
        w = tokens[i:i+window]
        ttrs.append(len(set(w)) / window)
    return float(np.mean(ttrs))


def mtld(tokens: List[str], threshold: float = 0.720) -> float:
    """
    Measure of Textual Lexical Diversity — computed bidirectionally
    and averaged.
    """
    def _one_pass(toks):
        factors  = 0
        n_tokens = 0
        types    = set()
        for tok in toks:
            n_tokens += 1
            types.add(tok)
            ttr = len(types) / n_tokens
            if ttr <= threshold:
                factors  += 1
                n_tokens  = 0
                types     = set()
        if n_tokens > 0:
            partial_ttr = len(types) / n_tokens
            if partial_ttr < 1.0:
                factors += (1.0 - partial_ttr) / (1.0 - threshold)
        return len(toks) / factors if factors > 0 else float(len(toks))

    if len(tokens) < 10:
        return float("nan")
    fwd = _one_pass(tokens)
    bwd = _one_pass(list(reversed(tokens)))
    return round((fwd + bwd) / 2, 3)


def corpus_lexical_diversity(series: pd.Series) -> Dict[str, float]:
    all_tokens = []
    per_ttr    = []
    per_mattr  = []
    for text in series:
        toks = re.findall(r"[a-z']+", text.lower())
        if not toks:
            continue
        all_tokens.extend(toks)
        per_ttr.append(len(set(toks)) / len(toks))
        per_mattr.append(mattr(toks))

    corpus_toks = all_tokens
    return {
        "corpus_TTR"       : round(len(set(corpus_toks)) / len(corpus_toks), 4)
                             if corpus_toks else np.nan,
        "corpus_MATTR"     : round(mattr(corpus_toks, window=50), 4),
        "corpus_MTLD"      : mtld(corpus_toks),
        "mean_per_q_TTR"   : round(float(np.mean(per_ttr)), 4)   if per_ttr   else np.nan,
        "std_per_q_TTR"    : round(float(np.std(per_ttr,ddof=1)),4) if len(per_ttr)>1 else np.nan,
        "mean_per_q_MATTR" : round(float(np.mean(per_mattr)),4)  if per_mattr else np.nan,
        "std_per_q_MATTR"  : round(float(np.std(per_mattr,ddof=1)),4) if len(per_mattr)>1 else np.nan,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. SELF-BLEU (corpus-level diversity)
# ─────────────────────────────────────────────────────────────────────────────

def _sentence_bleu_1gram(hypothesis: List[str],
                          references: List[List[str]]) -> float:
    """Unigram BLEU of hypothesis against a list of reference token lists."""
    if not hypothesis:
        return 0.0
    ref_counts: Counter = Counter()
    for ref in references:
        ref_counts |= Counter(ref)
    hyp_counts = Counter(hypothesis)
    clipped    = {tok: min(cnt, ref_counts[tok]) for tok, cnt in hyp_counts.items()}
    precision  = sum(clipped.values()) / len(hypothesis)
    ref_lens   = [len(r) for r in references]
    closest_len = min(ref_lens, key=lambda rl: abs(rl - len(hypothesis)),
                      default=len(hypothesis))
    bp = 1.0 if len(hypothesis) >= closest_len else \
         math.exp(1 - closest_len / len(hypothesis))
    return round(bp * precision, 4)


def self_bleu(series: pd.Series, sample_n: int = SELF_BLEU_SAMPLE_N) -> float:
    """
    Mean unigram Self-BLEU over a random sample.
    Lower value = more lexically diverse corpus.
    """
    tokenised = [re.findall(r"[a-z']+", t.lower()) for t in series if t.strip()]
    if len(tokenised) > sample_n:
        random.seed(42)
        tokenised = random.sample(tokenised, sample_n)
    if len(tokenised) < 2:
        return float("nan")
    scores = []
    for i, hyp in enumerate(tokenised):
        refs = [tokenised[j] for j in range(len(tokenised)) if j != i]
        scores.append(_sentence_bleu_1gram(hyp, refs))
    return round(float(np.mean(scores)), 4)

# ─────────────────────────────────────────────────────────────────────────────
# 7. SENTENCE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

_sbert = None


def _get_sbert():
    global _sbert
    if _sbert is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sbert = SentenceTransformer(SBERT_MODEL)
            print(f"  [SBERT] Loaded model: {SBERT_MODEL}")
        except Exception as e:
            print(f"  [WARN] sentence-transformers unavailable: {e}")
            _sbert = False
    return _sbert


def encode_corpus(series: pd.Series, sample_n: int = EMBEDDING_SAMPLE_N,
                  batch_size: int = 64) -> Optional[np.ndarray]:
    sbert = _get_sbert()
    if sbert is False:
        return None
    texts = series.tolist()
    if len(texts) > sample_n:
        random.seed(42)
        texts = random.sample(texts, sample_n)
    try:
        embs = sbert.encode(texts, batch_size=batch_size,
                             show_progress_bar=False, convert_to_numpy=True)
        return embs.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Encoding failed: {e}")
        return None


def intra_domain_cosine(embs: np.ndarray, sample_pairs: int = 5000) -> float:
    n = len(embs)
    if n < 2:
        return float("nan")
    pairs = list(itertools.combinations(range(n), 2))
    if len(pairs) > sample_pairs:
        random.seed(42)
        pairs = random.sample(pairs, sample_pairs)
    sims = [float(cosine_similarity(embs[i].reshape(1,-1),
                                     embs[j].reshape(1,-1))[0,0])
            for i, j in pairs]
    return round(float(np.mean(sims)), 4)


def inter_domain_cosine_centroid(embs_a: np.ndarray,
                                  embs_b: np.ndarray) -> float:
    ca = embs_a.mean(axis=0, keepdims=True)
    cb = embs_b.mean(axis=0, keepdims=True)
    return round(float(cosine_similarity(ca, cb)[0,0]), 4)


def wasserstein_dist_1d_projected(embs_a: np.ndarray,
                                    embs_b: np.ndarray,
                                    n_projections: int = 50) -> float:
    """
    Sliced Wasserstein distance between two embedding distributions.
    Projects onto random 1-D lines and averages 1-D Wasserstein distances.
    """
    np.random.seed(42)
    D = embs_a.shape[1]
    distances = []
    for _ in range(n_projections):
        direction = np.random.randn(D).astype(np.float32)
        direction /= np.linalg.norm(direction)
        proj_a = embs_a @ direction
        proj_b = embs_b @ direction
        distances.append(wasserstein_distance(proj_a, proj_b))
    return round(float(np.mean(distances)), 4)

# ─────────────────────────────────────────────────────────────────────────────
# 8. STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def kruskal_dunn_nlq(groups: Dict[str, np.ndarray],
                      metric: str) -> Dict[str, Any]:
    """
    Kruskal-Wallis H + Dunn Bonferroni + Cohen's d + epsilon-squared (η²).
    """
    arrays = [v for v in groups.values() if len(v) >= 3]
    names  = [k for k, v in groups.items() if len(v) >= 3]
    N      = sum(len(a) for a in arrays)

    def _skip(reason: str) -> Dict[str, Any]:
        return {"metric": metric, "kw_H": None, "kw_p": None,
                "epsilon_sq": None, "skipped": reason,
                "pairwise_p": {}, "pairwise_d": {},
                "means"  : {n: round(float(groups[n].mean()),3) for n in names},
                "medians": {n: round(float(np.median(groups[n])),3) for n in names},
                "stds"   : {n: round(float(groups[n].std(ddof=1)),3)
                             if len(groups[n])>1 else np.nan for n in names}}

    if len(arrays) < 2:
        return _skip("fewer than 2 groups with n>=3")
    try:
        H, p = stats.kruskal(*arrays)
    except ValueError as exc:
        return _skip(str(exc))
    except Exception as exc:
        return _skip(str(exc))

    eps_sq = eta_squared(H, len(arrays), N)

    all_data  = np.concatenate(arrays)
    all_ranks = stats.rankdata(all_data)
    gsizes    = [len(a) for a in arrays]
    bounds    = np.cumsum([0] + gsizes)
    granks    = [all_ranks[bounds[i]:bounds[i+1]] for i in range(len(arrays))]
    m         = len(arrays)
    n_comp    = m*(m-1)/2
    pairwise_p: Dict[str, float] = {}
    pairwise_d: Dict[str, float] = {}

    for i, j in itertools.combinations(range(m), 2):
        se  = math.sqrt((N*(N+1)/12.0)*(1.0/gsizes[i]+1.0/gsizes[j]))
        z   = abs(granks[i].mean()-granks[j].mean())/se if se != 0 else 0.0
        adj = min(2*(1-stats.norm.cdf(z))*n_comp, 1.0)
        key = f"{names[i]} vs {names[j]}"
        pairwise_p[key] = round(adj, 5)
        pairwise_d[key] = round(cohen_d(arrays[i], arrays[j]), 3)

    return {"metric": metric, "kw_H": round(H,3), "kw_p": round(p,5),
            "epsilon_sq": round(eps_sq, 4) if not math.isnan(eps_sq) else None,
            "skipped": None, "pairwise_p": pairwise_p, "pairwise_d": pairwise_d,
            "means"  : {names[i]: round(arrays[i].mean(),3) for i in range(len(names))},
            "medians": {names[i]: round(float(np.median(arrays[i])),3) for i in range(len(names))},
            "stds"   : {names[i]: round(arrays[i].std(ddof=1),3) for i in range(len(names))}}


def statistical_tests_nlq(feat_dfs: Dict[str, pd.DataFrame],
                            metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        try:
            groups = {d: df[metric].dropna().values.astype(float)
                      for d, df in feat_dfs.items()
                      if metric in df.columns and not df[metric].isna().all()}
            if len(groups) < 2:
                continue
            result = kruskal_dunn_nlq(groups, metric)
            kw_p   = result["kw_p"]
            sig    = ("Yes"     if kw_p is not None and kw_p < 0.05 else
                      "Skipped" if result.get("skipped") else "No")
            row: Dict[str, Any] = {
                "Metric"      : metric,
                "KW_H"        : result["kw_H"],
                "KW_p"        : kw_p,
                "Epsilon_sq"  : result.get("epsilon_sq"),
                "Significant" : sig,
                "Skip_reason" : result.get("skipped") or "",
            }
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
            print(f"  [WARN] stats NLQ '{metric}': {exc}")
            rows.append({"Metric": metric, "KW_H": None, "KW_p": None,
                         "Epsilon_sq": None, "Significant": "Error",
                         "Skip_reason": str(exc)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 9. TF-IDF INTER-DOMAIN SEPARABILITY
# ─────────────────────────────────────────────────────────────────────────────

def tfidf_separability(domain_series: Dict[str, pd.Series]) -> pd.DataFrame:
    domains  = list(domain_series.keys())
    corpora  = [" ".join(domain_series[d].tolist()) for d in domains]
    vec      = TfidfVectorizer(ngram_range=(1,2), min_df=2,
                               max_features=10000, stop_words="english")
    try:
        M = vec.fit_transform(corpora)
    except Exception as e:
        print(f"  [WARN] TF-IDF failed: {e}")
        return pd.DataFrame()

    rows = []
    for i, j in itertools.combinations(range(len(domains)), 2):
        sim = cosine_similarity(M[i], M[j])[0,0]
        rows.append({"Domain_A": domains[i], "Domain_B": domains[j],
                     "TF-IDF_cosine": round(float(sim), 4)})

    feature_names = vec.get_feature_names_out()
    disc_rows = []
    for i, d in enumerate(domains):
        weights  = M[i].toarray().ravel()
        top_idx  = weights.argsort()[-20:][::-1]
        top_terms = [(feature_names[k], round(float(weights[k]), 4))
                     for k in top_idx if float(weights[k]) > 0]
        disc_rows.append({"Domain": d,
                          "Top20_discriminative_terms":
                              "; ".join(f"{t}({w})" for t, w in top_terms)})

    combined = pd.DataFrame(rows)
    disc_df  = pd.DataFrame(disc_rows)
    return combined, disc_df

# ─────────────────────────────────────────────────────────────────────────────
# 10. AGGREGATE TABLE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def surface_summary_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    scalar_cols = [
        "char_len","word_count","type_count","ttr","content_ratio",
        "avg_word_len","clause_count","has_comparative","has_negation",
        "has_temporal","has_numeric","has_aggregation","has_question_mark",
        "flesch_reading_ease","gunning_fog","flesch_kincaid_grade",
    ]
    rows = []
    for domain, df in feat_dfs.items():
        row = {"Domain": domain, "N": len(df)}
        for col in scalar_cols:
            if col not in df.columns:
                continue
            arr = df[col].dropna().values.astype(float)
            if len(arr) == 0:
                continue
            row[f"Mean_{col}"]   = round(arr.mean(), 3)
            row[f"Median_{col}"] = round(float(np.median(arr)), 3)
            row[f"Std_{col}"]    = round(arr.std(ddof=1), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def question_type_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    qtypes = [l for l, _ in QTYPE_PATTERNS]
    rows   = []
    for domain, df in feat_dfs.items():
        total  = len(df)
        counts = df["question_type"].value_counts()
        row    = {"Domain": domain, "N": total}
        for qt in qtypes:
            n = counts.get(qt, 0)
            row[f"{qt}_n"]   = n
            row[f"{qt}_pct"] = round(100*n/total, 2) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def lexical_diversity_table(domain_series: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for domain, series in domain_series.items():
        ld = corpus_lexical_diversity(series)
        row = {"Domain": domain, "N": len(series)}
        row.update(ld)
        rows.append(row)
    return pd.DataFrame(rows)


def pos_profile_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    pos_cols = [c for c in next(iter(feat_dfs.values())).columns
                if c.startswith("pos_") or c.startswith("pct_VB")]
    rows = []
    for domain, df in feat_dfs.items():
        row = {"Domain": domain, "N": len(df)}
        for col in pos_cols:
            arr = df[col].dropna().values.astype(float)
            row[f"Mean_{col}"] = round(arr.mean(), 3) if len(arr) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ner_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    ne_cols = [c for c in next(iter(feat_dfs.values())).columns
               if c.startswith("ne_")]
    rows = []
    for domain, df in feat_dfs.items():
        row = {"Domain": domain, "N": len(df),
               "mean_ne_count"  : round(df["ne_count"].mean(), 3)
                                   if "ne_count" in df else np.nan,
               "mean_ne_density": round(df["ne_density"].mean(), 4)
                                   if "ne_density" in df else np.nan}
        for col in ne_cols:
            arr = df[col].dropna().values.astype(float)
            row[f"mean_{col}"] = round(arr.mean(), 3) if len(arr) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def dep_complexity_table(feat_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for domain, df in feat_dfs.items():
        row = {"Domain": domain, "N": len(df)}
        for col in ["dep_tree_depth","has_passive","clause_count"]:
            if col in df.columns:
                arr = df[col].dropna().values.astype(float)
                row[f"Mean_{col}"]   = round(arr.mean(), 3)
                row[f"Median_{col}"] = round(float(np.median(arr)), 3)
                row[f"Std_{col}"]    = round(arr.std(ddof=1), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def self_bleu_table(domain_series: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for domain, series in domain_series.items():
        print(f"    Self-BLEU for {domain} (n≤{SELF_BLEU_SAMPLE_N})...")
        sb = self_bleu(series)
        rows.append({"Domain": domain, "N": len(series),
                     "Self_BLEU_unigram": sb,
                     "Interpretation": "Lower = more lexically diverse corpus"})
    return pd.DataFrame(rows)


def embedding_similarity_table(domain_series: Dict[str, pd.Series],
                                 domain_embs: Dict[str, Optional[np.ndarray]]
                                 ) -> pd.DataFrame:
    rows = []
    for domain, embs in domain_embs.items():
        if embs is None:
            rows.append({"Type": "Intra-domain", "Pair": f"{domain} internal",
                         "Cosine_mean": None, "Wasserstein": None})
            continue
        sim = intra_domain_cosine(embs)
        rows.append({"Type": "Intra-domain", "Pair": f"{domain} internal",
                     "Cosine_mean": sim, "Wasserstein": None})

    domains = [d for d, e in domain_embs.items() if e is not None]
    for d_a, d_b in itertools.combinations(domains, 2):
        ea, eb = domain_embs[d_a], domain_embs[d_b]
        cos  = inter_domain_cosine_centroid(ea, eb)
        wass = wasserstein_dist_1d_projected(ea, eb)
        rows.append({"Type": "Inter-domain",
                     "Pair": f"{d_a} vs {d_b}",
                     "Cosine_mean": cos, "Wasserstein": wass})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# 11. FULL PER-QUESTION FEATURISATION
# ─────────────────────────────────────────────────────────────────────────────

def featurise_nlq(text: str, nlp=None) -> Dict[str, Any]:
    feat = surface_features(text)
    feat.update(readability_features(text))
    if nlp and USE_SPACY:
        feat.update(spacy_features(text, nlp))
    return feat


def featurise_domain_nlq(series: pd.Series) -> pd.DataFrame:
    nlp  = _get_nlp() if USE_SPACY else None
    rows = []
    for text in series:
        try:
            rows.append(featurise_nlq(text, nlp))
        except Exception as exc:
            print(f"  [WARN] featurise_nlq failed: {exc}")
            rows.append({})
    df = pd.DataFrame(rows)
    df["nlq"] = series.values
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 12. NLQ SCALAR METRICS LIST
# ─────────────────────────────────────────────────────────────────────────────

NLQ_SCALAR_METRICS = [
    "char_len","word_count","type_count","ttr","content_ratio",
    "avg_word_len","clause_count",
    "has_comparative","has_negation","has_temporal","has_numeric",
    "has_aggregation","has_question_mark",
    "flesch_reading_ease","gunning_fog","flesch_kincaid_grade","smog_index",
    "dep_tree_depth","has_passive","ne_count","ne_density",
    "pos_NOUN","pos_VERB","pos_ADJ","pos_ADV","pos_PROPN","pos_NUM",
    "pct_VBZ_VBP","pct_VBD","pct_VBN",
]

# ─────────────────────────────────────────────────────────────────────────────
# 13. NARRATIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def generate_nlq_narrative(domain_series, feat_dfs, surf_tbl, qtype_tbl,
                             lex_tbl, pos_tbl, stats_tbl, sb_tbl,
                             emb_tbl, ner_tbl, dep_tbl) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("APPENDIX A.3 (NLQ COMPONENT) — CROSS-DOMAIN NLQ ANALYSIS")
    lines.append("=" * 80)

    lines.append("\n§1  DATASET SIZES")
    lines.append("-" * 40)
    for domain, series in domain_series.items():
        lines.append(f"  {domain:10s}: {len(series):4d} NLQs")

    lines.append("\n§2  SURFACE METRICS")
    lines.append("-" * 40)
    for _, row in surf_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: mean_words={row.get('Mean_word_count','NA'):.2f}  "
            f"mean_chars={row.get('Mean_char_len','NA'):.1f}  "
            f"mean_TTR={row.get('Mean_ttr','NA'):.4f}  "
            f"content_ratio={row.get('Mean_content_ratio','NA'):.3f}")

    lines.append("\n§3  QUESTION TYPE DISTRIBUTION")
    lines.append("-" * 40)
    top_types = ["What","Which","How_many","List","Is_Are","Who"]
    for _, row in qtype_tbl.iterrows():
        parts = [f"{qt}={row.get(f'{qt}_pct',0):.1f}%" for qt in top_types]
        lines.append(f"  {row['Domain']:10s}: " + "  ".join(parts))

    lines.append("\n§4  LEXICAL DIVERSITY")
    lines.append("-" * 40)
    for _, row in lex_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: corpus_TTR={row.get('corpus_TTR','NA')}  "
            f"MATTR={row.get('corpus_MATTR','NA')}  "
            f"MTLD={row.get('corpus_MTLD','NA')}")

    lines.append("\n§5  POS PROFILE")
    lines.append("-" * 40)
    for _, row in pos_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: "
            f"NOUN={row.get('Mean_pos_NOUN','NA'):.2f}%  "
            f"VERB={row.get('Mean_pos_VERB','NA'):.2f}%  "
            f"PROPN={row.get('Mean_pos_PROPN','NA'):.2f}%  "
            f"NUM={row.get('Mean_pos_NUM','NA'):.2f}%  "
            f"ADJ={row.get('Mean_pos_ADJ','NA'):.2f}%")

    lines.append("\n§6  DEPENDENCY COMPLEXITY")
    lines.append("-" * 40)
    for _, row in dep_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: "
            f"dep_depth={row.get('Mean_dep_tree_depth','NA'):.2f}  "
            f"clauses={row.get('Mean_clause_count','NA'):.2f}  "
            f"passive={row.get('Mean_has_passive','NA'):.3f}")

    lines.append("\n§7  NAMED ENTITY DENSITY")
    lines.append("-" * 40)
    for _, row in ner_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: "
            f"ne_density={row.get('mean_ne_density','NA'):.4f}  "
            f"mean_count={row.get('mean_ne_count','NA'):.3f}")

    lines.append("\n§8  CORPUS DIVERSITY — SELF-BLEU")
    lines.append("-" * 40)
    for _, row in sb_tbl.iterrows():
        lines.append(
            f"  {row['Domain']:10s}: Self-BLEU(unigram)={row.get('Self_BLEU_unigram','NA')}"
            f"  [lower = more diverse]")

    lines.append("\n§9  SEMANTIC EMBEDDING SIMILARITY")
    lines.append("-" * 40)
    for _, row in emb_tbl.iterrows():
        if row["Type"] == "Intra-domain":
            lines.append(f"  {row['Pair']:30s}  cosine={row.get('Cosine_mean','NA')}")
        else:
            lines.append(
                f"  {row['Pair']:30s}  centroid_cosine={row.get('Cosine_mean','NA')}  "
                f"Wasserstein={row.get('Wasserstein','NA')}")

    lines.append("\n§10 STATISTICAL SIGNIFICANCE (KW + Bonferroni Dunn; ε² effect size)")
    lines.append("-" * 40)
    lines.append("  Magnitude thresholds (ε²): negligible<0.01, small 0.01-0.06, "
                 "medium 0.06-0.14, large>=0.14")
    for _, row in stats_tbl.iterrows():
        sig = row.get("Significant")
        if sig == "Yes":
            lines.append(
                f"  '{row['Metric']}'  H={row['KW_H']}  p={row['KW_p']}  "
                f"e2={row.get('Epsilon_sq','NA')}")
            for col in row.index:
                if col.startswith("d_") and pd.notna(row[col]):
                    d   = row[col]
                    mag = "small" if abs(d)<0.5 else ("medium" if abs(d)<0.8 else "large")
                    lines.append(f"    {col[2:]}: d={d:.3f} ({mag})")
        elif sig in ("Skipped","Error"):
            lines.append(f"  '{row['Metric']}' — {row.get('Skip_reason','')}")

    lines.append("\n" + "="*80)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# 14. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_nlq_analysis() -> None:
    ensure_dirs()
    print("\nNLQ Cross-Domain Analysis Pipeline")
    print("=" * 60)

    print("\n[1] Loading NLQ data...")
    domain_series = load_nlq_data(DOMAIN_FILES)
    if not domain_series:
        print("[ERROR] No NLQ data loaded. Check DOMAIN_FILES and NLQ_COL.")
        return

    print("\n[2] Featurising NLQs (surface + readability + POS/dep/NER)...")
    feat_dfs: Dict[str, pd.DataFrame] = {}
    for domain, series in domain_series.items():
        print(f"  {domain}...")
        feat_dfs[domain] = featurise_domain_nlq(series)

    print("\n[3] Building analysis tables...")
    surf_tbl  = surface_summary_table(feat_dfs)
    qtype_tbl = question_type_table(feat_dfs)
    pos_tbl   = pos_profile_table(feat_dfs) if USE_SPACY else pd.DataFrame()
    ner_tbl   = ner_table(feat_dfs)         if USE_SPACY else pd.DataFrame()
    dep_tbl   = dep_complexity_table(feat_dfs)
    lex_tbl   = lexical_diversity_table(domain_series)

    print("\n[4] Running statistical tests (KW + Dunn + e2) on NLQ metrics...")
    stats_tbl = statistical_tests_nlq(feat_dfs, NLQ_SCALAR_METRICS)

    print("\n[5] Computing Self-BLEU (corpus-level lexical diversity)...")
    sb_tbl = self_bleu_table(domain_series)

    print("\n[6] Computing sentence embeddings (SBERT)...")
    domain_embs: Dict[str, Optional[np.ndarray]] = {}
    if USE_SENTENCE_TRANSFORMERS:
        for domain, series in domain_series.items():
            print(f"  Encoding {domain}...")
            domain_embs[domain] = encode_corpus(series)
    else:
        domain_embs = {d: None for d in domain_series}
    emb_tbl = embedding_similarity_table(domain_series, domain_embs)

    print("\n[7] TF-IDF separability and discriminative terms...")
    tfidf_result = tfidf_separability(domain_series)
    if isinstance(tfidf_result, tuple):
        tfidf_sim_tbl, tfidf_disc_tbl = tfidf_result
    else:
        tfidf_sim_tbl = tfidf_result
        tfidf_disc_tbl = pd.DataFrame()

    combined = pd.concat(
        [df.assign(Domain=domain) for domain, df in feat_dfs.items()],
        ignore_index=True)

    print("\n[8] Writing CSV outputs...")
    write_csv(surf_tbl,      "NLQ_B1_surface_metrics.csv")
    write_csv(qtype_tbl,     "NLQ_B2_question_type_distribution.csv")
    write_csv(lex_tbl,       "NLQ_B3_lexical_diversity.csv")
    if not pos_tbl.empty:
        write_csv(pos_tbl,   "NLQ_B4_pos_profile.csv")
    write_csv(stats_tbl,     "NLQ_B5_statistical_tests.csv")
    write_csv(sb_tbl,        "NLQ_B6_self_bleu.csv")
    write_csv(emb_tbl,       "NLQ_B7_embedding_similarity.csv")
    if not ner_tbl.empty:
        write_csv(ner_tbl,   "NLQ_B8_named_entity_density.csv")
    write_csv(dep_tbl,       "NLQ_B9_dependency_complexity.csv")
    write_csv(combined,      "NLQ_B10_combined_feature_vectors.csv")
    write_csv(tfidf_sim_tbl, "NLQ_B11_tfidf_separability.csv")
    if not tfidf_disc_tbl.empty:
        write_csv(tfidf_disc_tbl, "NLQ_B12_discriminative_terms.csv")

    print(f"\n[9] Writing Excel workbook → {XLSX_OUT}")
    with pd.ExcelWriter(XLSX_OUT, engine="openpyxl") as writer:
        surf_tbl.to_excel(writer,      sheet_name="B1_SurfaceMetrics",      index=False)
        qtype_tbl.to_excel(writer,     sheet_name="B2_QuestionTypes",       index=False)
        lex_tbl.to_excel(writer,       sheet_name="B3_LexicalDiversity",    index=False)
        if not pos_tbl.empty:
            pos_tbl.to_excel(writer,   sheet_name="B4_POS_Profile",         index=False)
        stats_tbl.to_excel(writer,     sheet_name="B5_StatisticalTests",    index=False)
        sb_tbl.to_excel(writer,        sheet_name="B6_SelfBLEU",            index=False)
        emb_tbl.to_excel(writer,       sheet_name="B7_EmbeddingSimilarity", index=False)
        if not ner_tbl.empty:
            ner_tbl.to_excel(writer,   sheet_name="B8_NER_Density",         index=False)
        dep_tbl.to_excel(writer,       sheet_name="B9_DepComplexity",       index=False)
        combined.to_excel(writer,      sheet_name="B10_FeatureVectors",     index=False)
        tfidf_sim_tbl.to_excel(writer, sheet_name="B11_TFIDF_Sep",          index=False)
        if not tfidf_disc_tbl.empty:
            tfidf_disc_tbl.to_excel(writer, sheet_name="B12_DiscrimTerms",  index=False)

        method = [
            ["Dimension","Method","Rationale"],
            ["Surface length","word/char/type count",
             "Establishes baseline complexity of input questions"],
            ["Question type","Regex taxonomy",
             "WHO/WHAT/HOW MANY etc. map to different SPARQL forms (ASK/SELECT/COUNT)"],
            ["Lexical diversity","TTR, MATTR, MTLD",
             "MATTR/MTLD are length-invariant unlike raw TTR — valid cross-corpus"],
            ["Readability","Flesch, Gunning-Fog",
             "Captures linguistic formality differences across domains"],
            ["POS profile","spaCy Universal POS",
             "NOUN/PROPN/NUM ratios reflect domain ontology richness"],
            ["Dependency depth","spaCy dep parser",
             "Deeper trees = syntactically more complex questions"],
            ["NER density","spaCy NER",
             "High PROPN/ORG/LAW density signals domain-specific named entities"],
            ["Corpus diversity","Self-BLEU",
             "Lower Self-BLEU = more lexically varied corpus"],
            ["Semantic similarity","SBERT cosine + Wasserstein",
             "Embedding space distances capture semantic (not just lexical) divergence"],
            ["Domain separability","TF-IDF cosine",
             "Centroid cosine quantifies how distinct corpora are in vocabulary space"],
            ["Statistics","KW H + Dunn + e2",
             "e2 provides effect size for non-parametric omnibus test"],
        ]
        pd.DataFrame(method[1:], columns=method[0]).to_excel(
            writer, sheet_name="Methodology", index=False)

    print(f"\n[10] Writing narrative summary → {TXT_OUT}")
    narrative = generate_nlq_narrative(
        domain_series, feat_dfs, surf_tbl, qtype_tbl, lex_tbl,
        pos_tbl, stats_tbl, sb_tbl, emb_tbl, ner_tbl, dep_tbl)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(narrative)
    print("\n" + narrative)

    print(f"\n✓ Excel  : {XLSX_OUT}")
    print(f"✓ Summary: {TXT_OUT}")
    print(f"✓ CSVs   : {CSV_DIR}/")


if __name__ == "__main__":
    run_nlq_analysis()
