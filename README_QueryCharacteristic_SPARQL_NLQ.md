---

## `src/query_characteristics/`

Analysis of structural and linguistic characteristics of the NLQ↔SPARQL corpus across all evaluated domains. Two independent pipelines are provided: one for the SPARQL side of the corpus and one for the NLQ side. Both write results into their respective `results/csv/` subfolder.

---

### `src/query_characteristics/sparql/`

**File:** `sparql_domain_comparison.py`

Cross-domain structural analysis of SPARQL queries. Generates all outputs cited in Appendix A.3 of the paper.

#### Feature dimensions extracted per query

| Dimension | Approach |
|---|---|
| Query form | Regex detection of `SELECT` / `ASK` / `DESCRIBE` / `CONSTRUCT` |
| Complexity | Weighted sum of 16 SPARQL operators |
| N-ary patterns | Blank-node sets, `FILTER EXISTS` pairs, inline pipeline comments |
| Compositionality | `operator_diversity × 2 + join_count × 1.5 + nesting_depth` |
| Entities / relations | `?variable` count, `rdf:type` triples, domain predicate triples |
| Lexical diversity | Type-Token Ratio on non-boilerplate tokens |
| Size | Line count, token count, character count |
| Boolean flags | Negation (`FILTER NOT EXISTS` / `MINUS`), NLQ annotation present |

#### Statistical methods

- Kruskal-Wallis H-test (non-parametric omnibus)
- Dunn post-hoc with Bonferroni correction
- Cohen's *d* per pairwise domain comparison (`< 0.5` small, `0.5–0.8` medium, `> 0.8` large)
- Token-set Jaccard similarity (intra-domain)
- TF-IDF cosine similarity (inter-domain)

#### Configuration

Edit `DOMAIN_FILES` at the top of the script to point to your `.xlsx` input files. Column names default to `"SPARQL Query"` and `"Natural Language Query"` but are located by fuzzy match if the exact name differs.

```python
DOMAIN_FILES = {
    "Domain_A": "data/domain_a.xlsx",
    "Domain_B": "data/domain_b.xlsx",
    "Domain_C": "data/domain_c.xlsx",
}
```

#### Dependencies

```bash
pip install pandas openpyxl scipy scikit-learn numpy xlsxwriter
```

#### Usage

```bash
python src/query_characteristics/sparql/sparql_domain_comparison.py
```

#### `results/csv/` outputs

```
results/csv/
├── A3_1_domain_feature_profiles.csv        # Table A3.1 — aggregate profiles per domain
├── A3_2_paired_nlq_sparql_examples.csv     # Table A3.2 — real paired NLQ↔SPARQL examples
├── A3_3_statistical_tests.csv              # Table A3.3 — KW H-test + Dunn + Cohen's d
├── A3_4_structural_similarity.csv          # Table A3.4 — Jaccard + TF-IDF cosine
├── scalar_summary.csv                      # Per-metric mean / median / std / min / max / P25 / P75
├── operator_profile.csv                    # Mean operator counts + % non-zero per domain
├── compositionality.csv                    # Operator diversity, join count, nesting depth
├── boolean_features.csv                    # Negation and annotation rates per domain
├── nary_patterns.csv                       # N-ary pattern counts per domain
├── query_form_distribution.csv             # SELECT / ASK / DESCRIBE / CONSTRUCT %
├── subgroup_analysis.csv                   # Per-query-form breakdown within each domain
└── all_feature_vectors.csv                 # Full per-query feature matrix
```

> **Note on Table A3.2:** One representative query is selected per (domain, complexity level) stratum — the query whose `complexity_score` is closest to the stratum median. Where no NLQ column is present in the input file, a structural description is auto-generated and marked `[Auto-inferred]`.

---

### `src/query_characteristics/nlq/`

**File:** `nlq_domain_analysis.py`

Cross-domain linguistic analysis of Natural Language Questions. Complements the SPARQL pipeline by characterising the input-language side of the corpus.

#### Feature dimensions extracted per NLQ

| Dimension | Approach |
|---|---|
| Surface length | Word count, character count, type count, TTR, content-word ratio |
| Question type | Regex taxonomy: `WHO / WHAT / WHICH / HOW MANY / LIST / IS_ARE` etc. |
| Readability | Flesch Reading Ease, Gunning Fog, Flesch-Kincaid Grade, SMOG |
| POS profile | spaCy Universal POS % per tag (`NOUN`, `VERB`, `PROPN`, `NUM`, …) |
| Dependency complexity | Dependency tree depth, clause count, passive voice detection |
| Named entity density | NE count and density; type breakdown (`ORG`, `GPE`, `LAW`, `DATE`, …) |
| Lexical diversity | Corpus TTR, MATTR (window = 50), MTLD (bidirectional) |
| Boolean flags | Negation, temporal reference, numeric quantifier, aggregation intent |
| Corpus diversity | Self-BLEU unigram — lower = more lexically diverse corpus |
| Semantic similarity | SBERT intra-domain cosine; inter-domain centroid cosine + sliced Wasserstein |
| Domain separability | TF-IDF bigram centroid cosine; top-20 discriminative terms per domain |

#### Statistical methods

- Kruskal-Wallis H-test
- Dunn post-hoc with Bonferroni correction
- Cohen's *d* (pairwise)
- Epsilon-squared η² (effect size for omnibus KW; `< 0.01` negligible, `0.01–0.06` small, `0.06–0.14` medium, `≥ 0.14` large)

#### Configuration

```python
DOMAIN_FILES = {
    "Domain_A": "data/domain_a.xlsx",
    ...
}
NLQ_COL = "Question"
```

Feature flags — set to `False` to skip optional heavy dependencies:

```python
USE_SPACY                 = True   # POS, dependency tree, NER
USE_SENTENCE_TRANSFORMERS = True   # SBERT embeddings (~90 MB on first run)
USE_TEXTSTAT              = True   # readability scores
```

#### Dependencies

```bash
pip install pandas openpyxl scipy scikit-learn numpy spacy nltk \
            sentence-transformers textstat

python -m spacy download en_core_web_sm
```

#### Usage

```bash
python src/query_characteristics/nlq/nlq_domain_analysis.py
```

#### `results/csv/` outputs

```
results/csv/
├── NLQ_B1_surface_metrics.csv              # Length, TTR, content ratio, readability
├── NLQ_B2_question_type_distribution.csv   # WHO / WHAT / HOW MANY / LIST … counts and %
├── NLQ_B3_lexical_diversity.csv            # Corpus TTR, MATTR, MTLD per domain
├── NLQ_B4_pos_profile.csv                  # Mean POS tag % per domain
├── NLQ_B5_statistical_tests.csv            # KW + Dunn + Cohen's d + ε²
├── NLQ_B6_self_bleu.csv                    # Unigram Self-BLEU per domain
├── NLQ_B7_embedding_similarity.csv         # Intra / inter cosine + Wasserstein
├── NLQ_B8_named_entity_density.csv         # NE density and type counts
├── NLQ_B9_dependency_complexity.csv        # Tree depth, clause count, passive rate
├── NLQ_B10_combined_feature_vectors.csv    # Full per-question feature matrix
├── NLQ_B11_tfidf_separability.csv          # Inter-domain TF-IDF cosine
└── NLQ_B12_discriminative_terms.csv        # Top-20 domain-discriminative terms
```

> **Note on SBERT:** Set `USE_SENTENCE_TRANSFORMERS = False` if running offline. All other analyses proceed normally; `NLQ_B7_embedding_similarity.csv` will contain `null` values for the affected rows.

---
