# SPARQL-Augmented LLM based Question Answering System

Codebase, prompts, and evaluation pipeline for an empirical study of SPARQL-encoded query reformulations and their effect on Large Language Model performance across domain-specific Question Answering tasks.

Three domains are covered: financial QA (FiQA), legal QA (LegalQA), and telecommunications QA (TelcoQA). Six language models were evaluated under four prompting strategies: Natural Language Query (NLQ), Chain-of-Thought (CoT), SPARQL, and SPARQL combined with CoT. Performance is measured along three axes: knowledge density of model outputs, factual grounding against ground-truth answers, and shifts in model-internal latent features.

> **Note on TelcoQA:** Due to proprietary restrictions the TelcoQA source data is not included. Evaluation scripts for that domain are provided with abstracted data-loading modules. See `data/raw/TelcoQA_readme.txt` for the required schema to reproduce those experiments.

---

## Research Questions

| ID | Question |
|---|---|
| RQ1 | To what extent does Formal Language Query (FLQ) reformulation increase the extractable knowledge content of model responses relative to Natural Language Queries? |
| RQ2 | To what extent do FLQ and CoT reformulations improve propositional grounding over NLQ baselines? |
| RQ3 | To what extent does FLQ and CoT prompt structure induce distinguishable shifts in model inference (Attention, Perplexity, Confidence), and do those shifts correlate with output quality? |

---

## Repository Structure

```
.
├── src/
│   ├── parsing/                  # Scripts for parsing LLM responses for 6 models across 3 domains
│   ├── generation/
│   │   └── templates/            # .txt prompt templates for NLQ, CoT, SPARQL, and SPARQL+CoT
│   └── transformation/           # NLQ-to-SPARQL conversion logic and query datasets for LegalQA and FiQA
├── data/
│   └── raw/                      # FiQA and LegalQA source files (Question-GroundTruth pairs)
├── eval/
│   ├── RQ1_KnowledgeDensity/
│   │   ├── fiqa/
│   │   ├── legalqa/
│   │   └── telcoqa/
│   ├── RQ2_FactualRecall/
│   │   ├── fiqa/
│   │   ├── legalqa/
│   │   └── telcoqa/
│   └── RQ3_LatentSpace/
│       ├── fiqa/
│       ├── legalqa/
│       └── telcoqa/
└── results/
    ├── RQ1_KnowledgeDensity/
    │   ├── fiqa/                 # win/loss plots (.png)
    │   ├── legalqa/              # win/loss plots (.png)
    │   └── telcoqa/              # win/loss plots (.png)
    ├── RQ2_FactualRecall/
    │   ├── fiqa/                 # BERTScore tables (.csv), summary (.txt)
    │   ├── legalqa/              # BERTScore tables (.csv), summary (.txt)
    │   └── telcoqa/              # BERTScore tables (.csv), summary (.txt)
    └── RQ3_LatentSpace/
        ├── fiqa/                 # angular distance correlation results (.csv, .png)
        ├── legalqa/              # angular distance correlation results (.csv, .png)
        └── telcoqa/              # angular distance correlation results (.csv, .png)
```

| Folder | Contents | Description |
|---|---|---|
| `src/parsing/` | Python scripts | Scripts for parsing LLM responses for 6 models across 3 domains. |
| `src/generation/` | Python scripts, templates | Implementation of NLQ, CoT, SPARQL, and SPARQL+CoT prompting methods and `.txt` templates. |
| `src/transformation/` | Python scripts, .xlsx | NLQ to SPARQL conversion logic and query datasets for LegalQA and FiQA. |
| `data/raw/` | .xlsx files | FiQA and LegalQA source files with Question and GroundTruth columns. Includes `TelcoQA_readme.txt`. |
| `eval/` | Python scripts | Evaluation logic for Knowledge Density (RQ1), Factual Recall (RQ2), and Latent Space (RQ3). Each RQ has subfolders per domain. |
| `results/` | .png, .csv, .txt | Mirrors the `eval/` structure. Contains win/loss plots (RQ1), BERTScore tables (RQ2), and angular distance correlation results (RQ3). |

---

## Experimental Pipeline

The pipeline runs in five sequential stages.

| Stage | Name | What happens |
|---|---|---|
| 1 | Sourcing | Domain-specific QA datasets are loaded from `data/raw/`. Each row contains a natural language question and its ground truth answer. |
| 2 | Transformation | Each NLQ is converted to a SPARQL query using `src/transformation/`. This produces the FLQ variants used in inference. |
| 3 | Inference | Six LLMs are prompted with each question under all four conditions. Latent features (attention, perplexity, confidence) are captured at this stage. |
| 4 | Post-Processing | Raw model outputs are passed through `src/parsing/` to extract propositional triples and compute token counts. |
| 5 | Evaluation | Three parallel assessments: knowledge density (triples per token), factual accuracy (BERTScore recall), and latent analysis (angular distance vs. Win/Loss). |

---

## Prompting Methods

| Method | Description |
|---|---|
| NLQ | The question is passed to the model as a natural language sentence with no additional structure. This is the baseline condition. |
| CoT | The prompt instructs the model to reason step by step before producing a final answer. No SPARQL structure is used. |
| SPARQL | The natural language question is reformulated as a SPARQL query and submitted as the prompt, encoding the information need in formal query syntax. |
| SPARQL+CoT | Combines both: the SPARQL query is included alongside a chain-of-thought instruction, pairing structural encoding with explicit reasoning direction. |

---

## Datasets

| Dataset | Domain | Notes |
|---|---|---|
| FiQA | Financial | Publicly available. Included as `data/raw/fiqa.xlsx` with Question and GroundTruth columns. |
| LegalQA | Legal | Publicly available. Included as `data/raw/legalqa.xlsx` with Question and GroundTruth columns. |
| TelcoQA | Telecommunications | Proprietary. Not included. See `data/raw/TelcoQA_readme.txt` for the required schema to reproduce experiments on this domain. |

---

## Evaluation Metrics

### RQ1 — Knowledge Density

Knowledge triples are extracted from each model response using rule-based parsing in `eval/RQ1_KnowledgeDensity/`. Density is computed as the ratio of extracted triples to total token count. Win/loss comparisons are plotted across all four prompting conditions for each domain.

### RQ2 — Factual Grounding

BERTScore is computed between model responses and ground truth answers for all conditions and domains. Claim recall percentage measures what proportion of ground truth propositions are recoverable from the model response.

### RQ3 — Latent Feature Analysis

Four latent features are extracted during inference: Attention Norm, Attention Std, model Confidence, and Perplexity. Angular distance between feature vectors across prompting conditions is computed and correlated with Win/Loss outcomes. Scripts write correlation matrices and scatter plots to `results/`.

---

## Dependencies

Python 3.9 or later is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `transformers` | Model loading, inference, and latent feature extraction |
| `bert_score` | BERTScore computation for RQ2 factual recall evaluation |
| `SPARQLWrapper` | Executing SPARQL queries during transformation |
| `pandas` | Reading and writing .xlsx dataset files and results tables |
| `matplotlib` / `seaborn` | Generating win/loss plots and latent space visualisations |
| `scipy` | Correlation and angular distance calculations in RQ3 |

---

## Running the Pipeline

All scripts must be run from the repository root. Replace `fiqa` with `legalqa` or `telcoqa` where applicable.

**Step 1 — Transform NLQ to SPARQL**

```bash
python src/transformation/nlq_to_sparql.py --dataset fiqa    --input data/raw/fiqa.xlsx    --output data/raw/fiqa_sparql.xlsx
python src/transformation/nlq_to_sparql.py --dataset legalqa --input data/raw/legalqa.xlsx --output data/raw/legalqa_sparql.xlsx
```

**Step 2 — Run Inference**

```bash
python src/generation/run_inference.py --dataset fiqa --method all --model <model_name>
```

**Step 3 — Parse Responses**

```bash
python src/parsing/generic_response_parser.py --input results/raw/ --output results/parsed/
```

**Step 4 — Run Evaluation**

```bash
python eval/RQ1_KnowledgeDensity/run_rq1.py --domain fiqa
python eval/RQ2_FactualRecall/run_rq2.py    --domain fiqa
python eval/RQ3_LatentSpace/run_rq3.py      --domain fiqa
```

Results are written to `results/plots/` and `results/tables/`.

---

## Notes on Paths

- All scripts use relative paths. The repository root must be the working directory when running any script.
- Windows users: use WSL or a shell that resolves forward slashes. Do not substitute backslashes.
- The `results/` directory must exist before running evaluation scripts. It is included in the repository with a `.gitkeep` file.

---

## Licence

This repository is provided for academic peer review. Reuse of code or data is subject to the terms in the `LICENSE` file in the repository root.
