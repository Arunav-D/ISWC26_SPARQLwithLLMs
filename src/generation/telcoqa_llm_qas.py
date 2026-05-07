"""
telco_neural_pathway_analyzer.py
==========================
Prompt-engineering evaluation harness for knowledge-based Q&A over an <ABC>
telecommunications domain ontology.

Supports four query modes that form a 2×2 factorial design:

    Mode          Input representation    Reasoning scaffold
    ──────────    ───────────────────    ──────────────────
    nlq           Natural-language Q      None (one-shot)
    nlq_cot       Natural-language Q      Chain-of-Thought
    sparql        SPARQL query string     None (one-shot)
    sparql_cot    SPARQL query string     Chain-of-Thought

Input CSV/XLSX columns required:
    Query_ID, Question, SPARQL_Query

Output JSON columns:
    Query_ID, Question, SPARQL_Query, Generated_Response

Resume support
--------------
Incremental checkpointing every 50 queries; interrupted runs pick up from
the last persisted Query_ID automatically (disable with --no-resume).

Prompt-output contract
----------------------
Every response is gated behind an ``ANSWER:`` tag.
Yes/No answers must include an inline explanation; bare Yes/No is rejected.
Token budget: 1 024 new tokens per response.

Supported models
----------------
gemma, mistral, codellama, llama3, qwen, qwen-coder
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
        set_seed,
    )
except ImportError as exc:
    sys.exit(
        f"transformers not found: {exc}\n"
        "Install with:  pip install transformers accelerate bitsandbytes"
    )

try:
    import scipy.stats as stats
    from scipy.spatial.distance import cosine
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    stats = cosine = cosine_similarity = None

set_seed(42)
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Query-mode constants
# ──────────────────────────────────────────────────────────────────────────────

QUERY_MODE_NLQ        = "nlq"
QUERY_MODE_NLQ_COT    = "nlq_cot"
QUERY_MODE_SPARQL     = "sparql"
QUERY_MODE_SPARQL_COT = "sparql_cot"

SUPPORTED_MODES: List[str] = [
    QUERY_MODE_NLQ,
    QUERY_MODE_NLQ_COT,
    QUERY_MODE_SPARQL,
    QUERY_MODE_SPARQL_COT,
]

# Maps each mode to the DataFrame column that supplies the query text.
MODE_SOURCE_COLUMN: Dict[str, str] = {
    QUERY_MODE_NLQ:        "Question",
    QUERY_MODE_NLQ_COT:    "Question",
    QUERY_MODE_SPARQL:     "SPARQL_Query",
    QUERY_MODE_SPARQL_COT: "SPARQL_Query",
}

# Short suffix used in output directory and file names.
MODE_SUFFIX: Dict[str, str] = {
    QUERY_MODE_NLQ:        "NLQ",
    QUERY_MODE_NLQ_COT:    "NLQ_COT",
    QUERY_MODE_SPARQL:     "SPARQL",
    QUERY_MODE_SPARQL_COT: "SPARQL_COT",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Per-model hyperparameters and loading hints."""

    name: str
    path: str
    num_layers: int = 32
    hidden_size: int = 4096
    attention_heads: int = 32
    supports_attention: bool = True
    supports_hidden_states: bool = True
    chat_template: Optional[str] = None
    max_sequence_length: int = 1600
    generation_max_new_tokens: int = 1024
    requires_auth: bool = False
    prefer_4bit: bool = True
    cpu_offload_threshold: float = 0.8
    layer_offload_pattern: str = "sequential"
    minimal_precision: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Numerical utilities
# ──────────────────────────────────────────────────────────────────────────────

class SafeNumericalHandler:
    """Converts arbitrary tensor/array/scalar types to a plain Python float
    without raising on NaN, Inf, or unexpected shapes."""

    SAFE_MAX = 1e8

    @staticmethod
    def safe_float(value: Union[float, "torch.Tensor", "np.ndarray"]) -> float:
        try:
            if isinstance(value, torch.Tensor):
                value = (
                    value.detach().cpu().item()
                    if value.numel() == 1
                    else float(value.detach().cpu().numpy().mean())
                )
            if isinstance(value, np.ndarray):
                value = float(value.item()) if value.size == 1 else float(np.mean(value))
            if isinstance(value, (tuple, list)):
                value = float(value[0]) if len(value) == 1 else float(np.mean(value))
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                return 0.5
            return float(np.clip(value, -SafeNumericalHandler.SAFE_MAX, SafeNumericalHandler.SAFE_MAX))
        except (ValueError, TypeError, AttributeError):
            return 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs model prompts for the four evaluation modes.

    Design rationale
    ----------------
    Each prompt is composed from four reusable blocks:

        _ROLE_BLOCK          — establishes persona and domain
        _GOAL_BLOCK          — states the answering contract
        _OUTPUT_CONSTRAINTS  — enforces structural rules (ANSWER: tag, Yes/No policy)
        _COT_BLOCK           — adds the chain-of-thought scaffold (CoT modes only)
        _SPARQL_FRAMING      — contextualises SPARQL input (SPARQL modes only)

    The 2×2 factorial structure means _COT_BLOCK is byte-for-byte identical in
    ``nlq_cot`` and ``sparql_cot``, keeping the reasoning scaffold constant while
    varying only the input representation.

    Output contract
    ---------------
    Non-CoT builders close with ``ANSWER:`` so the model opens the tag directly.
    CoT builders close with ``THINK:``; the model produces ``ANSWER:`` naturally
    at the end of its reasoning chain as instructed in ``_COT_BLOCK``.
    """

    # ── Shared building blocks ────────────────────────────────────────────── #

    _ROLE_BLOCK = (
        "You are a Customer Service Officer at <ABC>, "
        "proficient in both natural and programming languages."
    )

    _GOAL_BLOCK = (
        "Goal: Provide a factually correct, relevant, and succinct answer to the "
        "customer query. Do NOT repeat the question. Do NOT ask clarifying questions. "
        "If the answer is Yes or No, you MUST immediately follow it with a clear, "
        "concise explanation — never output a bare 'Yes' or 'No' without elaboration. "
        "Do NOT fabricate product names, tariff details, technical specifications, "
        "or service policies. "
        "If the answer is genuinely unknown or cannot be confirmed, say so explicitly "
        "and explain why.\n"
        "Avoid filler, repetition, or any content that does not directly resolve the query."
    )

    _OUTPUT_CONSTRAINTS = (
        "Output rules:\n"
        "- One-shot resolution: answer immediately and completely.\n"
        "- Always begin your final response with the tag ANSWER: on its own line, "
        "followed by your response text.\n"
        "- If your answer starts with Yes or No, append a colon and a clear explanation "
        "on the same line, e.g. 'ANSWER: Yes — <explanation>' or 'ANSWER: No — <explanation>'.\n"
        "- Use plain English; apply telecommunications domain vocabulary only when it "
        "adds precision.\n"
        "- Reference specific products, plans, or prices only when you are certain of "
        "their accuracy — do not guess or approximate service details.\n"
        "- If a product name, tariff, or policy cannot be confirmed with certainty, "
        "explicitly flag this uncertainty rather than presenting it as fact.\n"
        "- Be courteous and customer-friendly.\n"
        "- Never restate or paraphrase the question in your reply.\n"
        "- Never ask follow-up questions.\n"
        "- Never fabricate product offerings, network specifications, pricing, or "
        "service terms."
    )

    _COT_BLOCK = (
        "Reasoning requirement:\n"
        "Before giving your final answer, work through the problem using this structure:\n"
        "  THINK: Identify the core question and relevant domain knowledge.\n"
        "  PLAN: List the steps needed to reach the answer.\n"
        "  REASON: Execute those steps, checking facts at each step.\n"
        "  ANSWER: Begin with the tag ANSWER: then state your final, customer-facing "
        "response. If the answer starts with Yes or No, follow it immediately with an "
        "explanation on the same line. Never fabricate product details, pricing, or "
        "service terms in the ANSWER.\n"
        "Keep THINK/PLAN/REASON concise. Only the ANSWER section is shown to the customer."
    )

    _SPARQL_FRAMING = (
        "The following is a SPARQL query submitted to the <ABC>. "
        "Interpret the query intent and answer as if you had executed it against the "
        "knowledge-based source. Return only the resolved answer in natural language. "
        "Do not fabricate product names, tariff details, network specifications, or "
        "service policies that cannot be confirmed from established <ABC> knowledge sources."
    )

    # ── Public factory ────────────────────────────────────────────────────── #

    @classmethod
    def build(cls, query_text: str, mode: str, query_id: int) -> str:
        """Return a fully assembled prompt string for *mode*."""
        builders = {
            QUERY_MODE_NLQ:        cls._nlq,
            QUERY_MODE_NLQ_COT:    cls._nlq_cot,
            QUERY_MODE_SPARQL:     cls._sparql,
            QUERY_MODE_SPARQL_COT: cls._sparql_cot,
        }
        mode = mode.lower()
        if mode not in builders:
            raise ValueError(f"Unknown mode '{mode}'. Valid options: {SUPPORTED_MODES}")
        return builders[mode](query_text, query_id)

    # ── Private builders ──────────────────────────────────────────────────── #

    @classmethod
    def _nlq(cls, question: str, query_id: int) -> str:
        return "\n".join([
            cls._ROLE_BLOCK, "",
            cls._GOAL_BLOCK, "",
            cls._OUTPUT_CONSTRAINTS, "",
            f"[Query ID: {query_id}]",
            f"Customer Query: {question}", "",
            "ANSWER:",
        ])

    @classmethod
    def _nlq_cot(cls, question: str, query_id: int) -> str:
        return "\n".join([
            cls._ROLE_BLOCK, "",
            cls._GOAL_BLOCK, "",
            cls._OUTPUT_CONSTRAINTS, "",
            cls._COT_BLOCK, "",
            f"[Query ID: {query_id}]",
            f"Customer Query: {question}", "",
            "THINK:",
        ])

    @classmethod
    def _sparql(cls, sparql_query: str, query_id: int) -> str:
        return "\n".join([
            cls._ROLE_BLOCK, "",
            cls._GOAL_BLOCK, "",
            cls._OUTPUT_CONSTRAINTS, "",
            cls._SPARQL_FRAMING, "",
            f"[Query ID: {query_id}]",
            f"SPARQL Query:\n{sparql_query}", "",
            "ANSWER:",
        ])

    @classmethod
    def _sparql_cot(cls, sparql_query: str, query_id: int) -> str:
        return "\n".join([
            cls._ROLE_BLOCK, "",
            cls._GOAL_BLOCK, "",
            cls._OUTPUT_CONSTRAINTS, "",
            cls._SPARQL_FRAMING, "",
            cls._COT_BLOCK, "",
            f"[Query ID: {query_id}]",
            f"SPARQL Query:\n{sparql_query}", "",
            "THINK:",
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Model manager
# ──────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Loads and manages a single HuggingFace causal LM with progressive
    quantisation fallback:

        1. 4-bit NF4 quantisation  (preferred; lowest VRAM)
        2. 8-bit quantisation      (fallback)
        3. CPU fp32                (last resort)

    The ``format_prompt`` method applies the model's chat template after the
    task-level prompt has been built by ``PromptBuilder``.
    """

    CONFIGS: Dict[str, ModelConfig] = {
        "gemma": ModelConfig(
            name="gemma",
            path="google/gemma-7b-it",
            num_layers=18, hidden_size=2048, attention_heads=8,
            max_sequence_length=1800, generation_max_new_tokens=1024,
            requires_auth=True, prefer_4bit=True, cpu_offload_threshold=0.7,
            chat_template=(
                "<start_of_turn>user\n{prompt}<end_of_turn>\n"
                "<start_of_turn>model\n"
            ),
        ),
        "mistral": ModelConfig(
            name="mistral",
            path="mistralai/Mistral-7B-Instruct-v0.3",
            num_layers=32, hidden_size=4096, attention_heads=32,
            max_sequence_length=1800, generation_max_new_tokens=1024,
            requires_auth=False, prefer_4bit=True, cpu_offload_threshold=0.25,
            chat_template="<s>[INST] {prompt} [/INST]",
        ),
        "codellama": ModelConfig(
            name="codellama",
            path="codellama/CodeLlama-7b-Instruct-hf",
            num_layers=32, hidden_size=4096, attention_heads=32,
            max_sequence_length=1800, generation_max_new_tokens=1024,
            requires_auth=False, prefer_4bit=True, cpu_offload_threshold=0.25,
            chat_template="[INST] {prompt} [/INST]",
        ),
        "llama3": ModelConfig(
            name="llama3",
            path="meta-llama/Llama-3.1-8B-Instruct",
            num_layers=36, hidden_size=4096, attention_heads=32,
            max_sequence_length=1800, generation_max_new_tokens=1024,
            requires_auth=True, prefer_4bit=False, cpu_offload_threshold=0.8,
            chat_template=(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            ),
        ),
        "qwen": ModelConfig(
            name="qwen",
            path="Qwen/Qwen2.5-7B-Instruct",
            num_layers=36, hidden_size=4096, attention_heads=32,
            max_sequence_length=1800, generation_max_new_tokens=1024,
            requires_auth=False, prefer_4bit=True, cpu_offload_threshold=0.25,
            chat_template=(
                "<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        ),
        "qwen-coder": ModelConfig(
            name="qwen-coder",
            path="Qwen/Qwen2.5-Coder-7B-Instruct",
            num_layers=36, hidden_size=4096, attention_heads=32,
            max_sequence_length=1800, generation_max_new_tokens=1024,
            requires_auth=False, prefer_4bit=True, cpu_offload_threshold=0.25,
            chat_template=(
                "<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        ),
    }

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        cache_dir: str = "./model_cache/",
    ) -> None:
        self.model_name = model_name.lower()
        if self.model_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.CONFIGS.keys())}"
            )
        self.config   = self.CONFIGS[self.model_name]
        self.device   = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.hf_token  = hf_token or os.getenv("HF_TOKEN")
        self.cache_dir = cache_dir
        self.model:     Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    # ── Loading ───────────────────────────────────────────────────────────── #

    def load(self) -> None:
        """Attempt quantisation strategies in order; raise if all fail."""
        logger.info(f"Loading '{self.config.name}' …")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.environ.update({
            "HF_HOME":               self.cache_dir,
            "HF_DATASETS_CACHE":     self.cache_dir,
            "HUGGINGFACE_HUB_CACHE": self.cache_dir,
        })
        os.environ.pop("TRANSFORMERS_CACHE", None)

        auth: Dict = {}
        if self.config.requires_auth:
            if self.hf_token:
                auth["token"] = self.hf_token
            else:
                logger.warning(
                    f"Model '{self.config.name}' requires a HuggingFace token "
                    "(--hf_token or HF_TOKEN env var)."
                )

        self._load_tokenizer(auth)

        strategies = [
            ("4-bit NF4 quantisation", self._load_4bit),
            ("8-bit quantisation",     self._load_8bit),
            ("CPU fp32",               self._load_cpu),
        ]
        for label, fn in strategies:
            logger.info(f"  Trying: {label}")
            try:
                self.model = fn(auth)
                if self.model is not None:
                    logger.info(f"  Loaded with: {label}")
                    self._post_load()
                    return
            except Exception as exc:
                logger.warning(f"  {label} failed: {exc}")
                self.model = None
                torch.cuda.empty_cache()
                gc.collect()

        raise RuntimeError(f"All loading strategies failed for '{self.config.name}'.")

    def _load_tokenizer(self, auth: Dict) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
            model_max_length=self.config.max_sequence_length,
            **auth,
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def _load_4bit(self, auth: Dict) -> Any:
        return AutoModelForCausalLM.from_pretrained(
            self.config.path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map="auto",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **auth,
        )

    def _load_8bit(self, auth: Dict) -> Any:
        return AutoModelForCausalLM.from_pretrained(
            self.config.path,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=True,
            ),
            device_map="auto",
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **auth,
        )

    def _load_cpu(self, auth: Dict) -> Any:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
            **auth,
        )
        torch.set_num_threads(min(8, psutil.cpu_count()))
        model.config.use_cache            = False
        model.config.output_attentions    = False
        model.config.output_hidden_states = False
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _post_load(self) -> None:
        try:
            self.model.eval()
            self.model.config.use_cache = False
            for p in self.model.parameters():
                p.requires_grad = False
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:
            logger.warning(f"Post-load optimisation step failed: {exc}")

    # ── Prompt formatting ─────────────────────────────────────────────────── #

    def format_prompt(self, query_text: str, mode: str, query_id: int) -> str:
        """Wrap the task prompt in the model's chat template."""
        core = PromptBuilder.build(
            query_text=query_text,
            mode=mode,
            query_id=query_id,
        )
        return self.config.chat_template.format(prompt=core) if self.config.chat_template else core

    # ── Cleanup ───────────────────────────────────────────────────────────── #

    def cleanup(self) -> None:
        try:
            del self.model, self.tokenizer
            self.model = self.tokenizer = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Model resources released.")
        except Exception as exc:
            logger.warning(f"Cleanup encountered an error: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────────

class InferenceRunner:
    """Wraps generation config creation and the inference context manager."""

    def __init__(self, model: Any, tokenizer: Any, config: ModelConfig) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config

    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.config.generation_max_new_tokens,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.3,
            length_penalty=1.0,
            num_beams=1,
        )

    @contextmanager
    def inference_ctx(self):
        """Temporarily disable KV-cache and auxiliary outputs for speed."""
        saved = {
            "use_cache":            getattr(self.model.config, "use_cache",            None),
            "output_attentions":    getattr(self.model.config, "output_attentions",    None),
            "output_hidden_states": getattr(self.model.config, "output_hidden_states", None),
        }
        try:
            self.model.config.use_cache            = False
            self.model.config.output_attentions    = False
            self.model.config.output_hidden_states = False
            torch.set_num_threads(min(4, psutil.cpu_count()))
            yield
        finally:
            for k, v in saved.items():
                if v is not None:
                    setattr(self.model.config, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# Query processor
# ──────────────────────────────────────────────────────────────────────────────

class QueryProcessor:
    """Handles inference for a single dataset row."""

    def __init__(self, manager: ModelManager, query_mode: str) -> None:
        self.manager    = manager
        self.model      = manager.model
        self.tokenizer  = manager.tokenizer
        self.config     = manager.config
        self.device     = manager.device
        self.mode       = query_mode
        self.safe       = SafeNumericalHandler()
        self.runner     = InferenceRunner(self.model, self.tokenizer, self.config)

    def process(
        self,
        query_id:   int,
        question:   str,
        sparql:     str,
    ) -> Dict[str, Any]:
        source_col = MODE_SOURCE_COLUMN[self.mode]
        query_text = question if source_col == "Question" else sparql

        if not query_text.strip():
            return self._error_record(
                query_id, question, sparql,
                f"Empty source column '{source_col}' for mode '{self.mode}'.",
            )

        try:
            result = self._run(query_id, query_text)
            result.update({
                "Query_ID":     query_id,
                "Question":     question,
                "SPARQL_Query": sparql,
                "query_mode":   self.mode,
            })
            return result
        except Exception as exc:
            logger.error(f"Query {query_id} failed: {exc}")
            return self._error_record(query_id, question, sparql, str(exc))

    def _run(self, query_id: int, query_text: str) -> Dict[str, Any]:
        for attempt in range(2):
            try:
                return self._infer(query_id, query_text)
            except Exception as exc:
                if attempt == 1:
                    raise
                logger.warning(f"Retry {query_id} after: {exc}")
                torch.cuda.empty_cache()
                time.sleep(0.5)

    def _infer(self, query_id: int, query_text: str) -> Dict[str, Any]:
        prompt = self.manager.format_prompt(
            query_text=query_text,
            mode=self.mode,
            query_id=query_id,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(self.config.max_sequence_length, 1600),
        )
        inputs       = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with self.runner.inference_ctx():
            with torch.no_grad():
                t0      = time.time()
                outputs = self.model.generate(**inputs, generation_config=self.runner.generation_config())
                elapsed = time.time() - t0

        new_ids = outputs.sequences[0][input_length:]
        text    = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        text    = self._clean_degenerate(text)

        return {
            "Generated_Response":      text,
            "confidence_score":        self._confidence(outputs),
            "input_length":            input_length,
            "generated_length":        len(new_ids),
            "generation_time_seconds": elapsed,
            "processing_timestamp":    datetime.now().isoformat(),
        }

    def _confidence(self, outputs: Any) -> float:
        try:
            if hasattr(outputs, "scores") and outputs.scores:
                scores = outputs.scores[-min(3, len(outputs.scores)):]
                vals   = []
                for s in scores:
                    if isinstance(s, torch.Tensor):
                        probs = F.softmax(s, dim=-1)
                        vals.append(self.safe.safe_float(torch.max(probs, dim=-1)[0]))
                return self.safe.safe_float(np.mean(vals)) if vals else 0.7
        except Exception:
            pass
        return 0.7

    @staticmethod
    def _clean_degenerate(text: str) -> str:
        if not text:
            return text
        lines = text.split("\n")
        if len(lines) > 10:
            counts = Counter(l.strip() for l in lines if l.strip())
            top_line, top_count = counts.most_common(1)[0]
            if top_count > 10:
                logger.warning(f"Degenerate output: line '{top_line}' repeated {top_count}×")
                return f"DEGENERATE_OUTPUT: repeated token '{top_line}' ×{top_count}"
        words = text.split()
        if len(words) > 10:
            for token, grp in groupby(words):
                count = sum(1 for _ in grp)
                if count > 10:
                    logger.warning(f"Degenerate output: word '{token}' repeated {count}×")
                    return f"DEGENERATE_OUTPUT: repeated token '{token}' ×{count}"
        return text

    @staticmethod
    def _error_record(
        query_id: int,
        question: str,
        sparql:   str,
        error:    str,
    ) -> Dict[str, Any]:
        return {
            "Query_ID":                query_id,
            "Question":                question,
            "SPARQL_Query":            sparql,
            "Generated_Response":      f"ERROR: {error}",
            "confidence_score":        0.0,
            "input_length":            0,
            "generated_length":        0,
            "generation_time_seconds": 0.0,
            "processing_timestamp":    datetime.now().isoformat(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main analyser
# ──────────────────────────────────────────────────────────────────────────────

class NeuralPathwayAnalyzer:
    """
    Orchestrates data loading, model inference, checkpointing, and output.

    Output layout
    -------------
    <input_dir>/<model>_<MODE>/
        <model>_<MODE>.json     — full results with metadata
    """

    def __init__(
        self,
        model_name:      str,
        query_mode:      str,
        device:          Optional[str] = None,
        hf_token:        Optional[str] = None,
        cache_dir:       str           = "./model_cache/",
        input_file_path: Optional[str] = None,
        resume:          bool          = True,
    ) -> None:
        self.model_name      = model_name.lower()
        self.query_mode      = query_mode.lower()
        self.hf_token        = hf_token
        self.cache_dir       = cache_dir
        self.resume          = resume
        self.output_suffix   = MODE_SUFFIX[self.query_mode]

        if input_file_path:
            self.output_dir = Path(input_file_path).parent / f"{self.model_name}_{self.output_suffix}"
        else:
            self.output_dir = Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialising | model={model_name} | mode={query_mode} | "
            f"output={self.output_dir} | resume={resume}"
        )
        self._init_model(device)

    @property
    def output_file(self) -> Path:
        return self.output_dir / f"{self.model_name}_{self.output_suffix}.json"

    def _init_model(self, device: Optional[str]) -> None:
        self.manager = ModelManager(
            model_name=self.model_name,
            device=device,
            hf_token=self.hf_token,
            cache_dir=self.cache_dir,
        )
        self.manager.load()
        self.processor = QueryProcessor(self.manager, self.query_mode)
        logger.info(f"Model '{self.model_name}' ready.")

    # ── Resume helpers ────────────────────────────────────────────────────── #

    def _load_existing(self) -> Tuple[List[Dict[str, Any]], Set[int]]:
        if not self.output_file.exists():
            logger.info("No prior results — starting fresh.")
            return [], set()
        try:
            with open(self.output_file, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("results", [])
            ids      = {r["Query_ID"] for r in existing if "Query_ID" in r}
            logger.info(f"Loaded {len(existing)} prior results (IDs {min(ids)}–{max(ids)}).")
            return existing, ids
        except json.JSONDecodeError as exc:
            logger.warning(f"Corrupt results file ({exc}); starting fresh.")
            import shutil
            shutil.copy2(self.output_file, self.output_file.with_suffix(".json.backup"))
            return [], set()
        except Exception as exc:
            logger.warning(f"Could not read prior results: {exc}")
            return [], set()

    # ── Data loading ──────────────────────────────────────────────────────── #

    def load_data(self, path: str) -> pd.DataFrame:
        logger.info(f"Reading input: {path}")
        if path.endswith(".xlsx"):
            df = pd.read_excel(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("Input must be .xlsx or .csv.")

        df.columns = [c.strip() for c in df.columns]
        required   = ["Query_ID", "Question", "SPARQL_Query"]
        missing    = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

        df = df.fillna("")
        logger.info(f"Loaded {len(df)} rows.")
        return df

    # ── Processing loop ───────────────────────────────────────────────────── #

    def run(
        self,
        df:         pd.DataFrame,
        max_queries: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        if self.resume:
            results, done_ids = self._load_existing()
            pending           = df[~df["Query_ID"].isin(done_ids)].copy()
            if pending.empty:
                logger.info("All queries already processed.")
                return results
            logger.info(
                f"Resume | total={len(df)} done={len(done_ids)} remaining={len(pending)}"
            )
        else:
            results   = []
            done_ids  = set()
            pending   = df.copy()

        n_prior  = len(results)
        to_do    = pending.head(max_queries) if max_queries else pending
        n_todo   = len(to_do)
        logger.info(f"Processing {n_todo} queries this session.")

        for idx, (_, row) in enumerate(to_do.iterrows()):
            qid  = int(row["Query_ID"])
            q    = str(row["Question"]).strip()
            spq  = str(row["SPARQL_Query"]).strip()
            logger.info(f"[{idx+1}/{n_todo}] Query_ID={qid}")

            try:
                rec = self.processor.process(query_id=qid, question=q, sparql=spq)
            except Exception as exc:
                logger.error(f"Unhandled error for Query_ID={qid}: {exc}")
                logger.debug(traceback.format_exc())
                rec = QueryProcessor._error_record(qid, q, spq, str(exc))

            results.append(rec)
            logger.info(f"  → {rec.get('Generated_Response','')[:80].replace(chr(10),' ')} …")

            if (idx + 1) % 50 == 0:
                self._save(results)
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"Checkpoint at {idx+1}/{n_todo}.")

        if len(results) > n_prior or not self.resume:
            self._save(results)

        logger.info(f"Session complete. Total records: {len(results)}.")
        return results

    # ── Persistence ───────────────────────────────────────────────────────── #

    def _save(self, results: List[Dict[str, Any]]) -> None:
        ok     = [r for r in results if not str(r.get("Generated_Response", "")).startswith("ERROR")]
        errors = len(results) - len(ok)

        payload = {
            "metadata": {
                "model":              self.model_name,
                "mode":               self.query_mode,
                "suffix":             self.output_suffix,
                "timestamp":          datetime.now().isoformat(),
                "total":              len(results),
                "successful":         len(ok),
                "errors":             errors,
                "resume":             self.resume,
                "output_columns":     ["Query_ID", "Question", "SPARQL_Query", "Generated_Response"],
            },
            "results": results,
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(results)} records → {self.output_file}")

    def cleanup(self) -> None:
        self.manager.cleanup()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Neural Pathway Analyzer — prompt-engineering harness for <ABC> Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Query modes (2×2 factorial design)
───────────────────────────────────
  nlq          Natural-language question   ·  no reasoning scaffold
  nlq_cot      Natural-language question   ·  Chain-of-Thought
  sparql        SPARQL query string         ·  no reasoning scaffold
  sparql_cot    SPARQL query string         ·  Chain-of-Thought

Input source column per mode
─────────────────────────────
  nlq / nlq_cot    → Question
  sparql / sparql_cot → SPARQL_Query

Output
──────
  <input_dir>/<model>_<MODE>/<model>_<MODE>.json

Resume mode (default ON)
─────────────────────────
  Interrupted runs resume from the last saved Query_ID.
  Use --no-resume to re-process everything from scratch.

Examples
────────
  python telco_neural_pathway_analyzer.py --input queries.xlsx --model mistral    --mode nlq
  python telco_neural_pathway_analyzer.py --input queries.xlsx --model llama3     --mode nlq_cot
  python telco_neural_pathway_analyzer.py --input queries.xlsx --model qwen-coder --mode sparql
  python telco_neural_pathway_analyzer.py --input queries.xlsx --model qwen-coder --mode sparql_cot
  python telco_neural_pathway_analyzer.py --input queries.xlsx --model gemma      --mode nlq --no-resume
        """,
    )
    parser.add_argument("--input",       required=True,
                        help="Input .xlsx or .csv (columns: Query_ID, Question, SPARQL_Query)")
    parser.add_argument("--model",       required=True,
                        choices=list(ModelManager.CONFIGS.keys()),
                        help="Model identifier")
    parser.add_argument("--mode",        required=True, choices=SUPPORTED_MODES,
                        help="Query mode")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (default: cuda if available, else cpu)")
    parser.add_argument("--hf_token",    default=None,
                        help="HuggingFace token (required for gated models)")
    parser.add_argument("--cache_dir",   default="./model_cache/",
                        help="Directory for cached model weights")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Cap on new queries to process this session")
    parser.add_argument("--no-resume",   dest="resume", action="store_false",
                        help="Disable resume — reprocess all from scratch")
    parser.set_defaults(resume=True)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        return 1

    print("=" * 72)
    print("NEURAL PATHWAY ANALYZER")
    print("=" * 72)
    print(f"  Input      : {args.input}")
    print(f"  Model      : {args.model}")
    print(f"  Mode       : {args.mode}")
    print(f"  Source col : {MODE_SOURCE_COLUMN[args.mode]}")
    print(f"  Device     : {args.device}")
    print(f"  Resume     : {'ON' if args.resume else 'OFF'}")
    print(f"  Max queries: {args.max_queries or 'all'}")
    print(f"  Max tokens : 1024 per response")
    print("=" * 72)

    try:
        analyzer = NeuralPathwayAnalyzer(
            model_name=args.model,
            query_mode=args.mode,
            device=args.device,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            input_file_path=args.input,
            resume=args.resume,
        )
        df = analyzer.load_data(args.input)
        if df.empty:
            print("Error: no rows found in input file.")
            return 1

        t0      = datetime.now()
        results = analyzer.run(df, max_queries=args.max_queries)
        elapsed = datetime.now() - t0

        ok     = [r for r in results if not str(r.get("Generated_Response", "")).startswith("ERROR")]
        errors = len(results) - len(ok)

        print("\n" + "=" * 72)
        print("COMPLETE")
        print("=" * 72)
        print(f"  Total      : {len(results)}")
        print(f"  Successful : {len(ok)}")
        print(f"  Errors     : {errors}")
        print(f"  Time       : {elapsed}")
        print(f"  Output     : {analyzer.output_file}")
        print("=" * 72)

        analyzer.cleanup()
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())