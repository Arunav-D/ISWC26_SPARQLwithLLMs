"""
pipeline2a_triple_extraction.py
================================
QAS Evaluation Framework, Module B (Part A)
Knowledge Graph Triple Extraction: Rule-Based + spaCy

Overview
--------
Implements and evaluates two lightweight, GPU-free triple-extraction strategies
against model-generated answers from a telecom-domain question-answering system.

Extractor A — Rule-Based (domain ontology)
    A five-pass dependency-arc matcher built on top of a telecom-domain ontology.
    Each pass targets a distinct linguistic pattern:
        Pass 1  nsubj / dobj / pobj       →  Class / Instance       (NOUN_ARG)
        Pass 2  ROOT / xcomp / ccomp      →  Object property        (VERB_PROP)
        Pass 3  amod / nummod / advmod    →  Data property           (MOD_DATA)
        Pass 4  cc / conj                 →  Logical operator        (CONJ_LOGIC)
        Pass 5  prep + pobj               →  Mereological relation   (PREP_MERO)
    High precision on in-domain vocabulary; recall degrades for unseen predicates.

Extractor B — spaCy General NLP
    Pure dependency-parse triples without ontology mapping.
    Predicate URIs are raw lemma strings; subject/object spans are noun heads.
    Maximises recall at the cost of predicate normalisation.

Evaluation Metrics (TKF subset)
--------------------------------
    KD      Knowledge Density   =  typed triples (conf ≥ 0.75) / total tokens
    TKF     Composite score     =  KD only (W_KD = 1.0 in this module)
    triple_count_c   Candidate triple count (extractor output)
    triple_count_g   Gold triple count (rule-based on ground truth)

    Note: GED, SF, PAR, MA and predicate-level metrics are computed in
    pipeline2b_openie_rebel.py (Extractors C and D) for the full TKF composite.

Output
------
    pipeline2a_output.xlsx
        Pivot sheets  —  per query_id × extractor, for each key metric
        Detail sheets —  row-level results per model × extractor
        Triples sheet —  all extracted triples with provenance
        Leaderboard   —  aggregated TKF ranking across all models and methods
        Summary sheet —  per-model aggregated scores

Dependencies
------------
    pip install spacy numpy pandas openpyxl
    python -m spacy download en_core_web_sm

Companion script
----------------
    pipeline2b_openie_rebel.py  —  Extractors C (CoreNLP OpenIE) and D (REBEL)
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Input/output paths are resolved at runtime from CLI arguments (see main()).
# The constants below control sheet/column naming conventions and NLP settings;
# change them here if your data uses different column headers.

GROUND_TRUTH_SHEET = "ground_truth"
RESPONSE_SUFFIX    = "_response"
SKIP_SHEETS: List[str] = [GROUND_TRUTH_SHEET]

# Default filenames used when the CLI receives a directory rather than a file
DEFAULT_INPUT_FILENAME  = "ALL_MODELS_responses.xlsx"
DEFAULT_OUTPUT_FILENAME = "pipeline2a_output.xlsx"

NLP_MODEL = "en_core_web_sm"   # Upgrade to en_core_web_trf for publication runs

# TKF composite — this module uses KD only; full composite assembled in pipeline2b
W_KD: float = 1.0

# Minimum confidence threshold for a triple to count as "typed" in KD
KD_CONFIDENCE_THRESHOLD: float = 0.75

# ---------------------------------------------------------------------------
# spaCy singleton loader
# ---------------------------------------------------------------------------

_NLP: Optional[spacy.language.Language] = None


def get_nlp() -> spacy.language.Language:
    """Load and cache the spaCy pipeline (loaded once per process)."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(NLP_MODEL)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{NLP_MODEL}' not found.\n"
                f"Run:  python -m spacy download {NLP_MODEL}"
            )
    return _NLP


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or *default* when denominator is zero."""
    return numerator / denominator if denominator else default


def _to_camel_case(surface: str) -> str:
    """Convert a whitespace-separated surface string to lowerCamelCase."""
    parts = re.sub(r"[^a-zA-Z0-9\s]", " ", str(surface)).split()
    if not parts:
        return "x"
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


# ---------------------------------------------------------------------------
# Telecom-domain ontology
# ---------------------------------------------------------------------------
# Deliberately broad to cover multiple UK ISPs and regulators; no single
# provider should be identifiable from vocabulary alone.

TC_CLASSES: Dict[str, str] = {
    # --- Hardware ---
    "smart hub plus":    "broadband:SmartHubPlus",
    "smart hub 2":       "broadband:SmartHub2",
    "smart hub":         "broadband:SmartHub",
    "super hub":         "broadband:SuperHub",
    "wifi hub":          "broadband:WiFiHub",
    "wifi disc":         "broadband:WiFiDisc",
    "wifi extender":     "broadband:WiFiExtender",
    "router":            "broadband:Router",
    "hub":               "broadband:Hub",
    "modem":             "broadband:Modem",
    "ont":               "broadband:ONT",
    # --- Services ---
    "1.6gbps broadband": "broadband:Broadband1_6Gbps",
    "broadband":         "broadband:Broadband",
    "internet":          "broadband:InternetService",
    "wifi":              "broadband:WiFi",
    "fibre":             "broadband:FiberService",
    "fttp":              "broadband:FTTP",
    "fttc":              "broadband:FTTC",
    "guest network":     "broadband:GuestNetwork",
    "mesh network":      "broadband:MeshNetwork",
    "digital voice":     "landline:DigitalVoice",
    "phone service":     "landline:PhoneService",
    "landline":          "landline:Landline",
    "voip":              "landline:VoIP",
    "voicemail":         "landline:Voicemail",
    "phone":             "landline:Phone",
    # --- Providers / regulators ---
    "<abc>":             "broadband:ProviderABC",
    "openreach":         "broadband:Openreach",
    "ofcom":             "broadband:Ofcom",
    "sky":               "broadband:Sky",
    "virgin media":      "broadband:VirginMedia",
    "virgin":            "broadband:VirginMedia",
    "talktalk":          "broadband:TalkTalk",
    "vodafone":          "broadband:Vodafone",
    "o2":                "broadband:O2",
    "three":             "broadband:Three",
    "plusnet":           "broadband:Plusnet",
    "now broadband":     "broadband:NowBroadband",
    # --- Technical concepts ---
    "firmware":          "broadband:Firmware",
    "ethernet":          "broadband:Ethernet",
    "ipv6":              "broadband:IPv6",
    "ipv4":              "broadband:IPv4",
    "5ghz":              "broadband:Band5GHz",
    "2.4ghz":            "broadband:Band2_4GHz",
    "dns":               "broadband:DNS",
    "cgnat":             "broadband:CGNAT",
    "signal":            "broadband:Signal",
    "bandwidth":         "broadband:Bandwidth",
    "port forwarding":   "broadband:PortForwarding",
    "dmz":               "broadband:DMZ",
    # --- Devices ---
    "smart tv":          "broadband:SmartTV",
    "laptop":            "broadband:Laptop",
    "mobile":            "broadband:Mobile",
    "tablet":            "broadband:Tablet",
    "printer":           "broadband:Printer",
    "xbox":              "broadband:Xbox",
    "ps5":               "broadband:PS5",
    "pc":                "broadband:PC",
    # --- Billing / contract ---
    "termination fee":   "broadband:TerminationFee",
    "direct debit":      "broadband:DirectDebit",
    "contract":          "broadband:Contract",
    "payment":           "broadband:Payment",
    "account":           "broadband:Account",
    "bill":              "broadband:Bill",
    "plan":              "broadband:ServicePlan",
    "refund":            "broadband:Refund",
    # --- Faults ---
    "outage":            "broadband:Outage",
    "fault":             "broadband:Fault",
    "error":             "broadband:Error",
    "issue":             "broadband:Issue",
    "problem":           "broadband:Problem",
    "downtime":          "broadband:Downtime",
}

TC_OBJ_PROPS: Dict[str, str] = {
    "requires":          "broadband:requires",
    "needs":             "broadband:requires",
    "depends on":        "broadband:dependsOn",
    "supports":          "broadband:supports",
    "provides":          "broadband:provides",
    "offers":            "broadband:offers",
    "connects to":       "broadband:connectsTo",
    "connected to":      "broadband:connectedTo",
    "enables":           "broadband:enables",
    "disables":          "broadband:disables",
    "causes":            "broadband:hasCause",
    "affects":           "broadband:affects",
    "uses":              "broadband:uses",
    "includes":          "broadband:includes",
    "replaces":          "broadband:replaces",
    "cancels":           "broadband:cancels",
    "upgrades to":       "broadband:upgradesTo",
    "transfers to":      "broadband:transfersTo",
    "works with":        "broadband:isCompatibleWith",
    "compatible with":   "broadband:isCompatibleWith",
    "incompatible with": "broadband:isIncompatibleWith",
    "can":               "broadband:supports",
    "cannot":            "broadband:doesNotSupport",
    "makes call":        "landline:makesCall",
    "receives call":     "landline:receivesCall",
    "diverts to":        "landline:divertsTo",
    "forwards to":       "landline:forwardsTo",
    "regulated by":      "broadband:regulatedBy",
    "monitored by":      "broadband:monitoredBy",
}

TC_DATA_PROPS: Dict[str, str] = {
    "firmware version":  "broadband:hasFirmwareVersion",
    "ip address":        "broadband:hasIPAddress",
    "mac address":       "broadband:hasMACAddress",
    "password":          "broadband:hasPassword",
    "channel":           "broadband:hasChannel",
    "frequency":         "broadband:hasFrequency",
    "status":            "broadband:hasStatus",
    "speed":             "broadband:hasSpeed",
    "price":             "broadband:hasPrice",
    "cost":              "broadband:hasPrice",
    "fee":               "broadband:hasFee",
    "ssid":              "broadband:hasSSID",
    "port":              "broadband:hasPort",
    "contract end":      "broadband:hasContractEndDate",
    "expiry":            "broadband:hasExpiryDate",
    "colour":            "broadband:hasStatusColor",
    "color":             "broadband:hasStatusColor",
    "light":             "broadband:hasStatusLight",
    "pin":               "landline:hasPIN",
    "ring time":         "landline:hasRingTime",
    "version":           "broadband:hasVersion",
    "model":             "broadband:hasModel",
    "number":            "broadband:hasNumber",
    "date":              "broadband:hasDate",
    "time":              "broadband:hasTime",
    "quantity":          "broadband:hasQuantity",
    "priority":          "broadband:hasPriority",
}

TC_MEREOLOGY: Dict[str, str] = {
    "part of":       "broadband:partOf",
    "located in":    "broadband:locatedIn",
    "inside":        "broadband:locatedIn",
    "within":        "broadband:locatedIn",
    "contained in":  "broadband:containedIn",
    "belongs to":    "broadband:belongsTo",
    "component of":  "broadband:componentOf",
    "section of":    "broadband:sectionOf",
    "subset of":     "broadband:subsetOf",
    "attached to":   "broadband:attachedTo",
    "depends on":    "broadband:dependsOn",
    "included in":   "broadband:includedIn",
}

TC_LOGICAL: Dict[str, str] = {
    "and":    "owl:intersectionOf",
    "both":   "owl:intersectionOf",
    "or":     "owl:unionOf",
    "either": "owl:unionOf",
    "not":    "owl:complementOf",
    "except": "owl:complementOf",
    "unless": "owl:complementOf",
}

# ---------------------------------------------------------------------------
# Triple data class
# ---------------------------------------------------------------------------

class DepType(Enum):
    NOUN_ARG   = auto()
    VERB_PROP  = auto()
    MOD_DATA   = auto()
    CONJ_LOGIC = auto()
    PREP_MERO  = auto()
    GENERAL    = auto()


@dataclass
class Triple:
    """Represents a single subject–predicate–object triple with provenance."""

    subject:    str
    predicate:  str
    object:     str
    subj_uri:   str
    pred_uri:   str
    obj_uri:    str
    dep_type:   DepType = DepType.VERB_PROP
    negated:    bool    = False
    confidence: float   = 0.8
    data_value: str     = ""
    logical_op: str     = ""
    source:     str     = ""
    extractor:  str     = ""

    def key(self) -> str:
        """Unique string identifier for deduplication (null-byte delimited)."""
        return f"{self.subj_uri}\x00{self.pred_uri}\x00{self.obj_uri}"

    def to_turtle(self) -> str:
        """Return a compact Turtle-style serialisation of this triple."""
        neg = " [NEG]" if self.negated else ""
        dv  = f' "{self.data_value}"' if self.data_value else ""
        return f"{self.subj_uri}  {self.pred_uri}  {self.obj_uri}{dv} .{neg}"

    def as_dict(self) -> Dict:
        return {
            "subject":       self.subject,
            "predicate":     self.predicate,
            "object":        self.object,
            "subject_uri":   self.subj_uri,
            "predicate_uri": self.pred_uri,
            "object_uri":    self.obj_uri,
            "dep_type":      self.dep_type.name,
            "negated":       self.negated,
            "confidence":    self.confidence,
            "data_value":    self.data_value,
            "logical_op":    self.logical_op,
            "extractor":     self.extractor,
            "source":        self.source[:120],
            "turtle":        self.to_turtle(),
        }


# ---------------------------------------------------------------------------
# Ontology resolution helpers (Extractor A)
# ---------------------------------------------------------------------------

def _resolve_entity(surface: str) -> Tuple[str, str, float]:
    """
    Map a surface-form string to its closest ontology class URI.

    Matching is longest-first against TC_CLASSES to avoid spurious partial
    hits (e.g. "hub" should not match before "smart hub").  Returns a
    confidence score: 0.95 (exact phrase), 0.75 (word-level), 0.50 (fallback).
    """
    normalised = re.sub(r"\s+", " ", str(surface).lower().strip())
    # Longest-match phrase lookup
    for term in sorted(TC_CLASSES, key=len, reverse=True):
        if term in normalised:
            return term, TC_CLASSES[term], 0.95
    # Single-token fuzzy lookup
    for word in normalised.split():
        for term, uri in TC_CLASSES.items():
            if word == term or (len(word) > 4 and word in term):
                return word, uri, 0.75
    # Fallback: coin a new URI from the surface form
    slug = _to_camel_case(surface[:24])
    return surface.strip()[:40], f"broadband:{slug}", 0.50


def _resolve_obj_prop(surface: str, negated: bool) -> Tuple[str, str]:
    """Map a verb lemma to its object-property URI, applying negation if needed."""
    normalised = str(surface).lower().strip()
    for term in sorted(TC_OBJ_PROPS, key=len, reverse=True):
        if term in normalised:
            uri = TC_OBJ_PROPS[term]
            if negated:
                ns, loc = uri.split(":")
                uri = f"{ns}:doesNot{loc[0].upper()}{loc[1:]}"
            return term, uri
    slug = _to_camel_case(surface[:20])
    return surface[:30], f"broadband:{slug}"


def _resolve_data_prop(surface: str) -> Tuple[str, str]:
    """Map a modifier token to its data-property URI."""
    normalised = str(surface).lower().strip()
    for term in sorted(TC_DATA_PROPS, key=len, reverse=True):
        if term in normalised:
            return term, TC_DATA_PROPS[term]
    slug = _to_camel_case(surface[:20]).capitalize()
    return surface[:30], f"broadband:has{slug}"


def _resolve_mero_prop(surface: str) -> Tuple[str, str]:
    """Map a preposition to its mereological property URI."""
    normalised = str(surface).lower().strip()
    for term in sorted(TC_MEREOLOGY, key=len, reverse=True):
        if term in normalised:
            return term, TC_MEREOLOGY[term]
    return surface[:30], "broadband:partOf"


def _token_is_negated(token: spacy.tokens.Token) -> bool:
    """Return True if *token* has a negation child dependency."""
    return any(child.dep_ == "neg" for child in token.children)


# ---------------------------------------------------------------------------
# Extractor A — Rule-Based (five dependency passes)
# ---------------------------------------------------------------------------

def _pass1_noun_args(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 1: nsubj / dobj / pobj → NOUN_ARG triples."""
    triples: List[Triple] = []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue
        subj_tok = None
        obj_toks: List[spacy.tokens.Token] = []
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass") and subj_tok is None:
                subj_tok = child
            if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                obj_toks.append(child)
        if subj_tok is None or not obj_toks:
            continue
        neg = _token_is_negated(token)
        ss, su, sc = _resolve_entity(subj_tok.text)
        ps, pu     = _resolve_obj_prop(token.lemma_, neg)
        for ot in obj_toks:
            os_, ou, oc = _resolve_entity(ot.text)
            triples.append(Triple(
                subject=ss, predicate=ps, object=os_,
                subj_uri=su, pred_uri=pu, obj_uri=ou,
                dep_type=DepType.NOUN_ARG, negated=neg,
                confidence=round((sc + oc) / 2, 3),
                source=token.sent.text, extractor="rule_based",
            ))
    return triples


def _pass2_verb_prop(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 2: ROOT / xcomp / ccomp verbs → VERB_PROP triples."""
    triples: List[Triple] = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ not in ("ROOT", "xcomp", "ccomp"):
                continue
            if token.pos_ not in ("VERB", "AUX"):
                continue
            subj_tok = obj_tok = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and subj_tok is None:
                    subj_tok = child
                if child.dep_ in ("dobj", "attr", "pobj") and obj_tok is None:
                    obj_tok = child
            if subj_tok is None or obj_tok is None:
                continue
            neg = _token_is_negated(token)
            ss, su, sc  = _resolve_entity(subj_tok.text)
            ps, pu      = _resolve_obj_prop(token.lemma_, neg)
            os_, ou, oc = _resolve_entity(obj_tok.text)
            triples.append(Triple(
                subject=ss, predicate=ps, object=os_,
                subj_uri=su, pred_uri=pu, obj_uri=ou,
                dep_type=DepType.VERB_PROP, negated=neg,
                confidence=round((sc + oc) / 2, 3),
                source=sent.text, extractor="rule_based",
            ))
    return triples


def _pass3_data_props(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 3: amod / nummod / advmod → MOD_DATA (data property) triples."""
    triples: List[Triple] = []
    for token in doc:
        if token.dep_ not in ("amod", "nummod", "advmod", "npadvmod"):
            continue
        ss, su, sc   = _resolve_entity(token.head.text)
        dp_s, dp_uri = _resolve_data_prop(token.text)
        val          = token.text
        triples.append(Triple(
            subject=ss, predicate=dp_s, object=f'"{val}"',
            subj_uri=su, pred_uri=dp_uri, obj_uri=f'"{val}"',
            dep_type=DepType.MOD_DATA, data_value=val,
            confidence=round(sc * 0.85, 3),
            source=token.sent.text, extractor="rule_based",
        ))
    return triples


def _pass4_conjunctions(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 4: cc / conj → CONJ_LOGIC (logical operator) triples."""
    triples: List[Triple] = []
    for token in doc:
        if token.dep_ != "conj":
            continue
        cc_text = "and"
        for child in token.head.children:
            if child.dep_ == "cc":
                cc_text = child.text.lower()
                break
        op_uri     = TC_LOGICAL.get(cc_text, "owl:intersectionOf")
        ls, lu, lc = _resolve_entity(token.head.text)
        rs, ru, rc = _resolve_entity(token.text)
        if lu == ru:
            continue
        triples.append(Triple(
            subject=ls, predicate=cc_text, object=rs,
            subj_uri=lu, pred_uri=op_uri, obj_uri=ru,
            dep_type=DepType.CONJ_LOGIC, logical_op=op_uri,
            confidence=round((lc + rc) / 2, 3),
            source=token.sent.text, extractor="rule_based",
        ))
    return triples


def _pass5_mereology(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 5: prep + pobj → PREP_MERO (mereological relation) triples."""
    triples: List[Triple] = []
    for token in doc:
        if token.dep_ != "prep":
            continue
        if token.text.lower() not in TC_MEREOLOGY:
            continue
        pobj_list = [c for c in token.children if c.dep_ == "pobj"]
        if not pobj_list:
            continue
        ps, pu     = _resolve_mero_prop(token.text.lower())
        cs, cu, cc = _resolve_entity(token.head.text)
        for pobj in pobj_list:
            pa, pau, pc = _resolve_entity(pobj.text)
            triples.append(Triple(
                subject=cs, predicate=ps, object=pa,
                subj_uri=cu, pred_uri=pu, obj_uri=pau,
                dep_type=DepType.PREP_MERO,
                confidence=round((cc + pc) / 2, 3),
                source=token.sent.text, extractor="rule_based",
            ))
    return triples


def extract_rule_based(text: str) -> List[Triple]:
    """
    Extractor A: run all five dependency passes over *text* and return
    a deduplicated, confidence-sorted list of triples.

    Input is capped at 10,000 characters to bound spaCy parse time.
    """
    if not str(text).strip():
        return []
    doc  = get_nlp()(str(text)[:10000])
    seen: set = set()
    out:  List[Triple] = []
    for pass_fn in (
        _pass1_noun_args,
        _pass2_verb_prop,
        _pass3_data_props,
        _pass4_conjunctions,
        _pass5_mereology,
    ):
        for triple in pass_fn(doc):
            if triple.key() not in seen:
                seen.add(triple.key())
                out.append(triple)
    out.sort(key=lambda t: t.confidence, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Extractor B — spaCy General NLP (no ontology)
# ---------------------------------------------------------------------------

def extract_spacy_general(text: str) -> List[Triple]:
    """
    Extractor B: produce triples from dependency arcs without any ontology
    mapping.  Subject/object nodes are root lemmas; predicate URIs are raw
    verb lemmas prefixed with ``rel:``.

    Maximises recall; use alongside Extractor A to measure ontology coverage.
    """
    if not str(text).strip():
        return []
    doc  = get_nlp()(str(text)[:10000])
    seen: set = set()
    out:  List[Triple] = []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue
        subj_tok = None
        obj_toks: List[spacy.tokens.Token] = []
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass") and subj_tok is None:
                subj_tok = child
            if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                obj_toks.append(child)
        if subj_tok is None or not obj_toks:
            continue
        neg     = _token_is_negated(token)
        s_surf  = subj_tok.lemma_.lower()
        p_lemma = token.lemma_.lower()
        p_uri   = f"rel:{'neg_' if neg else ''}{p_lemma}"
        s_uri   = f"ent:{s_surf.replace(' ', '_')}"
        for ot in obj_toks:
            o_surf = ot.lemma_.lower()
            o_uri  = f"ent:{o_surf.replace(' ', '_')}"
            triple = Triple(
                subject=s_surf, predicate=p_lemma, object=o_surf,
                subj_uri=s_uri, pred_uri=p_uri, obj_uri=o_uri,
                dep_type=DepType.GENERAL, negated=neg,
                confidence=0.70,
                source=token.sent.text, extractor="spacy_general",
            )
            if triple.key() not in seen:
                seen.add(triple.key())
                out.append(triple)
    return out


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------

EXTRACTORS: Dict[str, callable] = {
    "rule_based":    extract_rule_based,
    "spacy_general": extract_spacy_general,
}

EXTRACTOR_HEADER_COLORS = {
    "rule_based":    "1F497D",
    "spacy_general": "4A235A",
}


def extract_all(text: str) -> Dict[str, List[Triple]]:
    """Run all registered extractors over *text* and return results by name."""
    return {name: fn(text) for name, fn in EXTRACTORS.items()}


# ---------------------------------------------------------------------------
# TKF metrics (KD / triple counts only — this module)
# ---------------------------------------------------------------------------

def knowledge_density(triples: List[Triple], text: str) -> float:
    """
    Knowledge Density (KD) = high-confidence triples / total tokens.

    A triple is considered "typed" (high-confidence) when its confidence
    score meets or exceeds KD_CONFIDENCE_THRESHOLD.  Token count uses
    simple whitespace splitting on the lowercased response text.
    """
    tokens = [w for w in str(text).lower().split() if w.strip()]
    typed  = sum(1 for t in triples if t.confidence >= KD_CONFIDENCE_THRESHOLD)
    return round(safe_div(typed, len(tokens)), 4) if tokens else 0.0


# Metrics produced by this module (kept minimal — see pipeline2b for full TKF)
ALL_METRICS = [
    "kd_tkf",
    "tkf_score",
    "triple_count_c",
    "triple_count_g",
]


def compute_metrics(
    gold_triples: List[Triple],
    candidate_triples: List[Triple],
    response_text: str,
    extractor_name: str,
) -> Dict:
    """
    Compute KD, TKF (= KD in this module), and triple counts for one
    extractor run against a single response.

    Parameters
    ----------
    gold_triples      : triples extracted from the ground-truth answer
    candidate_triples : triples extracted from the model response
    response_text     : raw response string (used for token count)
    extractor_name    : label written into the result dict
    """
    kd  = knowledge_density(candidate_triples, response_text)
    tkf = round(W_KD * kd, 4)
    return {
        "extractor":      extractor_name,
        "kd_tkf":         kd,
        "tkf_score":      tkf,
        "triple_count_c": len(candidate_triples),
        "triple_count_g": len(gold_triples),
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_ground_truth(xl: pd.ExcelFile) -> Dict[str, str]:
    """
    Read the ground-truth sheet and return a mapping of Query_ID → answer text.

    The sheet must contain columns matching the patterns:
        - Query_ID  (regex: query.?id | qid | ^id$)
        - ground_truth  (regex: ground_truth | groundtruth)
    """
    if GROUND_TRUTH_SHEET not in xl.sheet_names:
        raise FileNotFoundError(
            f"Sheet '{GROUND_TRUTH_SHEET}' not found. "
            f"Available sheets: {xl.sheet_names}"
        )
    df = pd.read_excel(xl, sheet_name=GROUND_TRUTH_SHEET, header=0)

    qid_col = next(
        (c for c in df.columns if re.search(r"query.?id|qid|^id$", str(c).lower())), None
    )
    gt_col = next(
        (c for c in df.columns
         if "ground_truth" in str(c).lower() or "groundtruth" in str(c).lower()), None
    )
    if qid_col is None or gt_col is None:
        raise ValueError(
            f"Required columns missing. Expected Query_ID and ground_truth. "
            f"Found: {list(df.columns)}"
        )

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        qid = str(row[qid_col]).strip()
        gt  = str(row[gt_col]).strip() if pd.notna(row[gt_col]) else ""
        if qid and qid.lower() != "nan":
            mapping[qid] = gt

    print(f"    Ground truth loaded: {len(mapping)} entries")
    return mapping


def resolve_method_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify response columns (those ending with RESPONSE_SUFFIX) and return
    (column_names, method_labels) where method_labels strip the suffix.

    Falls back to all non-metadata columns when the suffix is absent.
    """
    suffix     = RESPONSE_SUFFIX.lower()
    meta_lower = {
        str(c).lower() for c in df.columns
        if re.search(r"query.?id|qid|^id$|question|prompt|input", str(c).lower())
    }
    primary = [
        c for c in df.columns
        if str(c).lower().endswith(suffix) and str(c).lower() not in meta_lower
    ]
    if primary:
        labels = [
            str(c)[:-(len(RESPONSE_SUFFIX))]
            if str(c).lower().endswith(suffix) else str(c)
            for c in primary
        ]
        return primary, labels

    fallback = [c for c in df.columns if str(c).lower() not in meta_lower]
    return fallback, [str(c) for c in fallback]


# ---------------------------------------------------------------------------
# Row-level evaluation
# ---------------------------------------------------------------------------

def evaluate_row(
    ground_truth: str,
    method_responses: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, Dict[str, List[Triple]]]]:
    """
    Evaluate all extractor × method combinations for a single query row.

    Returns
    -------
    per_method : method → extractor → metric dict
    inventory  : method → extractor → list of extracted triples
    """
    gold_triples = extract_rule_based(ground_truth)
    zero         = {k: 0.0 for k in ALL_METRICS}

    per_method: Dict[str, Dict[str, Dict]] = {}
    inventory:  Dict[str, Dict[str, List[Triple]]] = {}

    for method, response_raw in method_responses.items():
        response = str(response_raw).strip()
        if response.lower() in ("", "nan", "none", "n/a"):
            per_method[method] = {ext: {**zero, "missing": True} for ext in EXTRACTORS}
            inventory[method]  = {ext: [] for ext in EXTRACTORS}
            continue

        ext_triples = extract_all(response)
        per_method[method] = {}
        for ext_name, candidates in ext_triples.items():
            per_method[method][ext_name] = {
                **compute_metrics(gold_triples, candidates, response, ext_name),
                "missing": False,
            }
        inventory[method] = ext_triples

    return per_method, inventory


# ---------------------------------------------------------------------------
# Sheet-level processor
# ---------------------------------------------------------------------------

def process_sheet(
    model: str,
    df: pd.DataFrame,
    gt_map: Dict[str, str],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Evaluate all rows in *df* for model *model* and return three result lists:
        detail_rows  — one row per query × method × extractor
        triple_rows  — one row per extracted triple
        agg_rows     — one row per method × extractor (aggregated stats)
    """
    method_cols, method_names = resolve_method_columns(df)
    q_col   = next((c for c in df.columns if "question" in str(c).lower()), df.columns[1])
    qid_col = next(
        (c for c in df.columns if re.search(r"query.?id|qid|^id$", str(c).lower())), None
    )

    print(f"\n    Model   : {model}")
    print(f"    Methods : {method_names}")
    print(f"    Rows    : {len(df)}")

    detail_rows: List[Dict] = []
    triple_rows: List[Dict] = []
    accum = {
        m: {ext: collections.defaultdict(list) for ext in EXTRACTORS}
        for m in method_names
    }
    missing_gt_count = 0

    for ri, row in df.iterrows():
        question = str(row.get(q_col, "")).strip()
        if not question or question.lower() == "nan":
            continue

        qid = (
            str(row[qid_col]).strip()
            if qid_col and pd.notna(row.get(qid_col))
            else str(ri + 1)
        )
        ground_truth = gt_map.get(qid, "")
        if not ground_truth:
            missing_gt_count += 1

        method_data = {
            name: (str(row[col]).strip() if pd.notna(row.get(col)) else "")
            for col, name in zip(method_cols, method_names)
        }

        per_method, inventory = evaluate_row(ground_truth, method_data)

        for method in method_names:
            for ext_name in EXTRACTORS:
                metrics = per_method.get(method, {}).get(ext_name, {})
                clean   = {k: v for k, v in metrics.items() if k != "missing"}
                detail_rows.append({
                    "model":        model,
                    "query_id":     qid,
                    "question":     question[:120],
                    "ground_truth": ground_truth[:120],
                    "method":       method,
                    "extractor":    ext_name,
                    "response":     method_data.get(method, "")[:150],
                    "gt_available": bool(ground_truth),
                    "missing":      metrics.get("missing", False),
                    **clean,
                })
                for k, v in clean.items():
                    if isinstance(v, (int, float)):
                        accum[method][ext_name][k].append(float(v))

            for ext_name, triples in inventory.get(method, {}).items():
                for triple in triples:
                    triple_rows.append({
                        "model":    model,
                        "query_id": qid,
                        "method":   method,
                        **triple.as_dict(),
                    })

    if missing_gt_count:
        print(f"    WARNING: {missing_gt_count} rows had no ground-truth entry.")

    # Aggregate stats per method × extractor
    agg_rows: List[Dict] = []
    for method in method_names:
        for ext_name in EXTRACTORS:
            acc = accum[method][ext_name]
            agg: Dict = {"model": model, "method": method, "extractor": ext_name}
            for metric in ALL_METRICS:
                vals = acc.get(metric, [])
                if vals:
                    agg[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                    agg[f"{metric}_std"]  = round(float(np.std(vals)),  4)
                    agg[f"{metric}_min"]  = round(float(np.min(vals)),  4)
                    agg[f"{metric}_max"]  = round(float(np.max(vals)),  4)
                else:
                    agg[f"{metric}_mean"] = agg[f"{metric}_std"] = \
                    agg[f"{metric}_min"]  = agg[f"{metric}_max"] = 0.0
            agg_rows.append(agg)

    return detail_rows, triple_rows, agg_rows


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------

_SCORE_FILLS = {
    "high": PatternFill("solid", fgColor="C6EFCE"),
    "mid":  PatternFill("solid", fgColor="FFEB9C"),
    "low":  PatternFill("solid", fgColor="FFCCCC"),
    "mean": PatternFill("solid", fgColor="F2F2F2"),
}
_HEADER_FONT = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
_BODY_FONT   = Font(name="Calibri", size=9)
_BOLD_FONT   = Font(bold=True, name="Calibri", size=9)
_CENTER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _write_worksheet(
    ws,
    df: pd.DataFrame,
    header_color: str = "1F3864",
    score_columns: List[str] = None,
) -> None:
    """Write *df* to openpyxl worksheet *ws* with formatting and score colouring."""
    header_fill = PatternFill("solid", fgColor=header_color)
    columns     = list(df.columns)
    score_set   = set(score_columns or [])

    # Header row
    for ci, col in enumerate(columns, 1):
        cell = ws.cell(1, ci, str(col))
        cell.fill, cell.font, cell.alignment = header_fill, _HEADER_FONT, _CENTER_ALIGN

    # Data rows
    for ri, row in enumerate(df.itertuples(index=False), 2):
        is_summary_row = str(row[0]) == "MEAN"
        for ci, val in enumerate(row, 1):
            display = val if not (isinstance(val, float) and math.isnan(val)) else ""
            cell    = ws.cell(ri, ci, display)
            cell.font = _BOLD_FONT if is_summary_row else _BODY_FONT
            if is_summary_row:
                cell.fill = _SCORE_FILLS["mean"]
                continue
            if columns[ci - 1] in score_set:
                try:
                    v = float(val)
                    cell.fill = (
                        _SCORE_FILLS["high"] if v >= 0.70 else
                        _SCORE_FILLS["mid"]  if v >= 0.40 else
                        _SCORE_FILLS["low"]
                    )
                except (TypeError, ValueError):
                    pass

    # Column widths
    for ci, col in enumerate(columns, 1):
        width = max(
            len(str(col)),
            int(df[col].astype(str).str.len().max()) if len(df) else 8,
        )
        ws.column_dimensions[get_column_letter(ci)].width = min(width + 2, 40)
    ws.freeze_panes = "D2"


def _build_extractor_pivot(
    detail_rows: List[Dict],
    model: str,
    method: str,
    metric: str,
) -> Optional[pd.DataFrame]:
    """
    Pivot detail rows into a query_id × extractor table for one metric.
    Appends a MEAN summary row and best-extractor columns.
    """
    rows = [r for r in detail_rows if r["model"] == model and r["method"] == method]
    if not rows:
        return None

    data: Dict = {}
    for r in rows:
        qid = r["query_id"]
        if qid not in data:
            data[qid] = {"query_id": qid, "question": r["question"][:80]}
        data[qid][r["extractor"]] = r.get(metric, "")

    df = pd.DataFrame(list(data.values()))
    try:
        df["_sort"] = pd.to_numeric(df["query_id"], errors="coerce")
        df = df.sort_values("_sort").drop(columns="_sort")
    except Exception:
        df = df.sort_values("query_id")

    ext_cols = [c for c in df.columns if c not in ("query_id", "question")]
    numeric  = df[ext_cols].apply(pd.to_numeric, errors="coerce")
    df["best_extractor"]   = numeric.idxmax(axis=1)
    df["best_score"]       = numeric.max(axis=1).round(4)
    df["extractor_spread"] = (numeric.max(axis=1) - numeric.min(axis=1)).round(4)

    # Summary row
    summary: Dict = {"query_id": "MEAN", "question": "-- column mean --"}
    for col in ext_cols:
        try:    summary[col] = round(float(numeric[col].mean()), 4)
        except: summary[col] = ""
    for col in ("best_score", "extractor_spread"):
        try:    summary[col] = round(float(df[col].mean()), 4)
        except: summary[col] = ""
    summary["best_extractor"] = ""

    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def write_excel_output(
    out_path: str,
    all_detail: List[Dict],
    all_triples: List[Dict],
    all_agg: List[Dict],
    models: List[str],
) -> None:
    """
    Write the full evaluation output workbook.

    Sheet structure
    ---------------
    Pivot sheets   — one per model × method × metric
    Detail sheets  — row-level results per model × extractor
    Triples sheet  — all extracted triples
    Leaderboard    — global TKF ranking
    Summary sheets — per-model aggregated scores
    """
    wb = Workbook()
    wb.remove(wb.active)

    KEY_METRICS = ["tkf_score", "kd_tkf", "triple_count_c", "triple_count_g"]

    # Collect method names per model from detail rows
    model_methods: Dict[str, List[str]] = {}
    for r in all_detail:
        model_methods.setdefault(r["model"], [])
        if r["method"] not in model_methods[r["model"]]:
            model_methods[r["model"]].append(r["method"])

    # --- Pivot sheets ---
    for model in models:
        sm = re.sub(r"[^a-zA-Z0-9]", "_", model)[:10]
        for method in model_methods.get(model, []):
            smth = re.sub(r"[^a-zA-Z0-9]", "_", method)[:14]
            for metric in KEY_METRICS:
                pivot = _build_extractor_pivot(all_detail, model, method, metric)
                if pivot is None:
                    continue
                title = f"{sm}_{smth}__{metric}"[:31]
                ws    = wb.create_sheet(title)
                score_cols = [
                    c for c in pivot.columns
                    if c not in ("query_id", "question", "best_extractor")
                ]
                _write_worksheet(ws, pivot, score_columns=score_cols)

    # --- Detail sheets (one per model × extractor) ---
    for model in models:
        df_model = pd.DataFrame([r for r in all_detail if r["model"] == model])
        if df_model.empty:
            continue
        for ext in EXTRACTORS:
            df_ext = df_model[df_model["extractor"] == ext]
            if df_ext.empty:
                continue
            sm = re.sub(r"[^a-zA-Z0-9]", "_", model)[:10]
            ws = wb.create_sheet(f"Detail_{sm}_{ext[:14]}"[:31])
            _write_worksheet(
                ws, df_ext.reset_index(drop=True),
                header_color=EXTRACTOR_HEADER_COLORS.get(ext, "1F3864"),
                score_columns=["tkf_score", "kd_tkf"],
            )

    # --- Triples sheets ---
    for model in models:
        df_triples = pd.DataFrame([r for r in all_triples if r["model"] == model])
        if df_triples.empty:
            continue
        sm = re.sub(r"[^a-zA-Z0-9]", "_", model)[:10]
        ws = wb.create_sheet(f"Triples__{sm}")
        _write_worksheet(ws, df_triples.reset_index(drop=True), header_color="375623")

    # --- Global leaderboard ---
    if all_agg:
        df_lb = (
            pd.DataFrame(all_agg)
            .sort_values("tkf_score_mean", ascending=False)
            .reset_index(drop=True)
        )
        df_lb.insert(0, "rank", range(1, len(df_lb) + 1))
        ws = wb.create_sheet("Leaderboard")
        _write_worksheet(
            ws, df_lb, header_color="1F3864",
            score_columns=["tkf_score_mean", "kd_tkf_mean"],
        )

    # --- Per-model summary sheets ---
    for model in models:
        df_s = pd.DataFrame([r for r in all_agg if r["model"] == model])
        if df_s.empty:
            continue
        sm = re.sub(r"[^a-zA-Z0-9]", "_", model)[:10]
        ws = wb.create_sheet(f"Summary__{sm}")
        _write_worksheet(
            ws, df_s, header_color="17375E",
            score_columns=["tkf_score_mean", "kd_tkf_mean"],
        )

    wb.save(out_path)
    print(f"\n    Output saved: {out_path}  ({len(wb.worksheets)} sheets)")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _progress_bar(value: float, width: int = 20) -> str:
    """Return a simple ASCII progress bar for a value in [0, 1]."""
    filled = int(max(0.0, min(1.0, float(value))) * width)
    return "=" * filled + "-" * (width - filled)


def print_row_report(
    qid: str,
    question: str,
    ground_truth: str,
    method_names: List[str],
    per_method: Dict[str, Dict[str, Dict]],
) -> None:
    """Print a formatted per-row evaluation summary to stdout."""
    EXT_NAMES = list(EXTRACTORS.keys())
    COL_W     = 22
    EXT_W     = 16

    ROWS = [
        ("KD (typed/tokens)", "kd_tkf"),
        ("TKF score",         "tkf_score"),
        ("Triple count C",    "triple_count_c"),
        ("Triple count G",    "triple_count_g"),
    ]

    print(f"\n  Q{qid} {'_' * 60}")
    print(f"  Question     : {question[:110]}")
    print(f"  Ground truth : {ground_truth[:110]}")

    for method in method_names:
        print(f"\n  Method: {method}")
        header = f"  {'Metric':<{COL_W}}" + "".join(f"{e:>{EXT_W}}" for e in EXT_NAMES)
        print(header)
        print("  " + "_" * (COL_W + EXT_W * len(EXT_NAMES)))

        for label, key in ROWS:
            line = f"  {label:<{COL_W}}"
            for ext in EXT_NAMES:
                val = per_method.get(method, {}).get(ext, {}).get(key, "N/A")
                line += f"{(f'{val:.4f}' if isinstance(val, float) else str(val)):>{EXT_W}}"
            print(line)

        print(f"\n  TKF ranking ({method}):")
        ranked = sorted(
            EXT_NAMES,
            key=lambda e: per_method.get(method, {}).get(e, {}).get("tkf_score", 0),
            reverse=True,
        )
        for rank, ext in enumerate(ranked, 1):
            tkf  = per_method.get(method, {}).get(ext, {}).get("tkf_score", 0)
            kd   = per_method.get(method, {}).get(ext, {}).get("kd_tkf", 0)
            miss = " (MISSING)" if per_method.get(method, {}).get(ext, {}).get("missing") else ""
            print(
                f"    #{rank} {ext:<22} TKF={tkf:.4f} [{_progress_bar(tkf)}]"
                f"  KD={kd:.4f}{miss}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline 2A — Rule-Based + spaCy triple extraction and TKF evaluation.\n"
            "Reads an Excel workbook of model responses and writes an evaluation workbook."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / DEFAULT_INPUT_FILENAME,
        help=(
            "Path to the input Excel file (ALL_MODELS_responses.xlsx) "
            "or directory containing it."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the output Excel file. "
            "Defaults to <input_directory>/eval/pipeline2a_output.xlsx."
        ),
    )
    parser.add_argument(
        "--nlp-model",
        default=NLP_MODEL,
        metavar="MODEL",
        help="spaCy model name to use for dependency parsing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve input path: accept either a .xlsx file or the directory containing it
    input_path: Path = args.input.resolve()
    if input_path.is_dir():
        input_path = input_path / DEFAULT_INPUT_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Resolve output path: default to eval/ sub-directory next to input file
    output_path: Path = (
        args.output.resolve()
        if args.output
        else input_path.parent / "eval" / DEFAULT_OUTPUT_FILENAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Allow CLI override of the spaCy model
    global NLP_MODEL
    NLP_MODEL = args.nlp_model

    print(f"\n{'=' * 80}")
    print("  PIPELINE 2A — Rule-Based + spaCy Triple Extraction")
    print("  QAS Evaluation Framework")
    print(f"  Input      : {input_path}")
    print(f"  Output     : {output_path}")
    print(f"  Extractors : {list(EXTRACTORS.keys())}")
    print(f"  Metrics    : KD, TKF, triple counts (GED/SF/PAR/MA in pipeline2b)")
    print(f"{'=' * 80}")

    print("\n  Loading spaCy model ...")
    get_nlp()
    print(f"    Loaded '{NLP_MODEL}'")

    xl     = pd.ExcelFile(input_path)
    gt_map = load_ground_truth(xl)
    sheets = [s for s in xl.sheet_names if s not in SKIP_SHEETS]
    print(f"\n  Model sheets: {sheets}")

    all_detail:  List[Dict] = []
    all_triples: List[Dict] = []
    all_agg:     List[Dict] = []
    models:      List[str]  = []

    for sheet in sheets:
        model = sheet.strip()
        models.append(model)
        df = pd.read_excel(input_path, sheet_name=sheet, header=0)
        print(f"\n  {'_' * 78}")
        print(f"  Processing: {model}  ({len(df)} rows)")

        detail, triples, agg = process_sheet(model, df, gt_map)
        all_detail.extend(detail)
        all_triples.extend(triples)
        all_agg.extend(agg)

        # Per-row console output
        method_cols, method_names = resolve_method_columns(df)
        q_col   = next((c for c in df.columns if "question" in str(c).lower()), df.columns[1])
        qid_col = next(
            (c for c in df.columns if re.search(r"query.?id|qid|^id$", str(c).lower())), None
        )
        for ri, row in df.iterrows():
            question = str(row.get(q_col, "")).strip()
            if not question or question.lower() == "nan":
                continue
            qid = (
                str(row[qid_col]).strip()
                if qid_col and pd.notna(row.get(qid_col)) else str(ri + 1)
            )
            ground_truth = gt_map.get(qid, "")
            method_data  = {
                name: (str(row[col]).strip() if pd.notna(row.get(col)) else "")
                for col, name in zip(method_cols, method_names)
            }
            per_method, _ = evaluate_row(ground_truth, method_data)
            print_row_report(qid, question, ground_truth, method_names, per_method)

        # Per-model aggregate summary
        print(f"\n  Aggregate [{model}] — TKF by method × extractor:")
        print(f"  {'Method':<20}{'Extractor':<18}{'TKF':>10}{'KD':>10}{'Cnt_C':>8}{'Cnt_G':>8}")
        print(f"  {'_' * 74}")
        for a in all_agg:
            if a["model"] != model:
                continue
            print(
                f"  {a['method']:<20}{a['extractor']:<18}"
                f"{a.get('tkf_score_mean', 0):>10.4f}"
                f"{a.get('kd_tkf_mean', 0):>10.4f}"
                f"{a.get('triple_count_c_mean', 0):>8.1f}"
                f"{a.get('triple_count_g_mean', 0):>8.1f}"
            )

    # Global leaderboard
    print(f"\n\n{'=' * 80}")
    print("  GLOBAL LEADERBOARD (Pipeline 2A — TKF score)")
    print(f"{'=' * 80}")
    df_lb = (
        pd.DataFrame(all_agg)
        .sort_values("tkf_score_mean", ascending=False)
        .reset_index(drop=True)
    )
    print(
        f"  {'#':<4}{'Model':<14}{'Method':<20}{'Extractor':<18}"
        f"{'TKF':>10}{'KD':>10}{'Cnt_C':>8}{'Cnt_G':>8}"
    )
    print(f"  {'_' * 86}")
    for i, row in df_lb.iterrows():
        print(
            f"  {i+1:<4}{str(row['model']):<14}{str(row['method']):<20}"
            f"{str(row['extractor']):<18}"
            f"{row.get('tkf_score_mean', 0):>10.4f}"
            f"{row.get('kd_tkf_mean', 0):>10.4f}"
            f"{row.get('triple_count_c_mean', 0):>8.1f}"
            f"{row.get('triple_count_g_mean', 0):>8.1f}"
        )

    write_excel_output(str(output_path), all_detail, all_triples, all_agg, models)
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
