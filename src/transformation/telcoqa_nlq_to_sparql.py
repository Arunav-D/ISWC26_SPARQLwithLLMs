"""
telecom_nl_to_sparql.py
==========================
NLQ -> Question Logical Form -> DL/OWL2 -> SPARQL

Converts natural language questions about <ABC> broadband and landline services
into SPARQL queries against a telecom domain ontology.

Pipeline stages:
    1. Linguistic analysis   — entity, data property, and status extraction via
                               domain dictionary lookup and spaCy NER + dependency parse
    2. Question type + intent classification
    3. SPO triple extraction  — subject/predicate/object from dependency graph
                                with dynamic namespace selection (broadband:/landline:)
    4. Question Logical Form  — presuppositions, restrictions, negations,
                                conjunctions, disjunctions, quantifiers
    5. DL/OWL2 expression     — concept/role/data/n-ary assertions, negation,
                                union groups (rdf:Statement reification never used)
    6. SPARQL generation      — SELECT (8 subtypes), ASK, DESCRIBE, CONSTRUCT

N-ary relation types (no rdf:Statement anti-pattern):
    PRECONDITION_CHAIN   -> separate broadband:requires triples (OWL2 intersection)
    QUANTIFIED_RELATION  -> main triple + data property bridge
    TEMPORAL_CONTEXT     -> main triple + OPTIONAL date variable
    SPATIAL_CONTEXT      -> main triple + OPTIONAL location variable
    CONDITIONAL_ROLE     -> main triple + FILTER EXISTS per extra condition
    PARTICIPANT_SET      -> blank node with broadband:hasParticipant

Namespace scheme:
    broadband: <https://<abc>.co.uk/broadband/>   — default for hardware/internet
    landline:  <https://<abc>.co.uk/landline/>    — phone/calling/voicemail semantics
    landline: prefix emitted only when the question touches the calling domain.

Input  : Excel file with a question column.
Output : Excel file with questions, generated SPARQL, and full logical form diagnostics.

Dependencies:
    pip install pandas openpyxl spacy
    python -m spacy download en_core_web_sm

Usage:
    python telecom_nl_to_sparql_v2.py --input questions.xlsx --output results.xlsx
    python telecom_nl_to_sparql_v2.py --input questions.xlsx --output results.xlsx --sheet Sheet1
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy as _spacy


# ---------------------------------------------------------------------------
# spaCy initialisation
# ---------------------------------------------------------------------------

def _load_spacy_model(model: str = "en_core_web_sm") -> _spacy.Language:
    try:
        return _spacy.load(model)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model}' not found.\n"
            f"Run:  python -m spacy download {model}"
        )


NLP = _load_spacy_model()


# ---------------------------------------------------------------------------
# Namespace / domain trigger lists
# ---------------------------------------------------------------------------

LANDLINE_TRIGGERS: List[str] = [
    "phone", "landline", " call", "calling", "voicemail", "voice mail",
    "ring", "ringing", "dial", "dialling", "dialing", "engaged",
    "incoming call", "outgoing call", "missed call", "hang up",
    "digital voice", "phone service", "voip", "make a call", "receive a call",
    "caller id", "caller display", "1471", "1571", "nuisance call",
    "call barring", "call waiting", "call forwarding", "call diversion",
    "three-way calling",
]

BROADBAND_TRIGGERS: List[str] = [
    "broadband", "internet", "wifi", "router", "hub", "modem", "ont",
    "speed", "bandwidth", "fibre", "fiber", "fttp", "ethernet",
    "download", "upload", "latency", "ping", "connection", "signal",
    "5ghz", "2.4ghz", "mesh", "disc", "extender", "dns",
    "port", "cgnat", "dmz", "ipv4", "ipv6", "firmware",
]


def _needs_landline(question: str) -> bool:
    tl = question.lower()
    return any(t in tl for t in LANDLINE_TRIGGERS)


def _needs_broadband(question: str) -> bool:
    tl = question.lower()
    return any(t in tl for t in BROADBAND_TRIGGERS)


# ---------------------------------------------------------------------------
# Domain ontology vocabulary
# ---------------------------------------------------------------------------

TELECOM_CLASSES: Dict[str, str] = {
    # Equipment
    "smart hub plus":     "broadband:SmartHubPlus",
    "smart hub 2":        "broadband:SmartHub2",
    "smart hub":          "broadband:SmartHub",
    "wifi extender":      "broadband:WiFiExtender",
    "wifi disc":          "broadband:WiFiDisc",
    "sh31b":              "broadband:SmartHubPlus_SH31B",
    "sh20a":              "broadband:SmartHub_SH20A",
    "sw10a":              "broadband:SmartWiFi_SW10A",
    "router":             "broadband:Router",
    "modem":              "broadband:Modem",
    "hub":                "broadband:Hub",
    "ont":                "broadband:ONT",
    # Broadband services
    "1.6gbps broadband":  "broadband:Broadband1_6Gbps",
    "1.6gb broadband":    "broadband:Broadband1_6Gbps",
    "guest network":      "broadband:GuestNetwork",
    "mesh network":       "broadband:MeshNetwork",
    "port forwarding":    "broadband:PortForwarding",
    "broadband":          "broadband:Broadband",
    "internet":           "broadband:InternetService",
    "fibre":              "broadband:FiberService",
    "fiber":              "broadband:FiberService",
    "fttp":               "broadband:FTTP",
    "wifi":               "broadband:WiFi",
    # Landline / calling
    "digital voice":      "landline:DigitalVoice",
    "phone service":      "landline:PhoneService",
    "call forwarding":    "landline:CallForwarding",
    "call diversion":     "landline:CallDiversion",
    "caller display":     "landline:CallerDisplay",
    "nuisance call":      "landline:NuisanceCall",
    "call barring":       "landline:CallBarring",
    "call waiting":       "landline:CallWaiting",
    "caller id":          "landline:CallerID",
    "voice mail":         "landline:Voicemail",
    "voicemail":          "landline:Voicemail",
    "landline":           "landline:Landline",
    "voip":               "landline:VoIP",
    # Providers
    "openreach":          "broadband:Openreach",
    "talktalk":           "broadband:TalkTalk",
    "virgin":             "broadband:Virgin",
    "ee":                 "broadband:EE",
    "sky":		  "broadband:SKY",	
    "bt":                 "broadband:BT",
    # Technical
    "bandwidth":  "broadband:Bandwidth",
    "firmware":   "broadband:Firmware",
    "ethernet":   "broadband:Ethernet",
    "latency":    "broadband:Latency",
    "signal":     "broadband:Signal",
    "2.4ghz":     "broadband:Band2_4GHz",
    "5ghz":       "broadband:Band5GHz",
    "cgnat":      "broadband:CGNAT",
    "ipv4":       "broadband:IPv4",
    "ipv6":       "broadband:IPv6",
    "dns":        "broadband:DNS",
    "dmz":        "broadband:DMZ",
    # Devices
    "smart tv":  "broadband:SmartTV",
    "laptop":    "broadband:Laptop",
    "mobile":    "broadband:Mobile",
    "printer":   "broadband:Printer",
    "tablet":    "broadband:Tablet",
    "phone":     "landline:Phone",
    "xbox":      "broadband:Xbox",
    "ps5":       "broadband:PS5",
    "pc":        "broadband:PC",
    # Account / commercial
    "termination fee": "broadband:TerminationFee",
    "direct debit":    "broadband:DirectDebit",
    "contract":        "broadband:Contract",
    "payment":         "broadband:Payment",
    "account":         "broadband:Account",
    "plan":            "broadband:ServicePlan",
    "bill":            "broadband:Bill",
    # Issues
    "outage":   "broadband:Outage",
    "downtime": "broadband:Downtime",
    "problem":  "broadband:Problem",
    "fault":    "broadband:Fault",
    "error":    "broadband:Error",
    "issue":    "broadband:Issue",
}

TELECOM_ROLES: Dict[str, str] = {
    "not compatible with": "broadband:isIncompatibleWith",
    "incompatible with":   "broadband:isIncompatibleWith",
    "compatible with":     "broadband:isCompatibleWith",
    "works with":          "broadband:isCompatibleWith",
    "work with":           "broadband:isCompatibleWith",
    "connected to":        "broadband:connectedTo",
    "connects to":         "broadband:connectsTo",
    "connect to":          "broadband:connectsTo",
    "instead of":          "broadband:replacedBy",
    "upgrade to":          "broadband:upgradesTo",
    "transfer to":         "broadband:transfersTo",
    "belongs to":          "broadband:belongsTo",
    "depend on":           "broadband:dependsOn",
    "depends on":          "broadband:dependsOn",
    "belong to":           "broadband:belongsTo",
    "replaces":            "broadband:replaces",
    "replace":             "broadband:replaces",
    "affects":             "broadband:affects",
    "affect":              "broadband:affects",
    "requires":            "broadband:requires",
    "require":             "broadband:requires",
    "supports":            "broadband:supports",
    "support":             "broadband:supports",
    "provides":            "broadband:provides",
    "provide":             "broadband:provides",
    "includes":            "broadband:includes",
    "include":             "broadband:includes",
    "offers":              "broadband:offers",
    "offer":               "broadband:offers",
    "broadcasts":          "broadband:broadcasts",
    "broadcast":           "broadband:broadcasts",
    "needs":               "broadband:requires",
    "need":                "broadband:requires",
    "causes":              "broadband:hasCause",
    "cause":               "broadband:hasCause",
    "uses":                "broadband:uses",
    "use":                 "broadband:uses",
    "fix":                 "broadband:hasFix",
    "cancel":              "broadband:cancels",
    "has issue":           "broadband:hasIssue",
    "have issue":          "broadband:hasIssue",
    "make a call":         "landline:makesCall",
    "receive a call":      "landline:receivesCall",
    "forwards":            "landline:forwardsTo",
    "forward":             "landline:forwardsTo",
    "diverts":             "landline:divertsTo",
    "divert":              "landline:divertsTo",
    "bar":                 "landline:bars",
}

TELECOM_DATA_PROPS: Dict[str, str] = {
    "firmware version": "broadband:hasFirmwareVersion",
    "contract end":     "broadband:hasContractEndDate",
    "mac address":      "broadband:hasMACAddress",
    "ip address":       "broadband:hasIPAddress",
    "forward number":   "landline:hasForwardNumber",
    "divert number":    "landline:hasDivertNumber",
    "call duration":    "landline:hasCallDuration",
    "password":         "broadband:hasPassword",
    "username":         "broadband:hasUsername",
    "solution":         "broadband:hasSolution",
    "frequency":        "broadband:hasFrequency",
    "expiry":           "broadband:hasExpiryDate",
    "channel":          "broadband:hasChannel",
    "colour":           "broadband:hasStatusColor",
    "color":            "broadband:hasStatusColor",
    "status":           "broadband:hasStatus",
    "light":            "broadband:hasStatusLight",
    "speed":            "broadband:hasSpeed",
    "price":            "broadband:hasPrice",
    "cost":             "broadband:hasPrice",
    "ssid":             "broadband:hasSSID",
    "port":             "broadband:hasPort",
    "fee":              "broadband:hasFee",
    "dns":              "broadband:hasDNSSetting",
    "voicemail":        "landline:hasVoicemail",
    "ring time":        "landline:hasRingTime",
    "pin":              "landline:hasPIN",
}

STATUS_COLORS = [
    "red", "green", "blue", "orange", "amber", "aqua",
    "purple", "yellow", "white", "pink",
]
STATUS_LIGHTS = [
    "blinking", "flashing", "solid", "steady", "pulsing", "off", "on", "dim",
]

SPACY_NER_MAP: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "PERSON":      (None,                     "broadband:hasPersonName"),
    "NORP":        ("broadband:Group",         None),
    "FAC":         ("broadband:Facility",      None),
    "ORG":         ("broadband:Organisation",  None),
    "GPE":         ("broadband:Location",      None),
    "LOC":         ("broadband:Location",      None),
    "PRODUCT":     ("broadband:Product",       None),
    "EVENT":       ("broadband:Event",         None),
    "WORK_OF_ART": (None,                      None),
    "LAW":         ("broadband:Regulation",    None),
    "LANGUAGE":    (None,                      None),
    "DATE":        (None,                      "broadband:hasDate"),
    "TIME":        (None,                      "broadband:hasTime"),
    "PERCENT":     (None,                      "broadband:hasPercentage"),
    "MONEY":       (None,                      "broadband:hasPrice"),
    "QUANTITY":    (None,                      "broadband:hasQuantity"),
    "ORDINAL":     (None,                      "broadband:hasOrdinal"),
    "CARDINAL":    (None,                      "broadband:hasCardinal"),
}

# Known provider names -> ontology URI (used to refine spaCy ORG hits)
ORG_SURFACE_MAP: Dict[str, str] = {
    "ee":        "broadband:EE",
    "bt":        "broadband:BT",
    "sky":        "broadband:SKY",
    "virgin":    "broadband:Virgin",
    "talktalk":  "broadband:TalkTalk",
    "openreach": "broadband:Openreach",
    "ofcom":     "broadband:Ofcom",
}

# Verb lemma -> OWL property local name (namespace injected dynamically)
_BROADBAND_VERB_LOCALS: Dict[str, str] = {
    "require":    "requires",        "need":        "requires",
    "depend":     "dependsOn",       "support":     "supports",
    "provide":    "provides",        "connect":     "connectsTo",
    "affect":     "affects",         "include":     "includes",
    "replace":    "replaces",        "upgrade":     "upgradesTo",
    "cancel":     "cancels",         "use":         "uses",
    "offer":      "offers",          "cause":       "hasCause",
    "fix":        "hasFix",          "broadcast":   "broadcasts",
    "transfer":   "transfersTo",     "configure":   "canConfigure",
    "enable":     "canEnable",       "disable":     "canDisable",
    "install":    "canInstall",      "return":      "canReturn",
    "change":     "canChange",       "set":         "canConfigure",
    "drop":       "hasConnectionDrops", "fail":     "hasFault",
    "work":       "isCompatibleWith",   "test":     "canTest",
    "check":      "canCheck",        "reset":       "canReset",
    "restart":    "canRestart",      "update":      "canUpdate",
    "activate":   "activates",       "deactivate":  "deactivates",
    "access":     "canAccess",       "show":        "hasDisplay",
    "display":    "hasDisplay",      "limit":       "hasLimit",
    "block":      "blocks",          "allow":       "allows",
    "assign":     "assignsTo",       "detect":      "detects",
    "monitor":    "monitors",        "extend":      "extends",
    "prioritise": "prioritises",     "prioritize":  "prioritises",
    "associate":  "associatedWith",  "link":        "linkedTo",
    "route":      "routesTo",        "filter":      "filters",
    "bridge":     "bridges",         "authenticate": "authenticates",
    "authorise":  "authorises",      "authorize":   "authorises",
    "encrypt":    "encrypts",        "run":         "canRun",
    "manage":     "manages",         "control":     "controls",
    "reduce":     "reduces",         "increase":    "increases",
    "improve":    "improves",
}

_LANDLINE_VERB_LOCALS: Dict[str, str] = {
    "call":     "makesCall",      "ring":     "makesCall",
    "dial":     "makesCall",      "divert":   "divertsTo",
    "forward":  "forwardsTo",     "bar":      "bars",
    "answer":   "answers",        "receive":  "receivesCall",
    "redirect": "redirectsTo",    "mute":     "mutes",
    "unmute":   "unmutes",        "record":   "records",
    "listen":   "listensTo",      "retrieve": "retrievesMessage",
    "save":     "savesMessage",   "delete":   "deletesMessage",
    "transfer": "transfersCall",  "hold":     "holdsCall",
    "reject":   "rejectsCall",    "screen":   "screensCall",
}

_CLAUSE_HEAD_DEPS = {"ROOT", "relcl", "advcl", "xcomp", "ccomp", "conj", "acl"}
_SUBJ_DEPS        = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS         = {"dobj", "pobj", "attr", "ccomp", "xcomp", "oprd", "acomp"}


def _verb_to_uri(lemma: str, question: str) -> Optional[str]:
    """
    Resolve a verb lemma to a fully-qualified OWL property URI.

    Namespace is chosen dynamically:
        landline: if lemma belongs to the calling domain, or the question
                  triggers landline context and the lemma is transfer/forward/block.
        broadband: otherwise.

    Returns None for unknown lemmas; callers handle None without emitting a
    broken URI.
    """
    if lemma in _LANDLINE_VERB_LOCALS:
        return "landline:" + _LANDLINE_VERB_LOCALS[lemma]
    if lemma in _BROADBAND_VERB_LOCALS:
        if _needs_landline(question) and lemma in ("transfer", "forward", "block"):
            local = _LANDLINE_VERB_LOCALS.get(lemma, _BROADBAND_VERB_LOCALS[lemma])
            return ("landline:" if lemma in _LANDLINE_VERB_LOCALS else "broadband:") + local
        return "broadband:" + _BROADBAND_VERB_LOCALS[lemma]
    return None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QueryIntent(Enum):
    SELECT_GENERAL       = auto()
    SELECT_PROPERTY      = auto()
    SELECT_TROUBLESHOOT  = auto()
    SELECT_ISSUES        = auto()
    SELECT_COMPATIBILITY = auto()
    SELECT_REQUIREMENTS  = auto()
    SELECT_ACCOUNT       = auto()
    SELECT_CONFIGURE     = auto()
    ASK_BOOLEAN          = auto()
    ASK_EXISTS           = auto()
    DESCRIBE_ENTITY      = auto()
    CONSTRUCT_GRAPH      = auto()


class QuestionType(Enum):
    """
    Linguistic type of the interrogative act.

    NLQ questions are interrogative acts, not propositional assertions.
    The logical form is: lambda ?focusVar . P(?focusVar) given presuppositions.
    POLAR and EXISTENCE questions have no focus variable and map to ASK queries.
    All WH- types have a semantically derived focus variable and map to SELECT.
    """
    POLAR       = auto()
    WH_WHAT     = auto()
    WH_WHICH    = auto()
    WH_WHO      = auto()
    WH_WHEN     = auto()
    WH_WHERE    = auto()
    WH_HOW      = auto()
    WH_HOW_MANY = auto()
    WH_HOW_MUCH = auto()
    WH_WHY      = auto()
    IMPERATIVE  = auto()
    EXISTENCE   = auto()
    DEFINITION  = auto()


class NaryType(Enum):
    """
    Semantic type of an n-ary relation.

    rdf:Statement reification is never used — it is for provenance annotation
    only, not for multi-participant constraints.
    """
    PRECONDITION_CHAIN  = auto()
    PARTICIPANT_SET     = auto()
    TEMPORAL_CONTEXT    = auto()
    SPATIAL_CONTEXT     = auto()
    QUANTIFIED_RELATION = auto()
    CONDITIONAL_ROLE    = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    surface: str
    uri:     str
    role:    str  = "subject"
    negated: bool = False
    span:    int  = 0


@dataclass
class Relation:
    """
    A resolved OWL object property.

    source   : "surface_dict" | "dep_parse" | "fallback"
    dep_path : provenance arc chain, e.g. "nsubj>ROOT(require)>dobj"
    """
    surface:  str
    uri:      str
    source:   str           = "surface_dict"
    negated:  bool          = False
    subj_tok: Optional[str] = None
    obj_tok:  Optional[str] = None
    dep_path: Optional[str] = None


@dataclass
class DataProperty:
    surface: str
    uri:     str
    value:   Optional[str] = None


@dataclass
class SpacyNERHit:
    surface:    str
    label:      str
    start_char: int
    end_char:   int
    onto_class: Optional[str]
    data_prop:  Optional[str]
    negated:    bool = False

    @property
    def is_class_entity(self) -> bool:
        return self.onto_class is not None

    @property
    def is_data_value(self) -> bool:
        return self.data_prop is not None and self.onto_class is None


@dataclass
class SPOTriple:
    """
    Subject-Predicate-Object from the dependency parse.

    pred_is_uri=False means the verb could not be resolved to a known OWL
    property. The DL builder emits broadband:relatedTo with a comment rather
    than fabricating a URI.
    """
    subj:        str
    pred:        str
    obj:         str
    subj_is_uri: bool = False
    obj_is_uri:  bool = False
    pred_is_uri: bool = False
    negated:     bool = False
    dep_path:    str  = ""


@dataclass
class NaryAssertion:
    nary_type:    NaryType
    primary_role: str
    subject_var:  str
    object_var:   str
    conditions:   List[Tuple[str, str, str]]
    context_var:  Optional[str] = None
    context_prop: Optional[str] = None


@dataclass
class DLExpression:
    concept_assertions:  List[Tuple[str, str]]            = field(default_factory=list)
    role_assertions:     List[Tuple[str, str, str]]       = field(default_factory=list)
    negation_assertions: List[Tuple[str, str, str]]       = field(default_factory=list)
    nary_assertions:     List[NaryAssertion]              = field(default_factory=list)
    data_assertions:     List[Tuple[str, str, str]]       = field(default_factory=list)
    union_groups:        List[List[Tuple[str, str, str]]] = field(default_factory=list)
    class_exprs:         List[str]                        = field(default_factory=list)


@dataclass
class QuestionLogicalForm:
    """
    Logical form: lambda ?focusVar . P(?focusVar) given presuppositions.

    spo_triples is the primary relational content, derived from the dependency
    parse. It replaces a hardcoded verb->role table as the main predicate source.
    """
    raw_question:         str
    question_type:        QuestionType              = QuestionType.POLAR
    focus_var:            str                       = ""
    spo_triples:          List[SPOTriple]           = field(default_factory=list)
    presuppositions:      List[SPOTriple]           = field(default_factory=list)
    restrictions:         List[SPOTriple]           = field(default_factory=list)
    negations:            List[SPOTriple]           = field(default_factory=list)
    conjunctions:         List[SPOTriple]           = field(default_factory=list)
    disjunctions:         List[List[SPOTriple]]     = field(default_factory=list)
    entities:             List[Entity]              = field(default_factory=list)
    nary_args:            List[Entity]              = field(default_factory=list)
    relations:            List[Relation]            = field(default_factory=list)
    data_props:           List[DataProperty]        = field(default_factory=list)
    quantifier:           str                       = "existential"
    status_indicators:    List[Tuple[str, str]]     = field(default_factory=list)
    service_actions:      List[str]                 = field(default_factory=list)
    ner_hits:             List[SpacyNERHit]         = field(default_factory=list)
    intent:               Optional[QueryIntent]     = None
    disjunctive_entities: List[List[Entity]]        = field(default_factory=list)
    connective:           str                       = "AND"
    negated:              bool                      = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _varname(uri: str) -> str:
    local = uri.split(":")[-1]
    return "?" + local[0].lower() + local[1:]


def _tok_to_var(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]", "_", text.strip())
    if not clean or clean[0].isdigit():
        clean = "x" + clean
    return "?" + clean[0].lower() + clean[1:]


def _escape(text: str) -> str:
    return text.replace('"', '\\"')


# ---------------------------------------------------------------------------
# Stage 1 — Linguistic analysis
# ---------------------------------------------------------------------------

def _run_spacy_ner(text: str) -> List[SpacyNERHit]:
    doc  = NLP(text)
    hits: List[SpacyNERHit] = []
    for ent in doc.ents:
        if ent.label_ not in SPACY_NER_MAP:
            continue
        onto_class, data_prop = SPACY_NER_MAP[ent.label_]
        if ent.label_ == "ORG":
            resolved = ORG_SURFACE_MAP.get(ent.text.lower().strip())
            if resolved:
                onto_class, data_prop = resolved, None
        neg = any(
            tok.text.lower() in ("no", "not", "without", "never", "cannot")
            for tok in doc[max(0, ent.start - 3): ent.start]
        )
        hits.append(SpacyNERHit(
            surface=ent.text, label=ent.label_,
            start_char=ent.start_char, end_char=ent.end_char,
            onto_class=onto_class, data_prop=data_prop, negated=neg,
        ))
    return hits


def extract_entities(text: str) -> List[Entity]:
    """
    Two-pass extraction: domain dictionary (longest match first),
    then spaCy NER class hits for spans not already covered.
    """
    tl         = text.lower()
    seen_spans: List[Tuple[int, int]] = []
    found:      List[Entity]          = []

    for term in sorted(TELECOM_CLASSES, key=len, reverse=True):
        start = 0
        while True:
            pos = tl.find(term, start)
            if pos == -1:
                break
            end = pos + len(term)
            if not any(s <= pos < e or s < end <= e for s, e in seen_spans):
                seen_spans.append((pos, end))
                prefix = tl[max(0, pos - 20): pos]
                neg    = bool(re.search(
                    r"\b(no|not|without|never|cannot|can't|can\'t)\b", prefix
                ))
                found.append(Entity(surface=term, uri=TELECOM_CLASSES[term],
                                    role="subject", negated=neg, span=pos))
            start = pos + 1

    for hit in _run_spacy_ner(text):
        if not hit.is_class_entity:
            continue
        if any(s <= hit.start_char < e or s < hit.end_char <= e for s, e in seen_spans):
            continue
        seen_spans.append((hit.start_char, hit.end_char))
        found.append(Entity(surface=hit.surface.lower(), uri=hit.onto_class,
                            role="subject", negated=hit.negated, span=hit.start_char))

    found.sort(key=lambda e: e.span)
    for i, ent in enumerate(found):
        ent.role = "subject" if i == 0 else ("object" if i == 1 else "modifier")
    return found


def extract_data_properties(text: str) -> List[DataProperty]:
    """
    Three-pass extraction: domain dictionary, spaCy NER data hits,
    then a speed/unit regex for values like "1.6gbps" or "100mhz".
    """
    tl        = text.lower()
    seen_uris: set = set()
    dps:       List[DataProperty] = []

    for surf in sorted(TELECOM_DATA_PROPS, key=len, reverse=True):
        uri = TELECOM_DATA_PROPS[surf]
        if surf in tl and uri not in seen_uris:
            seen_uris.add(uri)
            m = re.search(re.escape(surf) + r"[\s:=]+([\w.\-#]+)", tl)
            dps.append(DataProperty(surface=surf, uri=uri,
                                    value=m.group(1).strip() if m else None))

    for hit in _run_spacy_ner(text):
        if hit.is_data_value and hit.data_prop not in seen_uris:
            seen_uris.add(hit.data_prop)
            dps.append(DataProperty(surface=hit.surface, uri=hit.data_prop, value=hit.surface))

    speed_m = re.search(r"(\d+(?:\.\d+)?)\s*(gbps|mbps|kbps|mhz|ghz|tb|gb|mb|kb)", tl)
    if speed_m and "broadband:hasSpeed" not in seen_uris:
        dps.append(DataProperty(
            surface=speed_m.group(0),
            uri="broadband:hasSpeed",
            value=speed_m.group(0).replace(" ", ""),
        ))
    return dps


def extract_status_indicators(text: str) -> List[Tuple[str, str]]:
    """Return (indicator_type, value) pairs for light states and colours."""
    tl    = text.lower()
    found: List[Tuple[str, str]] = []
    for ls in STATUS_LIGHTS:
        if ls in tl:
            for c in STATUS_COLORS:
                if c in tl:
                    found.append(("light", f"{ls}_{c}"))
            if not any(f[0] == "light" for f in found):
                found.append(("light_status", ls))
    for c in STATUS_COLORS:
        if c in tl and not any(c in v for _, v in found):
            found.append(("color", c))
    return found


def extract_service_actions(text: str) -> List[str]:
    """Return action tokens matched against common service-request patterns."""
    tl      = text.lower()
    actions: List[str] = []
    for pat, action in [
        (r"set\s*up|install|activate",                 "setup"),
        (r"mov(e|ing)|transfer|relocat",               "transfer"),
        (r"cancel|terminat|end\s+(my|the)\s+contract", "cancel"),
        (r"upgrad|faster\s+plan|better\s+speed",       "upgrade"),
        (r"return|send\s+back|refund",                 "return"),
        (r"reset|restart|reboot|factory",              "reset"),
        (r"diagnos|troubleshoot|repair",               "diagnose"),
        (r"configur|chang(e|ing)|modify|adjust",       "configure"),
        (r"check|view|monitor|inspect",                "inspect"),
        (r"disable|turn\s+off|switch\s+off",           "disable"),
        (r"enable|turn\s+on|switch\s+on",              "enable"),
    ]:
        if re.search(pat, tl):
            actions.append(action)
    return actions


def _extract_surface_relations(text: str) -> List[Relation]:
    """Surface-dictionary relation extraction, used only as a fallback."""
    tl   = text.lower()
    seen: set = set()
    found: List[Relation] = []
    for surf in sorted(TELECOM_ROLES, key=len, reverse=True):
        if surf in tl and surf not in seen:
            seen.add(surf)
            pos    = tl.find(surf)
            prefix = tl[max(0, pos - 25): pos]
            neg    = bool(re.search(
                r"\b(not|doesn't|don't|cannot|can't|never|without)\b", prefix
            ))
            found.append(Relation(surface=surf, uri=TELECOM_ROLES[surf],
                                  source="surface_dict", negated=neg))
    return found


# ---------------------------------------------------------------------------
# Stage 2 — Question type and intent classification
# ---------------------------------------------------------------------------

_QT_PATTERNS: List[Tuple[str, QuestionType]] = [
    (r"\bhow\s+many\b",                                   QuestionType.WH_HOW_MANY),
    (r"\bhow\s+much\b",                                   QuestionType.WH_HOW_MUCH),
    (r"\bwhy\b",                                          QuestionType.WH_WHY),
    (r"\bhow\b",                                          QuestionType.WH_HOW),
    (r"\bwhen\b",                                         QuestionType.WH_WHEN),
    (r"\bwhere\b",                                        QuestionType.WH_WHERE),
    (r"\bwho\b|\bwhom\b",                                 QuestionType.WH_WHO),
    (r"\bwhich\b",                                        QuestionType.WH_WHICH),
    (r"\bwhat\s+is\s+(a|an|the)\b",                       QuestionType.DEFINITION),
    (r"\bwhat\b",                                         QuestionType.WH_WHAT),
    (r"\bis\s+there\b|\bare\s+there\b|\bexists?\b",       QuestionType.EXISTENCE),
    (r"^(show|list|give|tell|describe|explain|find|display)\b",
                                                          QuestionType.IMPERATIVE),
    (r"^(is|are|does|do|will|can|has|have|was|were|should|could|would)\b",
                                                          QuestionType.POLAR),
]


def classify_question_type(question: str) -> QuestionType:
    tl = question.lower().strip()
    for pat, qt in _QT_PATTERNS:
        if re.search(pat, tl):
            return qt
    return QuestionType.POLAR


def _infer_focus_var(qt: QuestionType, entities: List[Entity],
                     data_props: List[DataProperty]) -> str:
    focus_by_qt: Dict[QuestionType, str] = {
        QuestionType.WH_WHY:      "?cause",
        QuestionType.WH_WHEN:     "?date",
        QuestionType.WH_WHERE:    "?location",
        QuestionType.WH_WHO:      "?contact",
        QuestionType.WH_HOW_MANY: "?count",
        QuestionType.WH_HOW:      "?steps",
        QuestionType.DEFINITION:  "?description",
    }
    if qt in (QuestionType.POLAR, QuestionType.EXISTENCE):
        return ""
    if qt in focus_by_qt:
        return focus_by_qt[qt]
    if qt == QuestionType.WH_HOW_MUCH:
        for dp in data_props:
            if dp.uri in ("broadband:hasPrice", "broadband:hasFee"):
                return "?" + dp.uri.split(":")[1]
        return "?price"
    return _varname(entities[0].uri) if entities else "?answer"


_INTENT_PATTERNS: List[Tuple[str, QueryIntent]] = [
    (r"\bcompatible\s+with\b",                                          QueryIntent.SELECT_COMPATIBILITY),
    (r"\bwork\s+with\b|\bwork\s+alongside\b",                           QueryIntent.SELECT_COMPATIBILITY),
    (r"\bwhat\s+are\s+(the\s+)?(requirements?|prerequisites?)\b",       QueryIntent.SELECT_REQUIREMENTS),
    (r"\bwhat\s+(is|are)\s+(the\s+)?(maximum|minimum|current|)\s*"
     r"(speed|price|cost|status|colour|color|light|version|channel)\b", QueryIntent.SELECT_PROPERTY),
    (r"\bwhat\s+is\s+(the|a|an)\s+\w+\s+(of|for|on)\b",                QueryIntent.SELECT_PROPERTY),
    (r"\bwhat\s+is\s+the\s+\w+\s+of\b",                                QueryIntent.SELECT_PROPERTY),
    (r"\b(does|do|will)\b.{0,50}\b(require|need|depend\s+on|must\s+have|necessary)\b",
                                                                         QueryIntent.SELECT_REQUIREMENTS),
    (r"\bexists?\b|\bis\s+there\s+(an?|any)\b|\bare\s+there\s+any\b",  QueryIntent.ASK_EXISTS),
    (r"\b(service|broadband|internet)\s+(down|outage|unavailable|offline)\b",
                                                                         QueryIntent.ASK_EXISTS),
    (r"\bis.{0,20}(down|offline|unavailable|not\s+working)\b",          QueryIntent.ASK_EXISTS),
    (r"^(is|are|does|do|will|can|has|have|was|were)\b",                 QueryIntent.ASK_BOOLEAN),
    (r"\bcan\s+i\s+(use|connect|run|install|access|configure)\b",       QueryIntent.ASK_BOOLEAN),
    (r"\bis\s+(it|this|that)\s+(possible|available|supported|required|compatible)\b",
                                                                         QueryIntent.ASK_BOOLEAN),
    (r"\b(show\s+me\s+(all|the\s+full)|generate\s+a\s+(graph|diagram)"
     r"|full\s+(graph|network|topology))\b",                             QueryIntent.CONSTRUCT_GRAPH),
    (r"\b(troubleshoot|why\s+(is|does|isn't|doesn't|won't|can't)"
     r"|not\s+working|keeps?\s+(dropping|disconnecting))\b",            QueryIntent.SELECT_TROUBLESHOOT),
    (r"\b(common\s+(issues?|problems?|faults?|errors?)"
     r"|what\s+(issues|problems|errors|faults)\s+(are|does|do))\b",     QueryIntent.SELECT_ISSUES),
    (r"what\s+are\s+(the\s+)?(most\s+)?(common|known|typical|frequent)"
     r"\s+(issues?|problems?|faults?|errors?)",                          QueryIntent.SELECT_ISSUES),
    (r"^(tell\s+me\s+about|describe|what\s+is\s+(a|an|the)|explain)\b", QueryIntent.DESCRIBE_ENTITY),
    (r"\bgive\s+me\s+(info|details|information)\s+(about|on)\b",        QueryIntent.DESCRIBE_ENTITY),
    (r"\b(bill|invoice|payment|direct\s+debit|contract|cancel"
     r"|termination|fee|charge|refund|credit)\b",                        QueryIntent.SELECT_ACCOUNT),
    (r"\b(set\s+up|configure|change|modify|reset|enable|disable"
     r"|password|ssid|port\s+forward|dns\s+setting)\b",                 QueryIntent.SELECT_CONFIGURE),
    (r"\b(speed|price|cost|status|channel|version|light|colour|color"
     r"|signal\s+strength)\b",                                           QueryIntent.SELECT_PROPERTY),
]


def classify_intent(question: str) -> QueryIntent:
    tl = question.lower().strip()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, tl):
            return intent
    return QueryIntent.SELECT_GENERAL


# ---------------------------------------------------------------------------
# Stage 3 — SPO extraction from dependency parse
# ---------------------------------------------------------------------------

def _resolve_tok(tok, tok_to_uri: Dict[int, str],
                 entities: List[Entity]) -> Tuple[str, bool]:
    if tok.i in tok_to_uri:
        return tok_to_uri[tok.i], True
    sub_text = " ".join(
        t.text.lower() for t in tok.subtree
        if not t.is_punct and t.text.strip()
    )
    for ent in entities:
        if ent.surface in sub_text or sub_text in ent.surface:
            return ent.uri, True
    return _tok_to_var(tok.lemma_), False


def extract_spo_triples(text: str, entities: List[Entity],
                        question: str) -> List[SPOTriple]:
    """
    Extract SPO triples from clause-head verbs (ROOT/relcl/xcomp/advcl etc.).
    Verb predicates are resolved to OWL property URIs via _verb_to_uri();
    unknown lemmas set pred_is_uri=False and are handled gracefully downstream.
    """
    doc        = NLP(text)
    triples:   List[SPOTriple] = []
    tok_to_uri: Dict[int, str] = {}
    seen:       set             = set()

    for ent in entities:
        for tok in doc:
            if tok.text.lower() in ent.surface or ent.surface in tok.text.lower():
                tok_to_uri.setdefault(tok.i, ent.uri)

    for tok in doc:
        if tok.pos_ != "VERB" or tok.dep_ not in _CLAUSE_HEAD_DEPS:
            continue
        lemma    = tok.lemma_.lower()
        pred_uri = _verb_to_uri(lemma, question)
        pred_str = pred_uri if pred_uri else lemma
        if pred_str in seen:
            continue
        seen.add(pred_str)

        neg      = any(c.dep_ == "neg" for c in tok.children)
        subj_tok = next((c for c in tok.children if c.dep_ in _SUBJ_DEPS), None)
        obj_tok  = next((c for c in tok.children if c.dep_ in _OBJ_DEPS), None)

        subj_str, subj_is_uri = (
            _resolve_tok(subj_tok, tok_to_uri, entities) if subj_tok
            else ((entities[0].uri, True) if entities else ("?subject", False))
        )
        if obj_tok:
            obj_str, obj_is_uri = _resolve_tok(obj_tok, tok_to_uri, entities)
        else:
            prep = next((c for c in tok.children if c.dep_ == "prep"), None)
            if prep:
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                obj_str, obj_is_uri = (
                    _resolve_tok(pobj, tok_to_uri, entities) if pobj
                    else (_tok_to_var(prep.lemma_), False)
                )
            else:
                obj_str, obj_is_uri = (
                    (entities[1].uri, True) if len(entities) > 1 else ("?answer", False)
                )

        triples.append(SPOTriple(
            subj=subj_str, pred=pred_str, obj=obj_str,
            subj_is_uri=subj_is_uri, obj_is_uri=obj_is_uri,
            pred_is_uri=pred_uri is not None, negated=neg,
            dep_path=f"{subj_str}>>{tok.dep_}({lemma})>>{obj_str}",
        ))
    return triples


# ---------------------------------------------------------------------------
# Stage 4 — Question Logical Form
# ---------------------------------------------------------------------------

def _classify_triple(triple: SPOTriple, qt: QuestionType, focus_var: str) -> str:
    """
    Classify an SPO triple as 'presupposition' (background condition) or
    'restriction' (directly constrains the focus variable).

    For POLAR/EXISTENCE all triples are presuppositions — they form the
    proposition evaluated by the ASK query.
    """
    if qt in (QuestionType.POLAR, QuestionType.EXISTENCE):
        return "presupposition"

    def _v(val: str, is_uri: bool) -> str:
        return _varname(val) if is_uri else (val if val.startswith("?") else _tok_to_var(val))

    if focus_var and focus_var in (_v(triple.subj, triple.subj_is_uri),
                                   _v(triple.obj,  triple.obj_is_uri)):
        return "restriction"
    return "presupposition"


def build_logical_form(question: str) -> QuestionLogicalForm:
    all_entities = extract_entities(question)
    data_props   = extract_data_properties(question)
    ner_hits     = _run_spacy_ner(question)
    status       = extract_status_indicators(question)
    actions      = extract_service_actions(question)

    qt        = classify_question_type(question)
    intent    = classify_intent(question)
    focus_var = _infer_focus_var(qt, all_entities, data_props)

    spo_triples = extract_spo_triples(question, all_entities, question)

    presuppositions: List[SPOTriple] = []
    restrictions:    List[SPOTriple] = []
    for t in spo_triples:
        if _classify_triple(t, qt, focus_var) == "restriction":
            restrictions.append(t)
        else:
            presuppositions.append(t)

    negations    = [t for t in spo_triples if t.negated]
    tl           = question.lower()
    conjunctions = (
        [t for t in spo_triples if not t.negated]
        if re.search(r"\bboth\b|\band\b|\bas well as\b|\balong with\b", tl) else []
    )

    disjunctions: List[List[SPOTriple]] = []
    if " or " in tl:
        dg = [
            [t for t in spo_triples
             if any(e.surface in part
                    for e in all_entities if e.uri in (t.subj, t.obj))]
            for part in tl.split(" or ")
        ]
        dg = [g for g in dg if g]
        if len(dg) > 1:
            disjunctions = dg

    quantifier   = ("universal"
                    if re.search(r"\ball\b|\bevery\b|\beach\b", tl)
                    else "existential")
    surface_rels = _extract_surface_relations(question)

    disj_ent: List[List[Entity]] = []
    if " or " in tl:
        disj_ent = [
            [e for e in all_entities if e.surface in part]
            for part in tl.split(" or ")
        ]
        disj_ent = [g for g in disj_ent if g]

    return QuestionLogicalForm(
        raw_question         = question,
        question_type        = qt,
        focus_var            = focus_var,
        spo_triples          = spo_triples,
        presuppositions      = presuppositions,
        restrictions         = restrictions,
        negations            = negations,
        conjunctions         = conjunctions,
        disjunctions         = disjunctions,
        entities             = all_entities[:2],
        nary_args            = all_entities[2:],
        relations            = surface_rels,
        data_props           = data_props,
        quantifier           = quantifier,
        status_indicators    = status,
        service_actions      = actions,
        ner_hits             = ner_hits,
        intent               = intent,
        disjunctive_entities = disj_ent,
        connective           = "OR" if disjunctions else "AND",
        negated              = bool(negations),
    )


# ---------------------------------------------------------------------------
# Stage 5 — DL/OWL2 expression
# ---------------------------------------------------------------------------

def lf_to_dl(lf: QuestionLogicalForm) -> DLExpression:
    """
    Translate QuestionLogicalForm to DLExpression.
    Primary source: spo_triples (dep-parse derived).
    Fallback: surface-dict relations when SPO produces nothing.
    Generic fallback: broadband:associatedWith when both are empty.
    """
    dl    = DLExpression()
    added: set = set()

    for ent in lf.entities + lf.nary_args:
        dl.concept_assertions.append((_varname(ent.uri), ent.uri))

    def _sv(t: SPOTriple) -> Tuple[str, Optional[str], str]:
        def _v(val: str, is_uri: bool) -> str:
            return _varname(val) if is_uri else (val if val.startswith("?") else _tok_to_var(val))
        return _v(t.subj, t.subj_is_uri), (t.pred if t.pred_is_uri else None), _v(t.obj, t.obj_is_uri)

    for t in lf.spo_triples:
        s, p, o = _sv(t)
        if p is None:
            p = f"broadband:relatedTo  # unresolved verb: {t.pred}"
        key = (s, p.split("#")[0].strip(), o)
        if key in added:
            continue
        added.add(key)
        (dl.negation_assertions if t.negated else dl.role_assertions).append((s, p, o))

    if not dl.role_assertions and not dl.negation_assertions:
        sv = _varname(lf.entities[0].uri) if lf.entities else "?subject"
        ov = _varname(lf.entities[1].uri) if len(lf.entities) > 1 else "?object"
        for rel in lf.relations:
            (dl.negation_assertions if rel.negated else dl.role_assertions).append((sv, rel.uri, ov))

    if not dl.role_assertions and not dl.negation_assertions and len(lf.entities) >= 2:
        dl.role_assertions.append((
            _varname(lf.entities[0].uri), "broadband:associatedWith", _varname(lf.entities[1].uri)
        ))

    # N-ary assertions
    if lf.nary_args and lf.relations:
        subj = lf.entities[0] if lf.entities else None
        obj  = lf.entities[1] if len(lf.entities) > 1 else None
        role = lf.relations[0].uri
        if subj:
            sv = _varname(subj.uri)
            ov = _varname(obj.uri) if obj else "?object"
            if role in ("broadband:requires", "broadband:dependsOn"):
                conditions = [(sv, role, _varname(e.uri)) for e in ([obj] + lf.nary_args) if e]
                dl.nary_assertions.append(NaryAssertion(
                    nary_type=NaryType.PRECONDITION_CHAIN, primary_role=role,
                    subject_var=sv, object_var=ov, conditions=conditions))
            elif any(dp.uri in ("broadband:hasSpeed", "broadband:hasPrice",
                                "broadband:hasQuantity") for dp in lf.data_props):
                ctx = next((dp for dp in lf.data_props
                            if dp.uri in ("broadband:hasSpeed", "broadband:hasPrice",
                                          "broadband:hasQuantity")), None)
                dl.nary_assertions.append(NaryAssertion(
                    nary_type=NaryType.QUANTIFIED_RELATION, primary_role=role,
                    subject_var=sv, object_var=ov, conditions=[(sv, role, ov)],
                    context_var=f"?{ctx.uri.split(':')[1]}" if ctx else None,
                    context_prop=ctx.uri if ctx else None))
            elif any(h.label == "DATE" for h in lf.ner_hits):
                dl.nary_assertions.append(NaryAssertion(
                    nary_type=NaryType.TEMPORAL_CONTEXT, primary_role=role,
                    subject_var=sv, object_var=ov, conditions=[(sv, role, ov)],
                    context_var="?date", context_prop="broadband:hasDate"))
            elif any(h.label in ("GPE", "LOC", "FAC") for h in lf.ner_hits):
                dl.nary_assertions.append(NaryAssertion(
                    nary_type=NaryType.SPATIAL_CONTEXT, primary_role=role,
                    subject_var=sv, object_var=ov, conditions=[(sv, role, ov)],
                    context_var="?location", context_prop="broadband:hasLocation"))
            else:
                dl.nary_assertions.append(NaryAssertion(
                    nary_type=NaryType.CONDITIONAL_ROLE, primary_role=role,
                    subject_var=sv, object_var=ov,
                    conditions=[(sv, role, ov)] + [
                        (_varname(e.uri), "rdf:type", e.uri) for e in lf.nary_args
                    ]))

    # Data property assertions
    if lf.entities:
        mv       = _varname(lf.entities[0].uri)
        seen_dp: set = set()
        for dp in lf.data_props:
            dvar = f"?{dp.uri.split(':')[1]}"
            val  = f'"{dp.value}"' if dp.value else dvar
            if dp.uri not in seen_dp:
                seen_dp.add(dp.uri)
                dl.data_assertions.append((mv, dp.uri, val))
        for hit in lf.ner_hits:
            if hit.is_data_value and hit.data_prop not in seen_dp:
                seen_dp.add(hit.data_prop)
                dl.data_assertions.append((mv, hit.data_prop, f'"{hit.surface}"'))
        for kind, val in lf.status_indicators:
            dl.data_assertions.append((mv, "broadband:hasStatusIndicator", f'"{kind}:{val}"'))

    for group in lf.disjunctive_entities:
        dl.class_exprs.append(f"owl:unionOf({' '.join(e.uri for e in group)})")
        dl.union_groups.append([(_varname(e.uri), "rdf:type", e.uri) for e in group])

    for disj_group in lf.disjunctions:
        grp = []
        for t in disj_group:
            s, p, o = _sv(t)
            if p:
                grp.append((s, p.split("#")[0].strip(), o))
        if grp:
            dl.union_groups.append(grp)

    if lf.quantifier == "universal":
        dl.class_exprs.append("owl:allValuesFrom  # forall -> FILTER in SPARQL")

    return dl


# ---------------------------------------------------------------------------
# Stage 6 — SPARQL builder
# ---------------------------------------------------------------------------

class SPARQLBuilder:

    def build(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        try:
            dispatch = {
                QueryIntent.ASK_BOOLEAN:         self._ask_boolean,
                QueryIntent.ASK_EXISTS:          self._ask_exists,
                QueryIntent.DESCRIBE_ENTITY:     self._describe,
                QueryIntent.CONSTRUCT_GRAPH:     self._construct,
                QueryIntent.SELECT_TROUBLESHOOT: self._select_troubleshoot,
                QueryIntent.SELECT_ISSUES:       self._select_issues,
                QueryIntent.SELECT_COMPATIBILITY: self._select_compatibility,
                QueryIntent.SELECT_REQUIREMENTS: self._select_requirements,
                QueryIntent.SELECT_ACCOUNT:      self._select_account,
                QueryIntent.SELECT_CONFIGURE:    self._select_configure,
                QueryIntent.SELECT_PROPERTY:     self._select_property,
                QueryIntent.SELECT_GENERAL:      self._select_general,
            }
            return dispatch.get(lf.intent or QueryIntent.SELECT_GENERAL,
                                self._select_general)(lf, dl)
        except Exception as exc:
            return self._fallback(lf, exc)

    def _prefixes(self, lf: QuestionLogicalForm) -> str:
        p = "PREFIX broadband: <https://<abc>.co.uk/broadband/>\n"
        if _needs_landline(lf.raw_question):
            p += "PREFIX landline:  <https://<abc>.co.uk/landline/>\n"
        return (
            p
            + "PREFIX owl:       <http://www.w3.org/2002/07/owl#>\n"
            + "PREFIX rdfs:      <http://www.w3.org/2000/01/rdf-schema#>\n"
            + "PREFIX rdf:       <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            + "PREFIX xsd:       <http://www.w3.org/2001/XMLSchema#>\n"
        )

    def _nlq_annotation(self, lf: QuestionLogicalForm, indent: str = "  ") -> str:
        """
        Embed the NLQ as a first-class RDF annotation inside the WHERE clause
        on a blank node _:queryMeta, wrapped in OPTIONAL for portability.

        Records the source question, intent, question type, and focus variable
        as typed literals, making the NLQ queryable against the metadata graph.
        """
        q  = _escape(lf.raw_question)
        fv = lf.focus_var or ""
        it = lf.intent.name if lf.intent else "UNKNOWN"
        qt = lf.question_type.name
        return (
            f"{indent}OPTIONAL {{\n"
            f"{indent}  _:queryMeta rdf:type broadband:SPARQLQuery ;\n"
            f'{indent}              rdfs:comment       "{q}"@en ;\n'
            f'{indent}              broadband:intent   "{it}"^^xsd:string ;\n'
            f'{indent}              broadband:questionType "{qt}"^^xsd:string ;\n'
            f'{indent}              broadband:focusVariable "{fv}"^^xsd:string .\n'
            f"{indent}}}"
        )

    def _type_triples(self, dl: DLExpression, indent: str = "  ") -> str:
        return "\n".join(f"{indent}{v} rdf:type {c} ." for v, c in dl.concept_assertions)

    def _role_triples(self, dl: DLExpression, indent: str = "  ",
                      emit_negation: bool = True) -> str:
        lines = [f"{indent}{s} {p} {o} ." for s, p, o in dl.role_assertions]
        if emit_negation:
            lines += [f"{indent}FILTER NOT EXISTS {{ {s} {p} {o} . }}"
                      for s, p, o in dl.negation_assertions]
        return "\n".join(lines)

    def _data_triples(self, dl: DLExpression, optional: bool = True,
                      indent: str = "  ") -> str:
        fmt = (f"{indent}OPTIONAL {{ {{s}} {{p}} {{o}} . }}"
               if optional else f"{indent}{{s}} {{p}} {{o}} .")
        return "\n".join(
            (f"{indent}OPTIONAL {{ {s} {p} {o} . }}" if optional
             else f"{indent}{s} {p} {o} .")
            for s, p, o in dl.data_assertions
        )

    def _union_block(self, dl: DLExpression, indent: str = "  ") -> str:
        if not dl.union_groups:
            return ""
        parts = ["{ " + " ".join(f"{s} {p} {o} ." for s, p, o in grp) + " }"
                 for grp in dl.union_groups]
        return indent + f"\n{indent}UNION\n{indent}".join(parts)

    def _nary_block(self, dl: DLExpression, indent: str = "  ") -> str:
        """
        Emit correct SPARQL for n-ary assertions.
        rdf:Statement is never used (provenance only, not constraints).
        """
        if not dl.nary_assertions:
            return ""
        lines: List[str] = []
        for i, na in enumerate(dl.nary_assertions):
            sv, ov = na.subject_var, na.object_var

            if na.nary_type == NaryType.PRECONDITION_CHAIN:
                owl_str = " \u2293 ".join(
                    f"(\u2203{na.primary_role}.{c[2]})" for c in na.conditions
                )
                lines.append(f"{indent}# n-ary PRECONDITION_CHAIN -- OWL2 intersection: {owl_str}")
                for s, p, o in na.conditions:
                    lines.append(f"{indent}{s} {p} {o} .")

            elif na.nary_type == NaryType.QUANTIFIED_RELATION:
                lines.append(f"{indent}# n-ary QUANTIFIED_RELATION")
                lines.append(f"{indent}{sv} {na.primary_role} {ov} .")
                if na.context_var and na.context_prop:
                    lines.append(f"{indent}{sv} {na.context_prop} {na.context_var} .")

            elif na.nary_type == NaryType.TEMPORAL_CONTEXT:
                lines.append(f"{indent}# n-ary TEMPORAL_CONTEXT")
                lines.append(f"{indent}{sv} {na.primary_role} {ov} .")
                lines.append(f"{indent}OPTIONAL {{ {sv} {na.context_prop} {na.context_var} . }}")

            elif na.nary_type == NaryType.SPATIAL_CONTEXT:
                lines.append(f"{indent}# n-ary SPATIAL_CONTEXT")
                lines.append(f"{indent}{sv} {na.primary_role} {ov} .")
                lines.append(f"{indent}OPTIONAL {{ {sv} {na.context_prop} {na.context_var} . }}")

            elif na.nary_type == NaryType.CONDITIONAL_ROLE:
                lines.append(f"{indent}# n-ary CONDITIONAL_ROLE")
                lines.append(f"{indent}{sv} {na.primary_role} {ov} .")
                for s, p, o in na.conditions[1:]:
                    lines.append(f"{indent}FILTER EXISTS {{ {s} {p} {o} . }}")

            else:  # PARTICIPANT_SET
                lines.append(f"{indent}# n-ary PARTICIPANT_SET")
                bn = f"_:naryRole{i}"
                lines.append(f"{indent}{bn} broadband:involvesRole {na.primary_role} .")
                lines.append(f"{indent}{bn} broadband:hasParticipant {sv} .")
                lines.append(f"{indent}{bn} broadband:hasParticipant {ov} .")
                for s, _, _ in na.conditions[1:]:
                    lines.append(f"{indent}{bn} broadband:hasParticipant {s} .")

        return "\n".join(lines)

    def _main_var(self, dl: DLExpression) -> Tuple[str, str]:
        if dl.concept_assertions:
            return dl.concept_assertions[0]
        return "?entity", "broadband:Device"

    def _select_general(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        mv, mc = self._main_var(dl)
        fv     = lf.focus_var or "?value"
        vars_  = " ".join(dict.fromkeys(
            [v for v, _ in dl.concept_assertions] + [fv, "?property"]
        ))
        q  = self._prefixes(lf)
        q += f"SELECT DISTINCT {vars_} WHERE {{\n"
        q += self._type_triples(dl) + "\n"
        q += self._role_triples(dl) + "\n"
        if dl.union_groups:
            q += self._union_block(dl) + "\n"
        q += f"  {mv} ?property {fv} .\n"
        q += "  OPTIONAL { ?property rdfs:label ?label . }\n"
        q += self._data_triples(dl) + "\n"
        q += ("  FILTER(STRSTARTS(STR(?property), STR(broadband:))"
              " || STRSTARTS(STR(?property), STR(landline:)))\n")
        q += self._nlq_annotation(lf) + "\n"
        q += "}\nORDER BY ?property\n"
        return q

    def _select_property(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        mv, _  = self._main_var(dl)
        fv     = lf.focus_var or "?propertyValue"
        dp_vars = " ".join(f"?{dp.uri.split(':')[1]}" for dp in lf.data_props) or fv
        q  = self._prefixes(lf)
        q += f"SELECT DISTINCT {mv} {dp_vars} WHERE {{\n"
        q += self._type_triples(dl) + "\n"
        if dl.data_assertions:
            q += self._data_triples(dl, optional=False) + "\n"
        else:
            q += f"  {mv} ?anyProp {fv} .\n"
            q += ("  FILTER(STRSTARTS(STR(?anyProp), STR(broadband:has))"
                  " || STRSTARTS(STR(?anyProp), STR(landline:has)))\n")
        q += self._role_triples(dl) + "\n"
        q += self._nlq_annotation(lf) + "\n"
        q += f"}}\nORDER BY {mv}\n"
        return q

    def _select_compatibility(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        q = self._prefixes(lf)
        e = lf.entities
        if len(e) >= 2:
            sv, ov = _varname(e[0].uri), _varname(e[1].uri)
            q += f"SELECT DISTINCT {sv} {ov} ?compatibilityStatus ?restriction WHERE {{\n"
            q += f"  {sv} rdf:type {e[0].uri} .\n"
            q += f"  {ov} rdf:type {e[1].uri} .\n"
            q += f"  {sv} broadband:isCompatibleWith {ov} .\n"
            q += f"  OPTIONAL {{ {sv} broadband:compatibilityStatus ?compatibilityStatus . }}\n"
            q += f"  OPTIONAL {{ {sv} broadband:hasRestriction ?restriction . }}\n"
            q += f"  FILTER NOT EXISTS {{ {sv} broadband:isIncompatibleWith {ov} . }}\n"
        elif len(e) == 1:
            fv = lf.focus_var or "?compatibleWith"
            sv = _varname(e[0].uri)
            q += f"SELECT DISTINCT {sv} {fv} ?category ?restriction WHERE {{\n"
            q += f"  {sv} rdf:type {e[0].uri} .\n"
            q += f"  {sv} broadband:isCompatibleWith {fv} .\n"
            q += f"  OPTIONAL {{ {fv} rdf:type ?category . }}\n"
            q += f"  OPTIONAL {{ {fv} broadband:hasRestriction ?restriction . }}\n"
        else:
            fv = lf.focus_var or "?compatibleWith"
            q += f"SELECT DISTINCT ?subject {fv} ?restriction WHERE {{\n"
            q += f"  ?subject broadband:isCompatibleWith {fv} .\n"
            q += f"  OPTIONAL {{ {fv} broadband:hasRestriction ?restriction . }}\n"
        if dl.union_groups:
            q += self._union_block(dl) + "\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\nORDER BY ?compatibleWith\n"
        return q

    def _select_requirements(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        q = self._prefixes(lf)
        e = lf.entities
        if len(e) >= 2:
            sv, ov = _varname(e[0].uri), _varname(e[1].uri)
            q += f"SELECT DISTINCT {sv} {ov} ?requirement ?isMandatory ?details WHERE {{\n"
            q += f"  {sv} rdf:type {e[0].uri} .\n"
            q += f"  {ov} rdf:type {e[1].uri} .\n"
            q += f"  # OWL2: {sv} SubClassOf (exists broadband:requires.Requirement)\n"
            q += f"  {sv} broadband:requires ?requirement .\n"
            q += f"  OPTIONAL {{ ?requirement broadband:isMandatory ?isMandatory . }}\n"
            q += f"  OPTIONAL {{ ?requirement broadband:hasDetails ?details . }}\n"
            q += f"  OPTIONAL {{ {ov} broadband:requires ?requirement . }}\n"
        elif len(e) == 1:
            sv = _varname(e[0].uri)
            if lf.quantifier == "universal":
                q += f"SELECT DISTINCT {sv} ?requirement ?isMandatory WHERE {{\n"
                q += f"  {sv} rdf:type {e[0].uri} .\n"
                q += f"  {sv} broadband:requires ?requirement .\n"
                q += "  OPTIONAL { ?requirement broadband:isMandatory ?isMandatory . }\n"
                q += '  FILTER(?isMandatory = "true"^^xsd:boolean)\n'
            else:
                q += f"SELECT DISTINCT {sv} ?requirement ?isMandatory ?priority WHERE {{\n"
                q += f"  {sv} rdf:type {e[0].uri} .\n"
                q += f"  {sv} broadband:requires ?requirement .\n"
                q += "  OPTIONAL { ?requirement broadband:isMandatory ?isMandatory . }\n"
                q += "  OPTIONAL { ?requirement broadband:hasPriority ?priority . }\n"
        else:
            q += "SELECT DISTINCT ?entity ?requirement ?isMandatory WHERE {\n"
            q += "  ?entity broadband:requires ?requirement .\n"
            q += "  OPTIONAL { ?requirement broadband:isMandatory ?isMandatory . }\n"
        if dl.nary_assertions:
            q += self._nary_block(dl) + "\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\nORDER BY DESC(?isMandatory) ?requirement\n"
        return q

    def _select_troubleshoot(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        mv, mc = self._main_var(dl)
        fv     = lf.focus_var or "?cause"
        q  = self._prefixes(lf)
        q += f"SELECT DISTINCT {mv} ?issue ?issueType {fv} ?solution ?priority ?nextStep WHERE {{\n"
        q += f"  {mv} rdf:type {mc} .\n"
        for kind, val in lf.status_indicators:
            q += f"  OPTIONAL {{ {mv} broadband:hasStatusIndicator ?si .\n"
            q += f'    FILTER(STR(?si) = "{kind}:{val}") }}\n'
        q += f"  {mv} broadband:hasIssue ?issue .\n"
        q += "  ?issue broadband:issueType ?issueType .\n"
        q += f"  ?issue broadband:hasCause {fv} .\n"
        q += "  ?issue broadband:hasSolution ?solution .\n"
        q += "  OPTIONAL { ?issue broadband:hasPriority ?priority . }\n"
        q += "  OPTIONAL { ?solution broadband:hasNextStep ?nextStep . }\n"
        for ent in lf.entities:
            if ent.negated:
                q += f"  FILTER NOT EXISTS {{ {mv} rdf:type {ent.uri} . }}\n"
        if dl.union_groups:
            q += self._union_block(dl) + "\n"
        if dl.nary_assertions:
            q += self._nary_block(dl) + "\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\nORDER BY ?priority ?issue\n"
        return q

    def _select_issues(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        mv, mc = self._main_var(dl)
        q  = self._prefixes(lf)
        q += f"SELECT DISTINCT {mv} ?issue ?issueType ?frequency ?cause ?solution WHERE {{\n"
        q += f"  {mv} rdf:type {mc} .\n"
        q += f"  {mv} broadband:hasKnownIssue ?issue .\n"
        q += "  ?issue broadband:issueType ?issueType .\n"
        q += "  ?issue broadband:hasFrequency ?frequency .\n"
        q += "  ?issue broadband:hasCause ?cause .\n"
        q += "  ?issue broadband:hasSolution ?solution .\n"
        if dl.union_groups:
            q += self._union_block(dl) + "\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\nORDER BY DESC(?frequency) ?issueType\n"
        return q

    def _select_account(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        q    = self._prefixes(lf)
        acts = lf.service_actions
        if "cancel" in acts:
            q += "SELECT DISTINCT ?account ?contractEndDate ?terminationF<ABC>?cancelProcess WHERE {\n"
            q += "  ?account rdf:type broadband:Account .\n"
            q += "  ?account broadband:hasContract ?contract .\n"
            q += "  OPTIONAL { ?contract broadband:hasContractEndDate ?contractEndDate . }\n"
            q += "  OPTIONAL { ?contract broadband:hasTerminationF<ABC>?terminationF<ABC>. }\n"
            q += "  OPTIONAL { ?account broadband:hasCancellationProcess ?cancelProcess . }\n"
            q += '  FILTER NOT EXISTS { ?account broadband:hasStatus "cancelled"^^xsd:string . }\n'
        elif "upgrade" in acts:
            mv, mc = self._main_var(dl)
            q += f"SELECT DISTINCT {mv} ?upgradePlan ?upgradePrice ?upgradeAvailability WHERE {{\n"
            q += f"  {mv} rdf:type {mc} .\n"
            q += f"  {mv} broadband:upgradesTo ?upgradePlan .\n"
            q += "  OPTIONAL { ?upgradePlan broadband:hasPrice ?upgradePrice . }\n"
            q += "  OPTIONAL { ?upgradePlan broadband:hasAvailability ?upgradeAvailability . }\n"
        else:
            fv = lf.focus_var or "?details"
            if lf.question_type == QuestionType.WH_WHEN:
                q += f"SELECT DISTINCT ?account {fv} WHERE {{\n"
                q += "  ?account rdf:type broadband:Account .\n"
                q += "  ?account broadband:hasContract ?contract .\n"
                q += f"  ?contract broadband:hasContractEndDate {fv} .\n"
            else:
                q += "SELECT DISTINCT ?account ?billAmount ?dueDate ?paymentMethod WHERE {\n"
                q += "  ?account rdf:type broadband:Account .\n"
                q += "  OPTIONAL { ?account broadband:hasBillAmount ?billAmount . }\n"
                q += "  OPTIONAL { ?account broadband:hasDueDate ?dueDate . }\n"
                q += "  OPTIONAL { ?account broadband:hasPaymentMethod ?paymentMethod . }\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\nORDER BY ?account\n"
        return q

    def _select_configure(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        mv, mc  = self._main_var(dl)
        fv      = lf.focus_var or "?steps"
        dp_vars = (" ".join(f"?{dp.uri.split(':')[1]}" for dp in lf.data_props)
                   or "?settingName ?currentValue")
        q  = self._prefixes(lf)
        q += f"SELECT DISTINCT {mv} {dp_vars} {fv} WHERE {{\n"
        q += f"  {mv} rdf:type {mc} .\n"
        if lf.data_props:
            for dp in lf.data_props:
                dvar = f"?{dp.uri.split(':')[1]}"
                if dp.value:
                    q += f"  {mv} {dp.uri} {dvar} .\n"
                    q += f'  FILTER(STR({dvar}) = "{dp.value}")\n'
                else:
                    q += f"  OPTIONAL {{ {mv} {dp.uri} {dvar} . }}\n"
        else:
            q += f"  {mv} broadband:hasSetting ?settingName .\n"
            q += "  OPTIONAL { ?settingName broadband:hasCurrentValue ?currentValue . }\n"
            q += "  OPTIONAL { ?settingName broadband:hasAllowedValues ?allowedValues . }\n"
        q += f"  OPTIONAL {{ {mv} broadband:hasConfigurationStep {fv} . }}\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\nORDER BY ?settingName\n"
        return q

    def _ask_boolean(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        q  = self._prefixes(lf)
        q += "ASK {\n"
        q += self._type_triples(dl) + "\n"
        q += self._role_triples(dl, emit_negation=True) + "\n"
        q += self._data_triples(dl, optional=False) + "\n"
        if dl.union_groups:
            q += self._union_block(dl) + "\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\n"
        return q

    def _ask_exists(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        q  = self._prefixes(lf)
        q += "ASK {\n"
        for var, cls in dl.concept_assertions:
            q += f"  {var} rdf:type {cls} .\n"
            for ent in lf.entities:
                if ent.negated and _varname(ent.uri) == var:
                    q += f'  FILTER NOT EXISTS {{ {var} broadband:hasStatus "inactive" . }}\n'
        if lf.relations:
            sv = _varname(lf.entities[0].uri) if lf.entities else "?entity"
            q += f"  {sv} {lf.relations[0].uri} ?target .\n"
        q += self._nlq_annotation(lf) + "\n"
        q += "}\n"
        return q

    def _describe(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        q = self._prefixes(lf)
        if lf.entities:
            uris = " ".join(e.uri for e in lf.entities)
            q   += f"DESCRIBE {uris}\nWHERE {{\n"
            if lf.relations:
                q += self._role_triples(dl) + "\n"
            q += self._nlq_annotation(lf, indent="  ") + "\n"
            q += "}\n"
        else:
            q += "DESCRIBE ?resource\nWHERE {\n"
            q += "  ?resource rdf:type broadband:Device .\n"
            q += self._nlq_annotation(lf, indent="  ") + "\n"
            q += "}\n"
        return q

    def _construct(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        q     = self._prefixes(lf)
        tmpl: List[str]  = []
        where: List[str] = []

        for var, cls in dl.concept_assertions:
            tmpl.append(f"  {var} rdf:type {cls} .")
            where.append(f"  {var} rdf:type {cls} .")
        for s, p, o in dl.role_assertions:
            tmpl.append(f"  {s} {p} {o} .")
            where.append(f"  {s} {p} {o} .")
        for s, p, o in dl.data_assertions:
            tmpl.append(f"  {s} {p} {o} .")
            where.append(f"  OPTIONAL {{ {s} {p} {o} . }}")

        if lf.entities:
            mv = _varname(lf.entities[0].uri)
            tmpl.append(f"  {mv} ?p ?o .")
            where.append(
                f"  OPTIONAL {{ {mv} ?p ?o .\n"
                "    FILTER(STRSTARTS(STR(?p), STR(broadband:)) || "
                "STRSTARTS(STR(?p), STR(landline:))) }}"
            )

        if dl.union_groups:
            where.append(self._union_block(dl))

        where.append(self._nlq_annotation(lf, indent="  "))
        q += ("CONSTRUCT {\n" + "\n".join(tmpl) + "\n}\nWHERE {\n"
              + "\n".join(where) + "\n}\n")
        return q

    def _fallback(self, lf: QuestionLogicalForm, exc: Exception) -> str:
        ns = "landline:" if _needs_landline(lf.raw_question) else "broadband:"
        return (
            self._prefixes(lf)
            + f"# ERROR: {exc}\n"
            + "SELECT DISTINCT ?subject ?predicate ?object WHERE {\n"
            + "  ?subject ?predicate ?object .\n"
            + f"  FILTER(STRSTARTS(STR(?predicate), STR({ns})))\n"
            + self._nlq_annotation(lf) + "\n"
            + "} LIMIT 20\n"
        )


_BUILDER = SPARQLBuilder()


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def nlq_to_sparql(question: str) -> Dict[str, Any]:
    """
    Run the full NLQ -> SPARQL pipeline for a single question.

    Returns a flat dict suitable for direct use as a DataFrame row,
    including the generated SPARQL, full logical form diagnostics,
    and NER breakdowns for all 18 spaCy entity types.
    """
    lf     = build_logical_form(question)
    dl     = lf_to_dl(lf)
    sparql = _BUILDER.build(lf, dl)

    ner_by_label: Dict[str, List[str]] = {}
    for h in lf.ner_hits:
        ner_by_label.setdefault(h.label, []).append(
            h.surface + (f"=>{h.onto_class}" if h.onto_class else f"=>{h.data_prop}")
        )

    return {
        "Question":           question,
        "Intent":             lf.intent.name if lf.intent else "UNKNOWN",
        "SPARQL Query":       sparql,
        "Question Type":      lf.question_type.name,
        "Focus Variable":     lf.focus_var,
        "SPO Triples":        str([(t.subj, t.pred, t.obj,
                                    "NEG" if t.negated else "", t.dep_path)
                                   for t in lf.spo_triples]),
        "Presuppositions":    str([(t.subj, t.pred, t.obj) for t in lf.presuppositions]),
        "Restrictions":       str([(t.subj, t.pred, t.obj) for t in lf.restrictions]),
        "Negations":          str([(t.subj, t.pred, t.obj) for t in lf.negations]),
        "Conjunctions":       str([(t.subj, t.pred, t.obj) for t in lf.conjunctions]),
        "Disjunctions":       str([[(t.subj, t.pred, t.obj) for t in g]
                                   for g in lf.disjunctions]),
        "Entities":           str([(e.surface, e.uri, e.role, "NEG" if e.negated else "")
                                   for e in lf.entities]),
        "N-ary Entities":     str([(e.surface, e.uri) for e in lf.nary_args]),
        "Relations (surface)": str([(r.surface, r.uri, r.source, "NEG" if r.negated else "")
                                    for r in lf.relations]),
        "Data Properties":    str([(d.surface, d.uri, d.value) for d in lf.data_props]),
        "Boolean Logic":      f"connective={lf.connective}  negated={lf.negated}",
        "Quantifier":         lf.quantifier,
        "Status Indicators":  str(lf.status_indicators),
        "Service Actions":    str(lf.service_actions),
        "DL Role Assertions": str(dl.role_assertions),
        "DL Negations":       str(dl.negation_assertions),
        "DL N-ary Count":     len(dl.nary_assertions),
        "DL N-ary Types":     str([na.nary_type.name for na in dl.nary_assertions]),
        "DL Union Groups":    len(dl.union_groups),
        "OWL2 Class Exprs":   str(dl.class_exprs),
        "NER_PERSON":         str(ner_by_label.get("PERSON",      [])),
        "NER_ORG":            str(ner_by_label.get("ORG",         [])),
        "NER_PRODUCT":        str(ner_by_label.get("PRODUCT",     [])),
        "NER_GPE":            str(ner_by_label.get("GPE",         [])),
        "NER_LOC":            str(ner_by_label.get("LOC",         [])),
        "NER_FAC":            str(ner_by_label.get("FAC",         [])),
        "NER_NORP":           str(ner_by_label.get("NORP",        [])),
        "NER_EVENT":          str(ner_by_label.get("EVENT",       [])),
        "NER_LAW":            str(ner_by_label.get("LAW",         [])),
        "NER_DATE":           str(ner_by_label.get("DATE",        [])),
        "NER_TIME":           str(ner_by_label.get("TIME",        [])),
        "NER_MONEY":          str(ner_by_label.get("MONEY",       [])),
        "NER_QUANTITY":       str(ner_by_label.get("QUANTITY",    [])),
        "NER_PERCENT":        str(ner_by_label.get("PERCENT",     [])),
        "NER_CARDINAL":       str(ner_by_label.get("CARDINAL",    [])),
        "NER_ORDINAL":        str(ner_by_label.get("ORDINAL",     [])),
        "NER_LANGUAGE":       str(ner_by_label.get("LANGUAGE",    [])),
        "NER_WORK_OF_ART":    str(ner_by_label.get("WORK_OF_ART", [])),
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _find_question_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "question" in str(col).lower():
            return col
    raise ValueError(f"No question column found. Available columns: {list(df.columns)}")


def process_excel(input_path: str | Path,
                  sheet_name: str | None = None) -> List[Dict[str, Any]]:
    """Load an Excel file and run the pipeline against every non-empty question row."""
    kwargs = {"sheet_name": sheet_name} if sheet_name else {}
    df     = pd.read_excel(input_path, **kwargs)
    q_col  = _find_question_column(df)

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        q = str(row[q_col]).strip()
        if q and q.lower() != "nan":
            try:
                results.append(nlq_to_sparql(q))
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "Question":     q,
                    "Intent":       "ERROR",
                    "SPARQL Query": f"# Error: {exc}",
                })
    return results


def save_results(results: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Write *results* to an Excel file at *output_path*."""
    if not results:
        print("No results to write.", file=sys.stderr)
        return
    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"Saved {len(results)} rows to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert natural language <ABC>telecom questions to SPARQL."
    )
    parser.add_argument("--input",  required=True, help="Path to input .xlsx file")
    parser.add_argument("--output", required=True, help="Path for output .xlsx file")
    parser.add_argument("--sheet",  default=None,  help="Sheet name (optional)")
    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    results = process_excel(input_path, sheet_name=args.sheet)
    save_results(results, args.output)

    for r in results[:3]:
        print(f"\nQ  : {r['Question']}")
        print(f"   QType  : {r['Question Type']}  |  FocusVar: {r['Focus Variable']}")
        print(f"   Intent : {r['Intent']}")
        print(f"   SPOs   : {r['SPO Triples'][:120]}")
        print(f"   N-ary  : {r['DL N-ary Types']}")
        print(f"\n   SPARQL:\n{r['SPARQL Query']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
