"""
legal_nl_to_sparql.py
=====================
NLQ -> Question Logical Form -> DL/OWL2 -> SPARQL

Converts natural language questions about Australian law into SPARQL queries
against a legal domain ontology.

Pipeline stages:
    1. Linguistic analysis    — entity, relation, and data property extraction via
                                dictionary lookup and spaCy NER + dependency parse
    2. Question classification — type (WH, polar, existence) and intent
    3. SPO triple extraction   — subject/predicate/object from dependency graph
    4. Question Logical Form   — presuppositions, restrictions, negations, quantifiers
    5. DL/OWL2 expression      — concept/role/data assertions, negation, union groups
    6. SPARQL generation       — SELECT with projected fields, OPTIONAL blocks,
                                FILTER NOT EXISTS for negation, NLQ annotation

Domain coverage:
    Query types  : CaseLaw, Legislation, AviationLaw, ProfessionalConduct,
                   RegulatoryCompliance, LegalEntities, LegalPrinciples, CourtProcedures
    Jurisdictions: NSW, Federal, Commonwealth, Tasmania, ACT,
                   Western Australia, Queensland, Victoria, Northern Territory, South Australia
    Legal topics : general_law, administrative, criminal, civil, immigration,
                   property, aviation, procedural, corporate, constitutional,
                   regulatory, taxation, environmental, employment, family, trade
    Doc types    : Case Law, Statute, Regulation, Airworthiness Directive,
                   Legal Document, Legal Principle
    Property cats: legal_context, legal_interpretation, legal_outcome,
                   legal_parties, legal_timing, legal_requirement,
                   disciplinary_action, legal_definition, legal_charges,
                   legal_argument, legal_appeal, legal_reasoning

Namespace scheme (selected dynamically from question jurisdiction):
    legal:    <https://www.legislation.{jurisdiction_domain}/>
    property: <https://www.legislation.{jurisdiction_domain}/property/>
    case:     <https://www.caselaw.{jurisdiction_domain}/>    (CaseLaw queries only)

Input  : Excel file with a 'question' column.
Output : Excel file with questions, generated SPARQL, and full logical form diagnostics.

Dependencies:
    pip install pandas openpyxl spacy
    python -m spacy download en_core_web_sm

Usage:
    python legal_nl_to_sparql.py --input legal_queries.xlsx --output results.xlsx
    python legal_nl_to_sparql.py --input legal_queries.xlsx --output results.xlsx --sheet Sheet1
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
# Jurisdiction / namespace mapping
# ---------------------------------------------------------------------------

JURISDICTION_NS: Dict[str, str] = {
    "new south wales":            "https://www.legislation.nsw.gov.au/",
    "nsw":                        "https://www.legislation.nsw.gov.au/",
    "federal":                    "https://www.legislation.gov.au/",
    "commonwealth":               "https://www.legislation.gov.au/",
    "australia":                  "https://www.legislation.gov.au/",
    "tasmania":                   "https://www.legislation.tas.gov.au/",
    "act":                        "https://www.legislation.act.gov.au/",
    "australian capital territory": "https://www.legislation.act.gov.au/",
    "western australia":          "https://www.legislation.wa.gov.au/",
    "queensland":                 "https://www.legislation.qld.gov.au/",
    "victoria":                   "https://www.legislation.vic.gov.au/",
    "northern territory":         "https://www.legislation.nt.gov.au/",
    "south australia":            "https://www.legislation.sa.gov.au/",
}

# Court abbreviation -> jurisdiction
COURT_JURISDICTION: Dict[str, str] = {
    "hca":       "commonwealth",      "fca":       "federal",
    "fcafc":     "federal",           "fcca":      "federal",
    "nswsc":     "nsw",               "nswca":     "nsw",
    "nswcca":    "nsw",               "nswdc":     "nsw",
    "nswlec":    "nsw",               "nswirc":    "nsw",
    "nswadt":    "nsw",               "nswcatap":  "nsw",
    "nswcatad":  "nsw",               "nswcatod":  "nsw",
    "nswcatcd":  "nsw",               "nswcatgd":  "nsw",
    "nswadtap":  "nsw",               "nswlc":     "nsw",
    "nswcc":     "nsw",               "irca":      "federal",
    "aata":      "commonwealth",      "aat":       "commonwealth",
    "fcaa":      "federal",           "vsc":       "victoria",
    "vsca":      "victoria",          "qsc":       "queensland",
    "qca":       "queensland",        "wasc":      "western australia",
    "wasca":     "western australia", "sasc":      "south australia",
    "sascfc":    "south australia",
}

_CASE_CITATION_RE = re.compile(r'\[(\d{4})\]\s*([A-Z]+(?:[A-Z]+)?)\s*(\d+)', re.IGNORECASE)
_ACT_RE           = re.compile(
    r'(?:the\s+)?([A-Z][A-Za-z\s]+(?:Act|Regulations?|Rules?|Order|Code|Directive)'
    r'(?:\s+\d{4})?(?:\s+\([A-Za-z]+\))?)',
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r's(?:ection)?\s*(\d+[A-Z]?(?:\([a-z0-9]+\))*)', re.IGNORECASE)
_PARTY_RE   = re.compile(
    r'(?:case\s+of\s+|in\s+|decision\s+in\s+)'
    r'([A-Z][A-Za-z\s&,\.\(\)\']+?)\s+v\s+([A-Z][A-Za-z\s&,\.\(\)\']+?)'
    r'(?:\s*\[|\s*\(|\s*$)',
    re.MULTILINE,
)


def _jurisdiction_from_question(question: str) -> str:
    """
    Infer jurisdiction from explicit place names, court abbreviations in case
    citations, or federal/commonwealth signal words. Defaults to 'federal'.
    """
    tl = question.lower()

    _norm: Dict[str, str] = {
        "tasmanian": "tasmania", "victorian": "victoria",
    }
    for jur in ("new south wales", "nsw", "tasmania", "tasmanian",
                "australian capital territory", "western australia", "queensland",
                "victoria", "victorian", "northern territory", "south australia"):
        if jur in tl:
            key = _norm.get(jur, jur)
            return key if key in JURISDICTION_NS else jur

    court_match = _CASE_CITATION_RE.search(question)
    if court_match:
        court = court_match.group(2).lower()
        if court in COURT_JURISDICTION:
            return COURT_JURISDICTION[court]

    if re.search(r'\bfederal\b|\bcommonwealth\b|\bcth\b|\b\(cth\)\b', tl):
        return "commonwealth"
    if re.search(r'\bfca\b|\bfcafc\b|\bhca\b|\bfederal court\b|\bhigh court\b', tl):
        return "federal"

    return "federal"


def _namespace_for(jurisdiction: str) -> str:
    return JURISDICTION_NS.get(jurisdiction.lower().strip(),
                               "https://www.legislation.gov.au/")


_CASE_LAW_QUERY_TYPES = {
    "CaseLaw", "LegalEntities", "ProfessionalConduct",
    "CourtProcedures", "LegalPrinciples",
}


def _build_prefixes(jurisdiction: str, query_type: str) -> str:
    ns      = _namespace_for(jurisdiction)
    prop_ns = ns.rstrip("/") + "/property/"
    case_ns = ns.replace("legislation", "caselaw")

    prefixes = f"PREFIX legal:    <{ns}>\nPREFIX property: <{prop_ns}>\n"
    if query_type in _CASE_LAW_QUERY_TYPES:
        prefixes += f"PREFIX case:     <{case_ns}>\n"
    return (
        prefixes
        + "PREFIX owl:      <http://www.w3.org/2002/07/owl#>\n"
        + "PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>\n"
        + "PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
        + "PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>\n"
        + "PREFIX skos:     <http://www.w3.org/2004/02/skos/core#>\n"
    )


# ---------------------------------------------------------------------------
# Domain ontology vocabulary
# ---------------------------------------------------------------------------

LEGAL_CLASSES: Dict[str, str] = {
    # Case law entities
    "administrative appeals tribunal": "legal:AdministrativeAppealsTribunal",
    "civil and administrative tribunal": "legal:CivilAdministrativeTribunal",
    "industrial relations commission": "legal:IndustrialRelationsCommission",
    "land and environment court":      "legal:LandEnvironmentCourt",
    "court of appeal":                 "legal:CourtOfAppeal",
    "airworthiness directive":         "legal:AirworthinessDirective",
    "statutory instrument":            "legal:StatutoryInstrument",
    "supreme court":                   "legal:SupremeCourt",
    "federal court":                   "legal:FederalCourt",
    "district court":                  "legal:DistrictCourt",
    "court decision":                  "legal:CourtCase",
    "high court":                      "legal:HighCourt",
    "court case":                      "legal:CourtCase",
    # Parties
    "prosecutor":  "legal:Prosecutor",
    "magistrate":  "legal:Magistrate",
    "respondent":  "legal:Respondent",
    "defendant":   "legal:Defendant",
    "appellant":   "legal:Appellant",
    "applicant":   "legal:Applicant",
    "barrister":   "legal:Barrister",
    "plaintiff":   "legal:Plaintiff",
    "claimant":    "legal:Claimant",
    "solicitor":   "legal:Solicitor",
    "counsel":     "legal:Counsel",
    "accused":     "legal:Accused",
    "justice":     "legal:Judge",
    "judge":       "legal:Judge",
    # Legislation
    "legislation": "legal:Legislation",
    "regulation":  "legal:Regulation",
    "amendment":   "legal:Amendment",
    "provision":   "legal:Provision",
    "statute":     "legal:Statute",
    "clause":      "legal:Clause",
    "section":     "legal:Section",
    "act":         "legal:Act",
    # Courts / proceedings
    "tribunal":  "legal:Tribunal",
    "appeal":    "legal:Appeal",
    "judgment":  "legal:CourtCase",
    "decision":  "legal:CourtCase",
    "case":      "legal:CourtCase",
    # Legal concepts
    "injunction": "legal:Injunction",
    "precedent":  "legal:Precedent",
    "principle":  "legal:LegalPrinciple",
    "contract":   "legal:Contract",
    "evidence":   "legal:Evidence",
    "sentence":   "legal:Sentence",
    "offence":    "legal:Offence",
    "offense":    "legal:Offence",
    "damages":    "legal:Damages",
    "hearing":    "legal:Hearing",
    "licence":    "legal:Licence",
    "license":    "legal:Licence",
    "warrant":    "legal:Warrant",
    "penalty":    "legal:Penalty",
    "charge":     "legal:Charge",
    "order":      "legal:Order",
    "test":       "legal:LegalTest",
    "fine":       "legal:Fine",
    # Property / land
    "landlord": "legal:Landlord",
    "property": "legal:Property",
    "tenant":   "legal:Tenant",
    "lease":    "legal:Lease",
    "land":     "legal:Land",
    # Aviation
    "airworthiness": "legal:Airworthiness",
    "aircraft":      "legal:Aircraft",
    "pilot":         "legal:Pilot",
    # Certificates / applications
    "certificate":  "legal:Certificate",
    "application":  "legal:Application",
    "submission":   "legal:Submission",
    "complaint":    "legal:Complaint",
    "commission":   "legal:Commission",
    "authority":    "legal:Authority",
    "minister":     "legal:Minister",
    "secretary":    "legal:Secretary",
    "director":     "legal:Director",
    "motion":       "legal:Motion",
}

LEGAL_ROLES: Dict[str, str] = {
    "presided over by": "legal:presidedBy",
    "represented by":   "legal:representedBy",
    "appealed by":      "legal:appealedBy",
    "decided by":       "legal:decidedBy",
    "presided by":      "legal:presidedBy",
    "brought by":       "legal:broughtBy",
    "heard by":         "legal:heardBy",
    "filed by":         "legal:filedBy",
    "governed by":      "legal:governedBy",
    "interpreted in":   "legal:interpretedIn",
    "pursuant to":      "legal:pursuantTo",
    "applies to":       "legal:appliesTo",
    "subject to":       "legal:subjectTo",
    "defined in":       "legal:definedIn",
    "amended by":       "legal:amendedBy",
    "replaced by":      "legal:replacedBy",
    "repealed by":      "legal:repealedBy",
    "applied for":      "legal:appliedFor",
    "concerns":         "legal:concerns",
    "involves":         "legal:involves",
    "relates to":       "legal:relatesTo",
    "based on":         "legal:basedOn",
    "cited in":         "legal:citedIn",
    "under":            "legal:underProvision",
    "cites":            "legal:cites",
    "overturned":       "legal:overturned",
    "adjourned":        "legal:adjourned",
    "dismissed":        "legal:dismissed",
    "remitted":         "legal:remitted",
    "granted":          "legal:granted",
    "refused":          "legal:refused",
    "allowed":          "legal:allowed",
    "upheld":           "legal:upheld",
    "awarded":          "legal:awarded",
}

LEGAL_DATA_PROPS: Dict[str, str] = {
    "case reference":   "property:caseReference",
    "applicant name":   "property:applicantName",
    "respondent name":  "property:respondentName",
    "decision date":    "property:decisionDate",
    "judgment date":    "property:decisionDate",
    "tribunal finding": "property:tribunalFinding",
    "appeal grounds":   "property:appealGrounds",
    "appeal outcome":   "property:appealOutcome",
    "judge name":       "property:presidingJudge",
    "case number":      "property:caseReference",
    "legal topic":      "property:legalTopic",
    "document type":    "property:documentType",
    "description":      "property:description",
    "jurisdiction":     "property:jurisdiction",
    "precedent":        "property:precedentCited",
    "citation":         "property:caseReference",
    "reasoning":        "property:judicialReasoning",
    "argument":         "property:legalArgument",
    "principle":        "property:legalPrinciple",
    "sentence":         "property:sentenceDetails",
    "regulation":       "property:regulation",
    "parties":          "property:partyNames",
    "statute":          "property:statute",
    "finding":          "property:judicialFinding",
    "outcome":          "property:caseOutcome",
    "penalty":          "property:penaltyAmount",
    "section":          "property:sectionNumber",
    "damages":          "property:damagesAmount",
    "charge":           "property:chargeDescription",
    "result":           "property:result",
    "court":            "property:courtName",
    "date":             "property:decisionDate",
    "fine":             "property:fineAmount",
    "year":             "property:year",
}

# Property category name -> property: URI
PROPERTY_CATEGORY_MAP: Dict[str, str] = {
    "legal_context":         "property:legal_context",
    "legal_interpretation":  "property:legal_interpretation",
    "legal_outcome":         "property:legal_outcome",
    "legal_parties":         "property:legal_parties",
    "legal_timing":          "property:legal_timing",
    "legal_requirement":     "property:legal_requirement",
    "disciplinary_action":   "property:disciplinary_action",
    "legal_definition":      "property:legal_definition",
    "legal_charges":         "property:legal_charges",
    "legal_argument":        "property:legal_argument",
    "legal_appeal":          "property:legal_appeal",
    "legal_reasoning":       "property:legal_reasoning",
    "legal_inquiry":         "property:legal_inquiry",
    "legal_capability":      "property:legal_capability",
    "legal_process":         "property:legal_process",
    "professional_misconduct": "property:professional_misconduct",
    "legal_finding":         "property:legal_finding",
}

# Extracted field name -> data property URI (used for OPTIONAL projections)
FIELD_PROP_MAP: Dict[str, str] = {
    "argument":   "property:legalArgument",
    "reasoning":  "property:judicialReasoning",
    "parties":    "property:partyNames",
    "appeal":     "property:appealDetails",
    "outcome":    "property:caseOutcome",
    "court":      "property:courtName",
    "statute":    "property:statute",
    "principle":  "property:legalPrinciple",
    "judge":      "property:presidingJudge",
    "complaint":  "property:complaintDetails",
    "regulation": "property:regulation",
    "precedent":  "property:precedentCited",
    "finding":    "property:judicialFinding",
    "dateDecided": "property:decisionDate",
    "section":    "property:sectionNumber",
    "charge":     "property:chargeDescription",
    "sentence":   "property:sentenceDetails",
}

QUERY_TYPE_DOC_TYPE: Dict[str, str] = {
    "CaseLaw":             "Case Law",
    "Legislation":         "Statute",
    "AviationLaw":         "Airworthiness Directive",
    "ProfessionalConduct": "Case Law",
    "RegulatoryCompliance": "Legal Document",
    "LegalEntities":       "Legal Document",
    "LegalPrinciples":     "Legal Principle",
    "CourtProcedures":     "Regulation",
}

# spaCy NER label -> (ontology class, data property)
SPACY_NER_MAP: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "PERSON":      ("legal:Person",          None),
    "NORP":        ("legal:Group",           None),
    "FAC":         ("legal:Facility",        None),
    "ORG":         ("legal:Organisation",    None),
    "GPE":         ("legal:Location",        None),
    "LOC":         ("legal:Location",        None),
    "PRODUCT":     ("legal:Product",         None),
    "EVENT":       ("legal:Event",           None),
    "LAW":         ("legal:LegalInstrument", None),
    "DATE":        (None,                    "property:decisionDate"),
    "TIME":        (None,                    "property:time"),
    "PERCENT":     (None,                    "property:percentage"),
    "MONEY":       (None,                    "property:monetaryAmount"),
    "QUANTITY":    (None,                    "property:quantity"),
    "ORDINAL":     (None,                    "property:ordinal"),
    "CARDINAL":    (None,                    "property:cardinal"),
}

# Verb lemma -> ontology predicate URI
_LEGAL_VERB_MAP: Dict[str, str] = {
    "hold":       "legal:holds",        "find":       "legal:finds",
    "decide":     "legal:decides",      "rule":       "legal:rules",
    "determine":  "legal:determines",   "dismiss":    "legal:dismisses",
    "allow":      "legal:allows",       "uphold":     "legal:upholds",
    "overturn":   "legal:overturns",    "remit":      "legal:remits",
    "award":      "legal:awards",       "grant":      "legal:grants",
    "refuse":     "legal:refuses",      "order":      "legal:orders",
    "convict":    "legal:convicts",     "acquit":     "legal:acquits",
    "sentence":   "legal:sentences",    "charge":     "legal:charges",
    "apply":      "legal:appliesTo",    "interpret":  "legal:interprets",
    "define":     "legal:defines",      "construe":   "legal:construes",
    "amend":      "legal:amends",       "repeal":     "legal:repeals",
    "cite":       "legal:cites",        "rely":       "legal:reliesOn",
    "argue":      "legal:argues",       "submit":     "legal:submits",
    "contend":    "legal:contends",     "allege":     "legal:alleges",
    "appeal":     "legal:appeals",      "lodge":      "legal:lodges",
    "file":       "legal:files",        "bring":      "legal:brings",
    "require":    "legal:requires",     "provide":    "legal:provides",
    "state":      "legal:states",       "enact":      "legal:enacts",
    "prescribe":  "legal:prescribes",   "prohibit":   "legal:prohibits",
    "permit":     "legal:permits",      "authorize":  "legal:authorizes",
    "authorise":  "legal:authorizes",   "delegate":   "legal:delegates",
    "consider":   "legal:considers",    "examine":    "legal:examines",
    "review":     "legal:reviews",      "affirm":     "legal:affirms",
    "reverse":    "legal:reverses",     "vary":       "legal:varies",
    "impose":     "legal:imposes",      "reduce":     "legal:reduces",
    "increase":   "legal:increases",    "direct":     "legal:directs",
    "instruct":   "legal:instructs",    "contain":    "legal:contains",
    "include":    "legal:includes",     "exclude":    "legal:excludes",
    "cover":      "legal:covers",       "relate":     "legal:relatesTo",
    "concern":    "legal:concerns",     "involve":    "legal:involves",
    "constitute": "legal:constitutes",  "establish":  "legal:establishes",
    "satisfy":    "legal:satisfies",    "breach":     "legal:breaches",
    "comply":     "legal:compliesWith", "contravene": "legal:contravenes",
    "infringe":   "legal:infringes",
}

_CLAUSE_HEAD_DEPS = {"ROOT", "relcl", "advcl", "xcomp", "ccomp", "conj", "acl"}
_SUBJ_DEPS        = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS         = {"dobj", "pobj", "attr", "ccomp", "xcomp", "oprd", "acomp"}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QueryIntent(Enum):
    SELECT_CASE_FACTS     = auto()
    SELECT_CASE_OUTCOME   = auto()
    SELECT_LEGAL_ARGUMENT = auto()
    SELECT_PARTIES        = auto()
    SELECT_LEGISLATION    = auto()
    SELECT_DEFINITION     = auto()
    SELECT_TIMING         = auto()
    SELECT_REQUIREMENTS   = auto()
    SELECT_APPEAL         = auto()
    SELECT_CHARGES        = auto()
    SELECT_AVIATION       = auto()
    SELECT_GENERAL        = auto()
    ASK_BOOLEAN           = auto()
    ASK_EXISTS            = auto()
    DESCRIBE_ENTITY       = auto()
    CONSTRUCT_GRAPH       = auto()


class QuestionType(Enum):
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
    Augmented with legal-specific metadata extracted from the source question.
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
    legal_meta:           Dict[str, Any]            = field(default_factory=dict)
    legal_query_type:     str                       = "CaseLaw"
    property_category:    str                       = "legal_context"
    jurisdiction:         str                       = "federal"
    extracted_fields:     List[str]                 = field(default_factory=list)


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
# Stage 1 — Legal metadata extraction
# ---------------------------------------------------------------------------

def extract_legal_metadata(text: str) -> Dict[str, Any]:
    """
    Extract structured legal metadata from NLQ text:
        case_number  — "[YYYY] COURT NNN" citation
        year         — four-digit year from citation
        court        — court abbreviation from citation
        applicant    — party name before "v"
        respondent   — party name after "v"
        sections     — section numbers referenced
        acts         — act/regulation names referenced
        jurisdiction — inferred jurisdiction string
    """
    meta: Dict[str, Any] = {}

    cite = _CASE_CITATION_RE.search(text)
    if cite:
        meta["year"]        = cite.group(1)
        meta["court"]       = cite.group(2)
        meta["case_number"] = f"[{cite.group(1)}] {cite.group(2)} {cite.group(3)}"

    party = _PARTY_RE.search(text)
    if party:
        meta["applicant"]  = party.group(1).strip()
        meta["respondent"] = party.group(2).strip()

    sections = _SECTION_RE.findall(text)
    if sections:
        meta["sections"] = sections

    acts = [a.strip() for a in _ACT_RE.findall(text) if len(a.strip()) > 5]
    if acts:
        meta["acts"] = acts

    meta["jurisdiction"] = _jurisdiction_from_question(text)
    return meta


# ---------------------------------------------------------------------------
# Stage 2 — Query type, property category, and intent classification
# ---------------------------------------------------------------------------

def classify_legal_query_type(question: str, legal_meta: Dict[str, Any]) -> str:
    """
    Classify into one of the 8 legal query types.
    Patterns are ordered from most specific to least specific.
    """
    tl = question.lower()

    patterns: List[Tuple[str, str]] = [
        (r'airworthiness\s+directive|ad/[a-z]+\s*\d+|amendment.*aeroplane'
         r'|aircraft.*regulation|civil aviation',
         "AviationLaw"),
        (r'professional\s+(conduct|misconduct|standard)|solicitor.*disciplin'
         r'|law\s+society|legal\s+practitioner.*disciplin'
         r'|unsatisfactory\s+professional|bill\s+of\s+costs.*privilege'
         r'|rules\s+of\s+practice.*solicitor',
         "ProfessionalConduct"),
        (r'regulatory\s+compliance|compliance\s+record|notification\s+requirement'
         r'|mutual\s+recognition|what\s+are\s+the\s+requirements?\s+for'
         r'|requirements?\s+for\s+(display|notif|cabin\s+crew|hydraulic'
         r'|pregnancy|corrugated|persons\s+advis)',
         "RegulatoryCompliance"),
        (r'principle[s]?\s+(governing|underlying|for|of\s+the\s+exercise)'
         r'|general\s+principle|statement\s+of\s+principles'
         r'|definition\s+of\s+[a-z].*(?:cumulative|equivalent\s+dose'
         r'|malignant|hypopitu)',
         "LegalPrinciples"),
        (r'rule\s+\d+[A-Z]?\s+of|when\s+did.*rules?\s+(?:come\s+into\s+)?(?:effect|force)'
         r'|process\s+for.*assessment.*funding|road.*amendment.*rules'
         r'|radiocommunications.*rules|fisheries.*rules'
         r'|criteria\s+for\s+(different\s+)?categories',
         "CourtProcedures"),
        (r'who\s+(is|are|were|was)\s+(the\s+)?(plaintiff|defendant|appellant'
         r'|respondent|applicant|parties|solicitor|counsel|judge)'
         r'|when\s+did\s+(the\s+)?(plaintiff|defendant)'
         r'|what\s+claims?\s+are|who\s+are\s+the\s+parties'
         r'|avenues?\s+for\s+an\s+applicant',
         "LegalEntities"),
    ]

    for pattern, qtype in patterns:
        if re.search(pattern, tl):
            return qtype

    # Legislation — check before CaseLaw to catch "Under the X Act..." patterns
    is_legislation = re.search(
        r'under\s+(?:the\s+)?[A-Z][a-zA-Z\s]+(?:Act|Regulation|Rules?|Code|Order)'
        r'|pursuant\s+to\s+(?:the\s+)?[A-Z]'
        r'|in\s+the\s+context\s+of\s+(?:the\s+)?[A-Z][a-zA-Z\s]+(?:Act|Regulation)'
        r'|what\s+(?:does|do|did)\s+(?:the\s+)?[A-Z][a-zA-Z\s]+(?:Act|Regulation|Rules?)'
        r'|act\s+\d{4}|regulation[s]?\s+\d{4}|section\s+\d+[A-Z]?\s+of\s+the',
        tl,
    )
    is_case = _CASE_CITATION_RE.search(question) and re.search(r'in\s+the\s+case\s+of', tl)
    if is_legislation and not is_case:
        return "Legislation"

    if _CASE_CITATION_RE.search(question) or re.search(
        r'in\s+the\s+case\s+of|the\s+court\s+(held|found|decided|ruled'
        r'|determined|dismissed|allowed)|court\s+of\s+appeal', tl
    ):
        return "CaseLaw"

    return "CaseLaw"


def classify_property_category(question: str, legal_meta: Dict[str, Any],
                                query_type: str) -> str:
    """
    Classify the property category that drives the primary result binding in SPARQL.
    Patterns are ordered from most specific to least specific.
    """
    tl = question.lower()

    patterns: List[Tuple[str, str]] = [
        (r'when\s+did|what\s+(date|time|year)|how\s+long|duration'
         r'|come\s+into\s+effect|commenced|took\s+effect',
         "legal_timing"),
        (r'who\s+(is|are|were|was)\s+(the\s+)?(plaintiff|defendant'
         r'|appellant|respondent|applicant|parties|solicitor|judge|counsel)'
         r'|what\s+.*parties',
         "legal_parties"),
        (r'what\s+(was|were|is|are)\s+the\s+(charge|offence|offense)'
         r'|charged\s+with|pleaded\s+guilty|convicted',
         "legal_charges"),
        (r'what\s+(was|were|is|are)\s+the\s+(outcome|decision|result|order|verdict'
         r'|judgment|holding|ruling)|court\s+(held|found|decided|ruled|dismissed'
         r'|allowed|upheld|overturned)',
         "legal_outcome"),
        (r'what\s+(was|were|is|are)\s+the\s+(argument|ground|submission|reason'
         r'|basis|claim)|argued\s+that|submitted\s+that',
         "legal_argument"),
        (r'what\s+(was|were|is|are)\s+the\s+(appeal|ground.*appeal'
         r'|appeal.*ground)|did\s+the\s+court.*appeal',
         "legal_appeal"),
        (r'what\s+(is|are|was|were)\s+the\s+(definition|meaning|interpretation'
         r'|construed|defined)',
         "legal_definition"),
        (r'disciplinar|misconduct|sanction|struck\s+off|reprimand'
         r'|professional.*conduct',
         "disciplinary_action"),
        (r'(require|must|shall|obligation|duty|power|right|entitled'
         r'|permitted|prohibited)\s',
         "legal_requirement"),
        (r'how\s+is\s+(the\s+)?term'
         r'|what\s+(does|did)\s+(the\s+)?court\s+(interpret|construe|hold|find|decide|rule|view|consider|treat)'
         r'|court\s+(interpretation|construction|view|stance|approach)'
         r'|what\s+(is|was|were)\s+the\s+(test|principle|rule|basis|standard'
         r'|interpretation|definition|meaning|scope|effect|construction'
         r'|circumstances?|considerations?|grounds?|issues?|actions?|measures?'
         r'|changes?|arguments?|obligations?|claims?|certifications?'
         r'|consequences?|relationship|assessment|cause\s+of)'
         r'|what\s+changes?\s+were|what\s+action[s]?\s+(does|did|were)'
         r'|how\s+is\s+|how\s+does\s+|how\s+was\s+|why\s+(was|is|did|does)'
         r'|consequences?\s+of|what\s+led\s+to|what\s+amount[s]?\s+to'
         r'|what\s+were\s+the\s+(key|main|two|three|relevant|particular)',
         "legal_interpretation"),
    ]

    for pattern, category in patterns:
        if re.search(pattern, tl):
            return category

    return "legal_context"


_QT_PATTERNS: List[Tuple[str, QuestionType]] = [
    (r"\bhow\s+many\b",                                  QuestionType.WH_HOW_MANY),
    (r"\bhow\s+much\b",                                  QuestionType.WH_HOW_MUCH),
    (r"\bwhy\b",                                         QuestionType.WH_WHY),
    (r"\bhow\b",                                         QuestionType.WH_HOW),
    (r"\bwhen\b",                                        QuestionType.WH_WHEN),
    (r"\bwhere\b",                                       QuestionType.WH_WHERE),
    (r"\bwho\b|\bwhom\b",                                QuestionType.WH_WHO),
    (r"\bwhich\b",                                       QuestionType.WH_WHICH),
    (r"\bwhat\s+is\s+(a|an|the)\b",                      QuestionType.DEFINITION),
    (r"\bwhat\b|\bwhat\s+were\b|\bwhat\s+was\b",         QuestionType.WH_WHAT),
    (r"\bis\s+there\b|\bare\s+there\b|\bexists?\b",      QuestionType.EXISTENCE),
    (r"^(did|does|do|is|are|was|were|has|have|had|will|would|can|could|should)\b",
                                                         QuestionType.POLAR),
]


def classify_question_type(question: str) -> QuestionType:
    tl = question.lower().strip()
    for pat, qt in _QT_PATTERNS:
        if re.search(pat, tl):
            return qt
    return QuestionType.WH_WHAT


_INTENT_PATTERNS: List[Tuple[str, QueryIntent]] = [
    (r"\bwhen\s+(did|was|were|does|is)\b|\bwhat\s+(date|year|time)\b"
     r"|\bcome\s+into\s+(effect|force)\b|\btook\s+effect\b",
     QueryIntent.SELECT_TIMING),
    (r"\bwho\s+(is|are|was|were)\s+(the\s+)?(plaintiff|defendant|appellant"
     r"|respondent|applicant|parties|solicitor|counsel|judge)\b"
     r"|\bparties\s+to\b|\bpresiding\s+judge\b",
     QueryIntent.SELECT_PARTIES),
    (r"\b(charge[sd]?|offence[s]?|offense[s]?|plead(ed)?\s+guilty"
     r"|convicted|indictment|crime[s]?)\b",
     QueryIntent.SELECT_CHARGES),
    (r"\b(definition|meaning|interpret(ation|ed)|constru(ed|ing)|defined)\b"
     r"|\bwhat\s+(is|are)\s+(a|an|the)\b",
     QueryIntent.SELECT_DEFINITION),
    (r"\b(require[sd]?|obligation[s]?|duty|duties|must|shall|power[s]?"
     r"|entitled|permitted|prohibited|right\s+to)\b",
     QueryIntent.SELECT_REQUIREMENTS),
    (r"\b(appeal(ed|ing|s)?|ground[s]?\s+of\s+appeal|leave\s+to\s+appeal"
     r"|court\s+of\s+appeal)\b",
     QueryIntent.SELECT_APPEAL),
    (r"\b(airworthiness\s+directive|ad/[a-z0-9]+|aircraft|aeroplane"
     r"|civil\s+aviation\s+regulations?)\b",
     QueryIntent.SELECT_AVIATION),
    (r"\b(outcome|decision|result|order|verdict|judgment|holding|ruling"
     r"|court\s+(held|found|decided|ruled|dismissed|allowed|upheld"
     r"|overturned)|what\s+did\s+the\s+court)\b",
     QueryIntent.SELECT_CASE_OUTCOME),
    (r"\b(argument|ground[s]?|submission|claim[s]?|contention|asserted"
     r"|alleged|submitted|argued)\b",
     QueryIntent.SELECT_LEGAL_ARGUMENT),
    (r"\b(act\s+\d{4}|regulations?\s+\d{4}|section\s+\d+|under\s+the\s+[A-Z]"
     r"|pursuant\s+to|provides?\s+that|states?\s+that)\b",
     QueryIntent.SELECT_LEGISLATION),
    (r"\bexists?\b|\bis\s+there\b|\bare\s+there\b",     QueryIntent.ASK_EXISTS),
    (r"^(did|does|do|is|are|was|were|has|have|had|will|would|can|could|should)\b",
     QueryIntent.ASK_BOOLEAN),
    (r"^(describe|what\s+is\s+(a|an|the)\s+[a-z]+\s+[a-z]+|tell\s+me\s+about)\b",
     QueryIntent.DESCRIBE_ENTITY),
    (r"\b(show\s+me\s+(all|the\s+full)|full\s+(case|graph|network))\b",
     QueryIntent.CONSTRUCT_GRAPH),
]


def classify_intent(question: str) -> QueryIntent:
    tl = question.lower().strip()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, tl):
            return intent
    return QueryIntent.SELECT_CASE_FACTS


def _infer_focus_var(qt: QuestionType, intent: QueryIntent,
                     entities: List[Entity]) -> str:
    focus_by_qt: Dict[QuestionType, str] = {
        QuestionType.WH_WHY:      "?reason",
        QuestionType.WH_WHEN:     "?date",
        QuestionType.WH_WHERE:    "?location",
        QuestionType.WH_WHO:      "?party",
        QuestionType.WH_HOW_MANY: "?count",
        QuestionType.WH_HOW_MUCH: "?amount",
        QuestionType.WH_HOW:      "?process",
        QuestionType.DEFINITION:  "?definition",
    }
    if qt in (QuestionType.POLAR, QuestionType.EXISTENCE):
        return ""
    if qt in focus_by_qt:
        return focus_by_qt[qt]

    focus_by_intent: Dict[QueryIntent, str] = {
        QueryIntent.SELECT_CASE_OUTCOME:   "?outcome",
        QueryIntent.SELECT_LEGAL_ARGUMENT: "?argument",
        QueryIntent.SELECT_PARTIES:        "?party",
        QueryIntent.SELECT_LEGISLATION:    "?provision",
        QueryIntent.SELECT_DEFINITION:     "?definition",
        QueryIntent.SELECT_TIMING:         "?date",
        QueryIntent.SELECT_REQUIREMENTS:   "?requirement",
        QueryIntent.SELECT_APPEAL:         "?appealDetails",
        QueryIntent.SELECT_CHARGES:        "?charge",
        QueryIntent.SELECT_AVIATION:       "?directive",
        QueryIntent.SELECT_CASE_FACTS:     "?result",
        QueryIntent.SELECT_GENERAL:        "?result",
    }
    if intent in focus_by_intent:
        return focus_by_intent[intent]
    return _varname(entities[0].uri) if entities else "?result"


def _infer_extracted_fields(question: str, intent: QueryIntent,
                             property_category: str) -> List[str]:
    """
    Select relevant projected fields based on question content and property category.
    Returned list is deduplicated with order preserved.
    """
    tl     = question.lower()
    fields: List[str] = []

    keyword_fields: List[Tuple[str, str]] = [
        (r'\bargument\b|\bclaim\b|\bsubmission\b|\bcontention\b', "argument"),
        (r'\breason\b|\breach\b|\bconsider\b|\banalys\b|\bfind\b',  "reasoning"),
        (r'\boutcome\b|\bdecision\b|\bresult\b|\border\b|\bverdict\b', "outcome"),
        (r'\bpart(y|ies)\b|\bplaintiff\b|\bdefendant\b|\bappellant\b'
         r'|\brespondent\b|\bapplicant\b', "parties"),
        (r'\bjudge\b|\bjustice\b|\bpresid\b', "judge"),
        (r'\bstatute\b|\bact\b|\blegislation\b|\bprovision\b|\bsection\b', "statute"),
        (r'\bprinciple\b|\btest\b|\brule\b', "principle"),
        (r'\bappeal\b|\bground\b', "appeal"),
        (r'\bregulation\b|\bdirective\b', "regulation"),
        (r'\bcharge\b|\boffence\b|\bguilty\b', "charge"),
        (r'\bdate\b|\bwhen\b|\btime\b|\byear\b', "dateDecided"),
        (r'\bcourt\b|\btribunal\b', "court"),
        (r'\bprecedent\b|\bcited\b|\bcases?\s+cited\b', "precedent"),
    ]
    for pattern, f in keyword_fields:
        if re.search(pattern, tl):
            fields.append(f)

    if not fields:
        defaults: Dict[str, List[str]] = {
            "legal_context":        ["argument", "reasoning", "parties"],
            "legal_interpretation": ["argument", "reasoning"],
            "legal_outcome":        ["outcome", "argument", "reasoning"],
            "legal_parties":        ["parties", "judge"],
            "legal_timing":         ["dateDecided"],
            "legal_requirement":    ["statute", "requirement"],
            "legal_charges":        ["charge", "outcome", "reasoning"],
            "legal_appeal":         ["appeal", "outcome", "reasoning"],
            "legal_argument":       ["argument", "reasoning"],
            "legal_definition":     ["statute", "principle"],
        }
        fields = defaults.get(property_category, ["argument", "reasoning"])

    return list(dict.fromkeys(fields))


# ---------------------------------------------------------------------------
# Stage 3 — Entity, relation, and data property extraction
# ---------------------------------------------------------------------------

def _run_spacy_ner(text: str) -> List[SpacyNERHit]:
    doc  = NLP(text)
    hits: List[SpacyNERHit] = []
    for ent in doc.ents:
        if ent.label_ not in SPACY_NER_MAP:
            continue
        onto_class, data_prop = SPACY_NER_MAP[ent.label_]
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
    Two-pass extraction: legal domain dictionary (longest match first),
    then spaCy NER class hits for spans not already covered.
    """
    tl         = text.lower()
    seen_spans: List[Tuple[int, int]] = []
    found:      List[Entity]          = []

    for term in sorted(LEGAL_CLASSES, key=len, reverse=True):
        start = 0
        while True:
            pos = tl.find(term, start)
            if pos == -1:
                break
            end = pos + len(term)
            if not any(s <= pos < e or s < end <= e for s, e in seen_spans):
                seen_spans.append((pos, end))
                prefix = tl[max(0, pos - 20): pos]
                neg    = bool(re.search(r"\b(no|not|without|never|cannot|can't)\b", prefix))
                found.append(Entity(surface=term, uri=LEGAL_CLASSES[term],
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


def extract_data_properties(text: str, legal_meta: Dict[str, Any]) -> List[DataProperty]:
    """
    Three-pass extraction:
        1. Surface dictionary lookup.
        2. spaCy NER numeric/temporal hits (DATE, MONEY, CARDINAL etc.).
        3. Legal metadata (case citation, section numbers).
    """
    tl        = text.lower()
    seen_uris: set = set()
    dps:       List[DataProperty] = []

    for surf in sorted(LEGAL_DATA_PROPS, key=len, reverse=True):
        uri = LEGAL_DATA_PROPS[surf]
        if surf in tl and uri not in seen_uris:
            seen_uris.add(uri)
            m = re.search(re.escape(surf) + r"[\s:=]+([\w.\-#\[\]]+)", tl)
            dps.append(DataProperty(surface=surf, uri=uri,
                                    value=m.group(1).strip() if m else None))

    for hit in _run_spacy_ner(text):
        if hit.is_data_value and hit.data_prop not in seen_uris:
            seen_uris.add(hit.data_prop)
            dps.append(DataProperty(surface=hit.surface, uri=hit.data_prop, value=hit.surface))

    for meta_key, dp_uri, val_fn in [
        ("case_number", "property:caseReference", lambda v: v),
        ("year",        "property:year",           lambda v: v),
    ]:
        if meta_key in legal_meta and dp_uri not in seen_uris:
            seen_uris.add(dp_uri)
            val = val_fn(legal_meta[meta_key])
            dps.append(DataProperty(surface=str(val), uri=dp_uri, value=str(val)))

    if "sections" in legal_meta:
        for sec in legal_meta["sections"]:
            uri = "property:sectionNumber"
            if uri not in seen_uris:
                seen_uris.add(uri)
                dps.append(DataProperty(surface=f"s{sec}", uri=uri, value=sec))

    return dps


def _extract_surface_relations(text: str) -> List[Relation]:
    tl   = text.lower()
    seen: set = set()
    found: List[Relation] = []

    for surf in sorted(LEGAL_ROLES, key=len, reverse=True):
        if surf in tl and surf not in seen:
            seen.add(surf)
            pos    = tl.find(surf)
            prefix = tl[max(0, pos - 25): pos]
            neg    = bool(re.search(
                r"\b(not|doesn't|don't|cannot|can't|never|without)\b", prefix
            ))
            found.append(Relation(surface=surf, uri=LEGAL_ROLES[surf],
                                  source="surface_dict", negated=neg))
    return found


# ---------------------------------------------------------------------------
# Stage 4 — SPO extraction from dependency parse
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


def extract_spo_triples(text: str, entities: List[Entity]) -> List[SPOTriple]:
    doc        = NLP(text)
    triples:   List[SPOTriple]  = []
    tok_to_uri: Dict[int, str]  = {}
    seen:       set              = set()

    for ent in entities:
        for tok in doc:
            if tok.text.lower() in ent.surface or ent.surface in tok.text.lower():
                tok_to_uri.setdefault(tok.i, ent.uri)

    for tok in doc:
        if tok.pos_ != "VERB" or tok.dep_ not in _CLAUSE_HEAD_DEPS:
            continue
        lemma    = tok.lemma_.lower()
        pred_uri = _LEGAL_VERB_MAP.get(lemma)
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
# Stage 5 — Question Logical Form
# ---------------------------------------------------------------------------

def _classify_triple(triple: SPOTriple, qt: QuestionType, focus_var: str) -> str:
    if qt in (QuestionType.POLAR, QuestionType.EXISTENCE):
        return "presupposition"

    def _v(val: str, is_uri: bool) -> str:
        return _varname(val) if is_uri else (val if val.startswith("?") else _tok_to_var(val))

    if focus_var and focus_var in (_v(triple.subj, triple.subj_is_uri),
                                   _v(triple.obj,  triple.obj_is_uri)):
        return "restriction"
    return "presupposition"


def build_logical_form(question: str) -> QuestionLogicalForm:
    legal_meta    = extract_legal_metadata(question)
    legal_qtype   = classify_legal_query_type(question, legal_meta)
    jurisdiction  = legal_meta.get("jurisdiction", "federal")

    all_entities  = extract_entities(question)
    data_props    = extract_data_properties(question, legal_meta)
    ner_hits      = _run_spacy_ner(question)

    qt            = classify_question_type(question)
    intent        = classify_intent(question)
    focus_var     = _infer_focus_var(qt, intent, all_entities)
    prop_category = classify_property_category(question, legal_meta, legal_qtype)
    ext_fields    = _infer_extracted_fields(question, intent, prop_category)

    spo_triples   = extract_spo_triples(question, all_entities)

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
        if re.search(r"\bboth\b|\band\b|\bas well as\b", tl) else []
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
        ner_hits             = ner_hits,
        intent               = intent,
        disjunctive_entities = disj_ent,
        connective           = "OR" if disjunctions else "AND",
        negated              = bool(negations),
        legal_meta           = legal_meta,
        legal_query_type     = legal_qtype,
        property_category    = prop_category,
        jurisdiction         = jurisdiction,
        extracted_fields     = ext_fields,
    )


# ---------------------------------------------------------------------------
# Stage 6 — DL/OWL2 expression
# ---------------------------------------------------------------------------

def lf_to_dl(lf: QuestionLogicalForm) -> DLExpression:
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
            p = f"legal:relatedTo  # unresolved verb: {t.pred}"
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
        dl.role_assertions.append(
            (_varname(lf.entities[0].uri), "legal:relatedTo", _varname(lf.entities[1].uri))
        )

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
# Stage 7 — SPARQL generation
# ---------------------------------------------------------------------------

class SPARQLBuilder:

    def build(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        try:
            return self._legal_select(lf, dl)
        except Exception as exc:
            return self._fallback(lf, exc)

    def _nlq_annotation(self, lf: QuestionLogicalForm, indent: str = "  ") -> str:
        """
        Embed NLQ as rdfs:comment + skos:definition on _:queryMeta.
        Wrapped in OPTIONAL for triplestore portability.
        Records intent, question type, legal query type, property category,
        jurisdiction, and focus variable as typed literals.
        """
        q   = _escape(lf.raw_question)
        ns  = _namespace_for(lf.jurisdiction).rstrip("/")
        it  = lf.intent.name if lf.intent else "UNKNOWN"
        qt  = lf.question_type.name
        lqt = lf.legal_query_type
        pc  = lf.property_category
        jur = lf.jurisdiction
        fv  = lf.focus_var or ""
        return (
            f"{indent}OPTIONAL {{\n"
            f"{indent}  _:queryMeta rdf:type          <{ns}/SPARQLQuery> ;\n"
            f'{indent}              rdfs:comment      "{q}"@en ;\n'
            f'{indent}              skos:definition   "{q}"@en ;\n'
            f'{indent}              <{ns}/property/intent>           "{it}"^^xsd:string ;\n'
            f'{indent}              <{ns}/property/questionType>     "{qt}"^^xsd:string ;\n'
            f'{indent}              <{ns}/property/legalQueryType>   "{lqt}"^^xsd:string ;\n'
            f'{indent}              <{ns}/property/propertyCategory> "{pc}"^^xsd:string ;\n'
            f'{indent}              <{ns}/property/jurisdiction>     "{jur}"^^xsd:string ;\n'
            f'{indent}              <{ns}/property/focusVariable>    "{fv}"^^xsd:string .\n'
            f"{indent}}}"
        )

    def _legal_select(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        """
        Build a SPARQL SELECT matching the legal_queries dataset pattern:

            SELECT ?result [?field2 ?field3] WHERE {
              ?legal_entity property:<property_category> ?result .
              ?legal_entity property:jurisdiction         "<jurisdiction>" .
              ?legal_entity property:document_type        "<document_type>" .
              ?legal_entity property:legal_topic          "<legal_topic>" .
              ?legal_entity property:description          "<question>"@en .
              [?legal_entity property:caseReference "<citation>" .]
              [?legal_entity property:sectionNumber "<section>" .]
              [?legal_entity property:statute "<act_name>" .]
              [OPTIONAL { ?legal_entity property:<field> ?field . }]
              [FILTER NOT EXISTS { ... }]
              [... UNION ...]
              [OPTIONAL { _:queryMeta ... }]
            }
        """
        q         = _escape(lf.raw_question)
        prop_cat  = PROPERTY_CATEGORY_MAP.get(lf.property_category,
                                               f"property:{lf.property_category}")
        doc_type  = QUERY_TYPE_DOC_TYPE.get(lf.legal_query_type, "Legal Document")

        intent_topic: Dict[QueryIntent, str] = {
            QueryIntent.SELECT_CHARGES:      "criminal",
            QueryIntent.SELECT_APPEAL:       "procedural",
            QueryIntent.SELECT_AVIATION:     "aviation",
            QueryIntent.SELECT_LEGISLATION:  "administrative",
            QueryIntent.SELECT_DEFINITION:   "general_law",
            QueryIntent.SELECT_REQUIREMENTS: "administrative",
            QueryIntent.SELECT_CASE_OUTCOME: "civil",
            QueryIntent.SELECT_PARTIES:      "civil",
            QueryIntent.SELECT_TIMING:       "general_law",
        }
        legal_topic = intent_topic.get(lf.intent, "general_law")

        fv         = lf.focus_var or "?result"
        extra_vars = list(dict.fromkeys(
            f"?{FIELD_PROP_MAP.get(f, f'property:{f}').split(':')[1]}"
            for f in lf.extracted_fields
            if f != "argument"
        ))[:3]
        select_vars = " ".join([fv] + extra_vars)

        lines = [_build_prefixes(lf.jurisdiction, lf.legal_query_type)]
        lines.append(f"SELECT {select_vars} WHERE {{")
        lines.append(f"  ?legal_entity {prop_cat} {fv} .")
        lines.append(f'  ?legal_entity property:jurisdiction  "{lf.jurisdiction}" .')
        lines.append(f'  ?legal_entity property:document_type "{doc_type}" .')
        lines.append(f'  ?legal_entity property:legal_topic   "{legal_topic}" .')
        lines.append(f'  ?legal_entity property:description   "{q}"@en .')

        if "case_number" in lf.legal_meta:
            ref = _escape(lf.legal_meta["case_number"])
            lines.append(f'  ?legal_entity property:caseReference "{ref}" .')

        if lf.legal_meta.get("acts"):
            act = _escape(lf.legal_meta["acts"][0])
            lines.append(f'  ?legal_entity property:statute "{act}" .')

        if lf.legal_meta.get("sections"):
            lines.append(f'  ?legal_entity property:sectionNumber "{lf.legal_meta["sections"][0]}" .')

        for f in lf.extracted_fields:
            prop = FIELD_PROP_MAP.get(f)
            if prop:
                var = f"?{prop.split(':')[1]}"
                if var != fv:
                    lines.append(f"  OPTIONAL {{ ?legal_entity {prop} {var} . }}")

        for s, p, o in dl.negation_assertions:
            lines.append(f"  FILTER NOT EXISTS {{ {s} {p} {o} . }}")

        if dl.union_groups:
            parts = [
                "{ " + " ".join(f"{s} {p} {o} ." for s, p, o in grp) + " }"
                for grp in dl.union_groups
            ]
            lines.append("  " + "\n  UNION\n  ".join(parts))

        lines.append(self._nlq_annotation(lf))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _fallback(self, lf: QuestionLogicalForm, exc: Exception) -> str:
        q = _escape(lf.raw_question)
        return (
            _build_prefixes(lf.jurisdiction, lf.legal_query_type)
            + f"# ERROR: {exc}\n"
            + "SELECT ?result WHERE {\n"
            + "  ?legal_entity property:legal_context ?result .\n"
            + f'  ?legal_entity property:description "{q}"@en .\n'
            + self._nlq_annotation(lf) + "\n"
            + "}\n"
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
    and NER breakdowns per label.
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
        "Legal Query Type":   lf.legal_query_type,
        "Property Category":  lf.property_category,
        "Jurisdiction":       lf.jurisdiction,
        "Extracted Fields":   str(lf.extracted_fields),
        "Case Reference":     lf.legal_meta.get("case_number", ""),
        "Year":               lf.legal_meta.get("year", ""),
        "Court":              lf.legal_meta.get("court", ""),
        "Applicant":          lf.legal_meta.get("applicant", ""),
        "Respondent":         lf.legal_meta.get("respondent", ""),
        "Sections":           str(lf.legal_meta.get("sections", [])),
        "Acts Referenced":    str(lf.legal_meta.get("acts", [])),
        "SPO Triples":        str([(t.subj, t.pred, t.obj,
                                    "NEG" if t.negated else "",
                                    t.dep_path) for t in lf.spo_triples]),
        "Presuppositions":    str([(t.subj, t.pred, t.obj) for t in lf.presuppositions]),
        "Restrictions":       str([(t.subj, t.pred, t.obj) for t in lf.restrictions]),
        "Boolean Logic":      f"connective={lf.connective}  negated={lf.negated}",
        "Quantifier":         lf.quantifier,
        "DL Role Assertions": str(dl.role_assertions),
        "DL Negations":       str(dl.negation_assertions),
        "DL N-ary Types":     str([na.nary_type.name for na in dl.nary_assertions]),
        "OWL2 Class Exprs":   str(dl.class_exprs),
        "NER_PERSON":         str(ner_by_label.get("PERSON",   [])),
        "NER_ORG":            str(ner_by_label.get("ORG",      [])),
        "NER_LAW":            str(ner_by_label.get("LAW",      [])),
        "NER_DATE":           str(ner_by_label.get("DATE",     [])),
        "NER_GPE":            str(ner_by_label.get("GPE",      [])),
        "NER_MONEY":          str(ner_by_label.get("MONEY",    [])),
        "NER_CARDINAL":       str(ner_by_label.get("CARDINAL", [])),
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _find_question_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() == "question":
            return col
    raise ValueError(f"No 'question' column found. Available columns: {list(df.columns)}")


def process_excel(input_path: str | Path,
                  sheet_name: str | None = None) -> List[Dict[str, Any]]:
    """Load an Excel file and run the pipeline against every non-empty question row."""
    kwargs = {"sheet_name": sheet_name} if sheet_name else {}
    df     = pd.read_excel(input_path, **kwargs)
    q_col  = _find_question_column(df)

    results: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        q = str(row[q_col]).strip()
        if not q or q.lower() == "nan":
            continue
        try:
            r = nlq_to_sparql(q)
            r["row_index"]          = row.get("row_index",       idx)
            r["source_sparql"]      = row.get("sparql_query",    "")
            r["source_query_type"]  = row.get("query_type",      "")
            r["legal_topic_src"]    = row.get("legal_topic",     "")
            r["document_type_src"]  = row.get("document_type",   "")
            r["case_reference_src"] = row.get("case_reference",  "")
            r["law_reference_src"]  = row.get("law_reference",   "")
        except Exception as exc:  # noqa: BLE001
            r = {
                "Question":     q,
                "row_index":    row.get("row_index", idx),
                "Intent":       "ERROR",
                "SPARQL Query": f"# Error: {exc}",
            }
        results.append(r)

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
        description="Convert natural language Australian legal queries to SPARQL."
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
        print(f"\nQ   : {r['Question'][:100]}")
        print(f"  QType    : {r['Question Type']}  |  FocusVar: {r['Focus Variable']}")
        print(f"  Intent   : {r['Intent']}")
        print(f"  LegalQType: {r['Legal Query Type']}  |  PropCat: {r['Property Category']}")
        print(f"  Jur      : {r['Jurisdiction']}  |  Court: {r['Court']}"
              f"  |  Year: {r['Year']}  |  Ref: {r['Case Reference']}")
        print(f"  Fields   : {r['Extracted Fields']}")
        print(f"\n  SPARQL:\n{r['SPARQL Query']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
