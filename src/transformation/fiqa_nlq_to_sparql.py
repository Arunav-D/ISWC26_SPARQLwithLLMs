"""
fi_nl_to_sparql.py
==================
NLQ -> Question Logical Form -> DL/OWL2 -> SPARQL

Converts natural language questions about financial services into SPARQL queries
against a financial industry domain ontology.

Pipeline stages:
    1. Linguistic analysis   — entity, relation, and data property extraction via
                               dictionary lookup and spaCy NER + dependency parse
    2. Question classification — type (WH, polar, existence) and intent
    3. SPO triple extraction  — subject/predicate/object from dependency graph
    4. Question Logical Form  — presuppositions, restrictions, negations, quantifiers
    5. DL/OWL2 expression     — concept/role/data assertions, negation, union groups
    6. SPARQL generation      — SELECT with projected fields, OPTIONAL blocks,
                               FILTER NOT EXISTS for negation, NLQ annotation

Domain coverage:
    Query types  : BankingOperations, FinancialReporting, InvestmentAnalysis,
                   RegulatoryCompliance, RiskManagement, TechnologyAndFintech,
                   TradingOperations, MarketData, CorporateFinance
    Institutes   : Federal Reserve, HKMA, EU
    Regulations  : Federal Reserve Regulations H/K/L/O/W/Y/YY,
                   HKMA Code of Banking Practice, HKMA AML/CFT Ordinance,
                   EU MiFID/CRR instruments

Namespace scheme (selected dynamically from institute):
    fi:       <https://www.federalreserve.gov/ontology/>
    fi:       <https://www.hkma.gov.hk/ontology/>
    fi:       <https://ec.europa.eu/finance/ontology/>
    property: <{base}/property/>

Input  : Excel file with a question column ('original_query', 'question', or 'query'),
         optionally with 'institute' and 'document' columns.
Output : Excel file with questions, generated SPARQL, and full logical form diagnostics.

Dependencies:
    pip install pandas openpyxl spacy
    python -m spacy download en_core_web_sm

Usage:
    python fi_nl_to_sparql.py --input fi_queries.xlsx --output results.xlsx
    python fi_nl_to_sparql.py --input fi_queries.xlsx --output results.xlsx --sheet Sheet1
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
# Institute / namespace mapping
# ---------------------------------------------------------------------------

INSTITUTE_NS: Dict[str, str] = {
    "federal reserve": "https://www.federalreserve.gov/ontology/",
    "hkma":            "https://www.hkma.gov.hk/ontology/",
    "eu":              "https://ec.europa.eu/finance/ontology/",
}

FED_DOC_MAP: Dict[str, str] = {
    "208": "Regulation H",
    "211": "Regulation K",
    "212": "Regulation L",
    "215": "Regulation O",
    "223": "Regulation W",
    "225": "Regulation Y",
    "252": "Regulation YY",
}


def _ns(institute: str) -> str:
    return INSTITUTE_NS.get(institute.lower().strip(),
                            "https://www.federalreserve.gov/ontology/")


def _build_prefixes(institute: str) -> str:
    base = _ns(institute).rstrip("/")
    return (
        f"PREFIX fi:       <{base}/>\n"
        f"PREFIX property: <{base}/property/>\n"
        "PREFIX owl:      <http://www.w3.org/2002/07/owl#>\n"
        "PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX rdf:      <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
        "PREFIX xsd:      <http://www.w3.org/2001/XMLSchema#>\n"
        "PREFIX skos:     <http://www.w3.org/2004/02/skos/core#>\n"
    )


# ---------------------------------------------------------------------------
# Query type field mappings
# ---------------------------------------------------------------------------

QUERY_TYPE_PRIMARY_FIELD: Dict[str, str] = {
    "MarketData":           "currentPrice",
    "FinancialReporting":   "revenue",
    "BankingOperations":    "loanPortfolio",
    "TechnologyAndFintech": "digitalAdoption",
    "RiskManagement":       "creditRisk",
    "InvestmentAnalysis":   "portfolioValue",
    "RegulatoryCompliance": "complianceStatus",
    "TradingOperations":    "tradingVolume",
    "CorporateFinance":     "debtStructure",
}

QUERY_TYPE_FIELDS: Dict[str, List[str]] = {
    "MarketData":           ["currentPrice", "priceHistory", "tradingVolume",
                             "marketCapitalization", "volatilityMetrics"],
    "FinancialReporting":   ["revenue", "netIncome", "totalAssets",
                             "shareholdersEquity", "operatingCashFlow"],
    "BankingOperations":    ["loanPortfolio", "depositBase", "netInterestMargin",
                             "loanLossProvision", "creditQuality", "revenue", "netIncome"],
    "TechnologyAndFintech": ["digitalAdoption", "technologyInvestments",
                             "cybersecurityMetrics", "apiUsage", "automationLevel"],
    "RiskManagement":       ["creditRisk", "marketRisk", "operationalRisk",
                             "valueAtRisk", "stressTestResults",
                             "loanPortfolio", "depositBase"],
    "InvestmentAnalysis":   ["portfolioValue", "assetAllocation", "performanceMetrics",
                             "riskMetrics", "benchmarkComparison"],
    "RegulatoryCompliance": ["complianceStatus", "regulatoryCapital", "auditFindings",
                             "enforcementActions", "regulatoryReporting"],
    "TradingOperations":    ["tradingVolume", "executionQuality", "tradingRevenue",
                             "positionData", "counterpartyExposure"],
    "CorporateFinance":     ["debtStructure", "equityStructure", "costOfCapital",
                             "valuationMultiples", "cashFlowProjections",
                             "valuationMetrics", "balanceSheetData"],
}

QUERY_TYPE_PROPERTY: Dict[str, str] = {
    "MarketData":           "market_data",
    "FinancialReporting":   "financial_reporting",
    "BankingOperations":    "banking_operations",
    "TechnologyAndFintech": "technology_and_fintech",
    "RiskManagement":       "risk_management",
    "InvestmentAnalysis":   "investment_analysis",
    "RegulatoryCompliance": "regulatory_compliance",
    "TradingOperations":    "trading_operations",
    "CorporateFinance":     "corporate_finance",
}

# Keywords used to score field relevance against a question
_FIELD_KEYWORDS: Dict[str, List[str]] = {
    "loanPortfolio":        ["loan", "credit", "mortgage", "borrow"],
    "depositBase":          ["deposit", "account", "saving"],
    "netInterestMargin":    ["interest", "margin", "rate", "nim"],
    "loanLossProvision":    ["loss", "provision", "impairment", "write"],
    "creditQuality":        ["credit quality", "default", "non.performing"],
    "revenue":              ["revenue", "income", "earning", "profit"],
    "netIncome":            ["net income", "profit", "earning"],
    "currentPrice":         ["price", "market price", "current price"],
    "tradingVolume":        ["volume", "trading", "trade"],
    "marketCapitalization": ["market cap", "capitaliz"],
    "volatilityMetrics":    ["volatil", "var", "risk"],
    "creditRisk":           ["credit risk", "default", "credit"],
    "marketRisk":           ["market risk", "price risk", "trading"],
    "operationalRisk":      ["operational risk", "process", "fraud"],
    "valueAtRisk":          ["value at risk", "var", "stress"],
    "stressTestResults":    ["stress test", "scenario", "capital"],
    "complianceStatus":     ["compli", "regulat", "enforcement"],
    "regulatoryCapital":    ["capital", "tier", "ratio", "adequacy"],
    "auditFindings":        ["audit", "finding", "examination"],
    "portfolioValue":       ["portfolio", "fund", "investment", "asset"],
    "assetAllocation":      ["allocat", "asset class", "diversif"],
    "digitalAdoption":      ["digital", "online", "internet", "app"],
    "cybersecurityMetrics": ["cyber", "security", "authentication", "hack"],
    "debtStructure":        ["debt", "bond", "borrow", "leverage"],
    "equityStructure":      ["equity", "share", "stock", "capital struct"],
    "costOfCapital":        ["cost of capital", "wacc", "discount rate"],
    "valuationMultiples":   ["valuation", "multiple", "pe ratio", "ebitda"],
}


# ---------------------------------------------------------------------------
# Domain ontology vocabulary
# ---------------------------------------------------------------------------

FI_CLASSES: Dict[str, str] = {
    # Institutions
    "foreign banking organization": "fi:ForeignBankingOrganization",
    "bank holding company":         "fi:BankHoldingCompany",
    "operating subsidiary":         "fi:OperatingSubsidiary",
    "financial institution":        "fi:FinancialInstitution",
    "customer due diligence":       "fi:CustomerDueDiligence",
    "know your customer":           "fi:KYC",
    "corporate customer":           "fi:CorporateCustomer",
    "state member bank":            "fi:StateMemberBank",
    "two-factor authentication":    "fi:TwoFactorAuthentication",
    "capital requirement":          "fi:CapitalRequirement",
    "money laundering":             "fi:MoneyLaundering",
    "terrorist financing":          "fi:TerroristFinancing",
    "beneficial owner":             "fi:BeneficialOwner",
    "internet banking":             "fi:InternetBanking",
    "operational risk":             "fi:OperationalRisk",
    "liquidity risk":               "fi:LiquidityRisk",
    "member bank":                  "fi:MemberBank",
    "enforcement":                  "fi:EnforcementAction",
    "credit card":                  "fi:CreditCard",
    "institution":                  "fi:Institution",
    "credit risk":                  "fi:CreditRisk",
    "market risk":                  "fi:MarketRisk",
    "authentication":               "fi:Authentication",
    "cybersecurity":                "fi:Cybersecurity",
    "counterparty":                 "fi:Counterparty",
    "interest rate":                "fi:InterestRate",
    "exchange rate":                "fi:ExchangeRate",
    "compliance":                   "fi:Compliance",
    "regulation":                   "fi:Regulation",
    "examination":                  "fi:Examination",
    "investment":                   "fi:Investment",
    "subsidiary":                   "fi:Subsidiary",
    "affiliate":                    "fi:Affiliate",
    "portfolio":                    "fi:Portfolio",
    "securities":                   "fi:Securities",
    "derivative":                   "fi:Derivative",
    "transaction":                  "fi:Transaction",
    "mortgage":                     "fi:Mortgage",
    "guideline":                    "fi:Guideline",
    "statement":                    "fi:Statement",
    "custodian":                    "fi:Custodian",
    "borrower":                     "fi:Borrower",
    "investor":                     "fi:Investor",
    "customer":                     "fi:Customer",
    "consumer":                     "fi:Consumer",
    "transfer":                     "fi:Transfer",
    "payment":                      "fi:Payment",
    "deposit":                      "fi:Deposit",
    "capital":                      "fi:Capital",
    "penalty":                      "fi:Penalty",
    "account":                      "fi:Account",
    "policy":                       "fi:Policy",
    "report":                       "fi:Report",
    "lender":                       "fi:Lender",
    "broker":                       "fi:Broker",
    "dealer":                       "fi:Dealer",
    "notice":                       "fi:Notice",
    "equity":                       "fi:Equity",
    "trade":                        "fi:Trade",
    "audit":                        "fi:Audit",
    "fund":                         "fi:Fund",
    "bank":                         "fi:Bank",
    "loan":                         "fi:Loan",
    "bond":                         "fi:Bond",
    "risk":                         "fi:Risk",
    "rule":                         "fi:Rule",
    "fee":                          "fi:Fee",
    "aml":                          "fi:AML",
    "kyc":                          "fi:KYC",
    "api":                          "fi:API",
}

FI_ROLES: Dict[str, str] = {
    "affiliated with":  "fi:affiliatedWith",
    "controlled by":    "fi:controlledBy",
    "complies with":    "fi:compliesWith",
    "comply with":      "fi:compliesWith",
    "maintained by":    "fi:maintainedBy",
    "prohibited from":  "fi:prohibitedFrom",
    "regulated by":     "fi:regulatedBy",
    "required to":      "fi:requiredTo",
    "supervised by":    "fi:supervisedBy",
    "authorized by":    "fi:authorizedBy",
    "established by":   "fi:establishedBy",
    "permitted to":     "fi:permittedTo",
    "applies to":       "fi:appliesTo",
    "governed by":      "fi:governedBy",
    "provided by":      "fi:providedBy",
    "reported to":      "fi:reportedTo",
    "invested in":      "fi:investedIn",
    "subject to":       "fi:subjectTo",
    "insured by":       "fi:insuredBy",
    "backed by":        "fi:backedBy",
    "exposed to":       "fi:exposedTo",
    "issued by":        "fi:issuedBy",
    "owned by":         "fi:ownedBy",
}

FI_DATA_PROPS: Dict[str, str] = {
    "capital ratio":  "fi:hasCapitalRatio",
    "notice period":  "fi:hasNoticePeriod",
    "interest rate":  "fi:hasInterestRate",
    "requirement":    "fi:hasRequirement",
    "regulation":     "fi:hasRegulation",
    "frequency":      "fi:hasFrequency",
    "threshold":      "fi:hasThreshold",
    "maturity":       "fi:hasMaturity",
    "section":        "fi:hasSection",
    "period":         "fi:hasPeriod",
    "amount":         "fi:hasAmount",
    "ratio":          "fi:hasRatio",
    "limit":          "fi:hasLimit",
    "charge":         "fi:hasCharge",
    "date":           "fi:hasDate",
    "days":           "fi:hasDays",
    "term":           "fi:hasTerm",
    "rate":           "fi:hasRate",
    "fee":            "fi:hasFee",
}

# spaCy NER label -> (ontology class, data property)
SPACY_NER_MAP: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "PERSON":      ("fi:Person",          None),
    "NORP":        ("fi:Group",           None),
    "FAC":         ("fi:Facility",        None),
    "ORG":         ("fi:Organisation",    None),
    "GPE":         ("fi:Location",        None),
    "LOC":         ("fi:Location",        None),
    "PRODUCT":     ("fi:FinancialProduct", None),
    "EVENT":       ("fi:Event",           None),
    "LAW":         ("fi:Regulation",      None),
    "DATE":        (None,                 "fi:hasDate"),
    "TIME":        (None,                 "fi:hasTime"),
    "PERCENT":     (None,                 "fi:hasPercentage"),
    "MONEY":       (None,                 "fi:hasAmount"),
    "QUANTITY":    (None,                 "fi:hasQuantity"),
    "ORDINAL":     (None,                 "fi:hasOrdinal"),
    "CARDINAL":    (None,                 "fi:hasCardinal"),
}

# Verb lemma -> ontology predicate URI
_FI_VERB_MAP: Dict[str, str] = {
    "require":     "fi:requires",     "need":       "fi:requires",
    "prohibit":    "fi:prohibits",    "permit":     "fi:permits",
    "allow":       "fi:allows",       "authorize":  "fi:authorizes",
    "authorise":   "fi:authorizes",   "restrict":   "fi:restricts",
    "limit":       "fi:limits",       "cap":        "fi:caps",
    "regulate":    "fi:regulates",    "supervise":  "fi:supervises",
    "examine":     "fi:examines",     "audit":      "fi:audits",
    "report":      "fi:reports",      "disclose":   "fi:discloses",
    "notify":      "fi:notifies",     "inform":     "fi:informs",
    "provide":     "fi:provides",     "offer":      "fi:offers",
    "maintain":    "fi:maintains",    "hold":       "fi:holds",
    "issue":       "fi:issues",       "grant":      "fi:grants",
    "approve":     "fi:approves",     "deny":       "fi:denies",
    "refuse":      "fi:refuses",      "close":      "fi:closes",
    "open":        "fi:opens",        "cancel":     "fi:cancels",
    "terminate":   "fi:terminates",   "charge":     "fi:charges",
    "waive":       "fi:waives",       "calculate":  "fi:calculates",
    "compute":     "fi:computes",     "apply":      "fi:appliesTo",
    "constitute":  "fi:constitutes",  "define":     "fi:defines",
    "include":     "fi:includes",     "exclude":    "fi:excludes",
    "classify":    "fi:classifies",   "treat":      "fi:treats",
    "consolidate": "fi:consolidates", "aggregate":  "fi:aggregates",
    "engage":      "fi:engagesIn",    "conduct":    "fi:conducts",
    "invest":      "fi:investsIn",    "trade":      "fi:tradesIn",
    "transfer":    "fi:transfers",    "bear":       "fi:bears",
    "comply":      "fi:compliesWith", "violate":    "fi:violates",
    "exceed":      "fi:exceeds",      "meet":       "fi:meets",
    "fail":        "fi:fails",        "use":        "fi:uses",
    "adopt":       "fi:adopts",       "implement":  "fi:implements",
    "verify":      "fi:verifies",     "identify":   "fi:identifies",
    "assess":      "fi:assesses",     "monitor":    "fi:monitors",
    "manage":      "fi:manages",      "mitigate":   "fi:mitigates",
}

_REG_REF_RE  = re.compile(r'(?:section|§)\s*(\d+[A-Za-z]?(?:\.\d+)?(?:\([a-z0-9]+\))*)', re.IGNORECASE)
_REG_NAME_RE = re.compile(r'Regulation\s+([A-Z]{1,3}[A-Z0-9\-]*)', re.IGNORECASE)
_PERCENT_RE  = re.compile(r'(\d+(?:\.\d+)?)\s*(?:percent|%)')
_DAYS_RE     = re.compile(r'(\d+)\s*days?', re.IGNORECASE)
_DOC_NUM_RE  = re.compile(r'\b(2\d{2})\b')

_CLAUSE_HEAD_DEPS = {"ROOT", "relcl", "advcl", "xcomp", "ccomp", "conj", "acl"}
_SUBJ_DEPS        = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS         = {"dobj", "pobj", "attr", "ccomp", "xcomp", "oprd", "acomp"}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QueryIntent(Enum):
    SELECT_REQUIREMENT  = auto()
    SELECT_DEFINITION   = auto()
    SELECT_PROCEDURE    = auto()
    SELECT_ENTITLEMENT  = auto()
    SELECT_OBLIGATION   = auto()
    SELECT_RISK_METRICS = auto()
    SELECT_MARKET_DATA  = auto()
    SELECT_REPORTING    = auto()
    SELECT_COMPLIANCE   = auto()
    SELECT_TECHNOLOGY   = auto()
    SELECT_GENERAL      = auto()
    ASK_BOOLEAN         = auto()
    ASK_EXISTS          = auto()
    DESCRIBE_ENTITY     = auto()


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
    concept_assertions:  List[Tuple[str, str]]           = field(default_factory=list)
    role_assertions:     List[Tuple[str, str, str]]      = field(default_factory=list)
    negation_assertions: List[Tuple[str, str, str]]      = field(default_factory=list)
    nary_assertions:     List[NaryAssertion]             = field(default_factory=list)
    data_assertions:     List[Tuple[str, str, str]]      = field(default_factory=list)
    union_groups:        List[List[Tuple[str, str, str]]] = field(default_factory=list)
    class_exprs:         List[str]                       = field(default_factory=list)


@dataclass
class QuestionLogicalForm:
    """
    Logical form: lambda ?focusVar . P(?focusVar) given presuppositions.
    Augmented with FI-specific metadata extracted from the source question.
    """
    raw_question:         str
    question_type:        QuestionType                 = QuestionType.POLAR
    focus_var:            str                          = ""
    spo_triples:          List[SPOTriple]              = field(default_factory=list)
    presuppositions:      List[SPOTriple]              = field(default_factory=list)
    restrictions:         List[SPOTriple]              = field(default_factory=list)
    negations:            List[SPOTriple]              = field(default_factory=list)
    conjunctions:         List[SPOTriple]              = field(default_factory=list)
    disjunctions:         List[List[SPOTriple]]        = field(default_factory=list)
    entities:             List[Entity]                 = field(default_factory=list)
    nary_args:            List[Entity]                 = field(default_factory=list)
    relations:            List[Relation]               = field(default_factory=list)
    data_props:           List[DataProperty]           = field(default_factory=list)
    quantifier:           str                          = "existential"
    status_indicators:    List[Tuple[str, str]]        = field(default_factory=list)
    service_actions:      List[str]                    = field(default_factory=list)
    ner_hits:             List[SpacyNERHit]            = field(default_factory=list)
    intent:               Optional[QueryIntent]        = None
    disjunctive_entities: List[List[Entity]]           = field(default_factory=list)
    connective:           str                          = "AND"
    negated:              bool                         = False
    fi_meta:              Dict[str, Any]               = field(default_factory=dict)
    fi_query_type:        str                          = "BankingOperations"
    institute:            str                          = "Federal Reserve"
    document:             str                          = ""
    projected_fields:     List[str]                    = field(default_factory=list)


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
# Stage 1 — Metadata extraction
# ---------------------------------------------------------------------------

def extract_fi_metadata(text: str) -> Dict[str, Any]:
    """Extract regulatory references, percentages, and day counts from question text."""
    meta: Dict[str, Any] = {}

    sections = _REG_REF_RE.findall(text)
    if sections:
        meta["sections"] = sections

    regs = _REG_NAME_RE.findall(text)
    if regs:
        meta["regulations"] = regs

    percents = _PERCENT_RE.findall(text)
    if percents:
        meta["percentages"] = percents

    days = _DAYS_RE.findall(text)
    if days:
        meta["days"] = days

    doc_nums  = _DOC_NUM_RE.findall(text)
    known_docs = [d for d in doc_nums if d in FED_DOC_MAP]
    if known_docs:
        meta["doc_numbers"] = known_docs
        meta["reg_names"]   = [FED_DOC_MAP[d] for d in known_docs]

    return meta


# ---------------------------------------------------------------------------
# Stage 1 — Institute and document inference
# ---------------------------------------------------------------------------

def _infer_institute(question: str, document: str = "") -> str:
    tl = (question + " " + document).lower()

    if re.search(r'hkma|hong\s+kong|code\s+of\s+banking\s+practice'
                 r'|aml.*ordinance|mis.transfer|fintech.*hkma'
                 r'|institution[s]?\s+should', tl):
        return "HKMA"

    if re.search(r'\beu\b|european\s+union|mifid|crd|crr|emir'
                 r'|regulation\s+\(eu\)|directive\s+\d{4}/\d+/eu', tl):
        return "EU"

    return "Federal Reserve"


def _infer_document(question: str, institute: str,
                    fi_meta: Dict[str, Any]) -> str:
    tl = question.lower()

    if institute == "HKMA":
        if re.search(r'aml|anti.money\s+laundering|beneficial\s+owner'
                     r'|terrorist\s+financing|ownership.*structure', tl):
            return "Anti-Money Laundering and Counter-Terrorist Financing Ordinance"
        if re.search(r'internet\s+banking|authentication|two.factor'
                     r'|technology\s+risk|cyber|online', tl):
            return "General Principles for Technology Risk Management"
        if re.search(r'credit\s+card|unauthorised.*transaction', tl):
            return "Credit Card Business"
        if re.search(r'credit\s+data|credit\s+reference', tl):
            return "Code of Practice on Consumer Credit Data (the Code)"
        if re.search(r'fund\s+transfer|mis.transfer', tl):
            return "Handling procedures for Following up Mis-transfer of Funds"
        return "Code of Banking Practice"

    if institute == "EU":
        return "REGULATION (EU)"

    # Federal Reserve — map document numbers or infer from content
    if "reg_names" in fi_meta:
        return fi_meta["reg_names"][0]
    if "doc_numbers" in fi_meta:
        return FED_DOC_MAP.get(fi_meta["doc_numbers"][0], "")

    if re.search(r'state\s+member\s+bank|regulation\s+h\b|208', tl):
        return "Regulation H"
    if re.search(r'foreign.*(?:bank|organization|banking)|regulation\s+k\b|211', tl):
        return "Regulation K"
    if re.search(r'interlocking|regulation\s+l\b|212', tl):
        return "Regulation L"
    if re.search(r'insider\s+loan|insider\s+credit|executive\s+officer'
                 r'|regulation\s+o\b|215', tl):
        return "Regulation O"
    if re.search(r'covered\s+transaction|affiliate\s+transaction|section\s+23[AB]'
                 r'|valuation\s+rule|regulation\s+w\b|223', tl):
        return "Regulation W"
    if re.search(r'bank\s+holding\s+company|nonbanking\s+activit'
                 r'|regulation\s+y\b|225', tl):
        return "Regulation Y"
    if re.search(r'enhanced\s+prudential|systemically\s+important'
                 r'|stress\s+test.*large|resolution\s+plan'
                 r'|regulation\s+yy\b|252', tl):
        return "Regulation YY"
    if re.search(r'longer.run\s+goal|monetary\s+policy\s+strategy'
                 r'|inflation\s+target|employment\s+goal', tl):
        return "Statement on Longer-Run Goals and Monetary Policy Strategy"

    return ""


# ---------------------------------------------------------------------------
# Stage 1 — Entity, relation and data property extraction
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
    Return deduplicated, span-sorted entities from dictionary lookup and spaCy NER.
    Longer surface forms are matched first to avoid partial-match shadowing.
    """
    tl         = text.lower()
    seen_spans: List[Tuple[int, int]] = []
    found:      List[Entity]          = []

    for term in sorted(FI_CLASSES, key=len, reverse=True):
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
                found.append(Entity(surface=term, uri=FI_CLASSES[term],
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


def extract_data_properties(text: str, fi_meta: Dict[str, Any]) -> List[DataProperty]:
    """Extract data properties from dictionary lookup, spaCy NER numeric hits, and FI metadata."""
    tl        = text.lower()
    seen_uris: set = set()
    dps:       List[DataProperty] = []

    for surf in sorted(FI_DATA_PROPS, key=len, reverse=True):
        uri = FI_DATA_PROPS[surf]
        if surf in tl and uri not in seen_uris:
            seen_uris.add(uri)
            m = re.search(re.escape(surf) + r"[\s:=]+([\w.\-#%]+)", tl)
            dps.append(DataProperty(surface=surf, uri=uri,
                                    value=m.group(1).strip() if m else None))

    for hit in _run_spacy_ner(text):
        if hit.is_data_value and hit.data_prop not in seen_uris:
            seen_uris.add(hit.data_prop)
            dps.append(DataProperty(surface=hit.surface, uri=hit.data_prop, value=hit.surface))

    for meta_key, dp_uri, fmt in [
        ("sections",    "fi:hasSection",    lambda v: v),
        ("days",        "fi:hasDays",       lambda v: v),
        ("percentages", "fi:hasPercentage", lambda v: v),
    ]:
        if meta_key in fi_meta and dp_uri not in seen_uris:
            seen_uris.add(dp_uri)
            val = fmt(fi_meta[meta_key][0])
            dps.append(DataProperty(surface=str(val), uri=dp_uri, value=str(val)))

    return dps


def _extract_surface_relations(text: str) -> List[Relation]:
    tl   = text.lower()
    seen: set = set()
    found: List[Relation] = []

    for surf in sorted(FI_ROLES, key=len, reverse=True):
        if surf in tl and surf not in seen:
            seen.add(surf)
            pos    = tl.find(surf)
            prefix = tl[max(0, pos - 25): pos]
            neg    = bool(re.search(
                r"\b(not|doesn't|don't|cannot|can't|never|without)\b", prefix
            ))
            found.append(Relation(surface=surf, uri=FI_ROLES[surf],
                                  source="surface_dict", negated=neg))
    return found


# ---------------------------------------------------------------------------
# Stage 2 — Query type and intent classification
# ---------------------------------------------------------------------------

def classify_fi_query_type(question: str) -> str:
    """
    Classify into one of the 9 FI query types.
    Patterns are ordered from most specific to least specific.
    """
    tl = question.lower()

    patterns: List[Tuple[str, str]] = [
        (r'market\s+price|stock\s+price|trading\s+volume|market\s+cap'
         r'|price\s+history|volatil|exchange\s+rate|bid|ask\s+price'
         r'|benchmark\s+rate|libor|sofr',
         "MarketData"),
        (r'trading\s+order|order\s+execution|exchange\s+member|foreign\s+banking'
         r'.*(?:place|accept)\s+order|engage.*business.*united\s+states'
         r'|underwriting|securities.*dealer|broker.*dealer',
         "TradingOperations"),
        (r'corporate\s+(?:finance|structure|governance)|merger|acquisition'
         r'|debt\s+structure|equity\s+structure|valuation\s+rule'
         r'|affiliate.*liabilit|operating\s+subsidiary.*acquisition'
         r'|section\s+23[AB]|valuation.*member\s+bank',
         "CorporateFinance"),
        (r'internet\s+banking|two.factor|authentication|cyber|fintech'
         r'|digital|e.statement|e-mail.*bank|notification.*email'
         r'|technology\s+risk|api|automated|online\s+banking',
         "TechnologyAndFintech"),
        (r'aml|anti.money\s+laundering|kyc|know\s+your\s+customer'
         r'|beneficial\s+owner|cdd|customer\s+due\s+diligence'
         r'|terrorist\s+financing|suspicious\s+transaction'
         r'|sanctions|ownership.*structure|corporate.*customer.*information',
         "RegulatoryCompliance"),
        (r'capital\s+(?:ratio|requirement|adequacy|buffer)'
         r'|basel|stress\s+test|value\s+at\s+risk|\bvar\b'
         r'|credit\s+risk|market\s+risk|operational\s+risk'
         r'|liquidity\s+(?:risk|coverage|ratio)'
         r'|risk\s+management|concentration\s+risk|exposure\s+limit',
         "RiskManagement"),
        (r'investment|portfolio|asset\s+allocation|fund\s+transfer'
         r'|mutual\s+fund|etf|securities\s+purchase|asset\s+management'
         r'|performance.*(?:metric|benchmark)|return\s+on',
         "InvestmentAnalysis"),
        (r'revenue|net\s+income|earnings|profit|loss|balance\s+sheet'
         r'|total\s+assets|shareholders\s+equity|cash\s+flow'
         r'|financial\s+statement|annual\s+report|quarterly\s+report'
         r'|interest\s+income|non.interest',
         "FinancialReporting"),
        (r'regulation\s+[a-z]{1,3}\b|section\s+\d+|member\s+bank'
         r'|state\s+member\s+bank|bank\s+holding\s+(?:company|act)'
         r'|examination|supervisory|examiners?|examinations?'
         r'|fdic|federal\s+reserve|board\s+of\s+governors'
         r'|what\s+constitutes|how\s+is\s+.*\s+defined'
         r'|compliance.*requirement|regulatory.*requirement',
         "RegulatoryCompliance"),
    ]

    for pattern, qtype in patterns:
        if re.search(pattern, tl):
            return qtype

    return "BankingOperations"


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
    (r"\bwhat\b",                                        QuestionType.WH_WHAT),
    (r"\bis\s+there\b|\bare\s+there\b|\bexists?\b",      QuestionType.EXISTENCE),
    (r"^(can|could|does|do|is|are|was|were|will|would|has|have|had|should|must)\b",
                                                         QuestionType.POLAR),
]


def classify_question_type(question: str) -> QuestionType:
    tl = question.lower().strip()
    for pat, qt in _QT_PATTERNS:
        if re.search(pat, tl):
            return qt
    return QuestionType.POLAR


_INTENT_PATTERNS: List[Tuple[str, QueryIntent]] = [
    (r'\bwhat\s+(constitutes?|is\s+(a|an|the)\s+definition\s+of'
     r'|does\s+.*\s+mean|is\s+considered|counts?\s+as|qualifies?\s+as)\b',
     QueryIntent.SELECT_DEFINITION),
    (r'\b(capital\s+ratio|capital\s+requirement|stress\s+test|value\s+at\s+risk'
     r'|\bvar\b|credit\s+risk|market\s+risk|operational\s+risk|liquidity\s+ratio'
     r'|risk\s+weight|leverage\s+ratio)\b',
     QueryIntent.SELECT_RISK_METRICS),
    (r'\b(market\s+price|stock\s+price|exchange\s+rate|interest\s+rate'
     r'|trading\s+volume|market\s+cap|volatility|benchmark\s+rate)\b',
     QueryIntent.SELECT_MARKET_DATA),
    (r'\b(internet\s+banking|two.factor|authentication|cyber|digital'
     r'|e.statement|technology\s+risk|api|automated)\b',
     QueryIntent.SELECT_TECHNOLOGY),
    (r'\b(financial\s+statement|annual\s+report|quarterly|revenue|net\s+income'
     r'|balance\s+sheet|earnings|cash\s+flow)\b',
     QueryIntent.SELECT_REPORTING),
    (r'\b(regulatory\s+(?:requirement|capital|reporting|compliance)'
     r'|compliance\s+(?:status|requirement)|enforcement|supervisory'
     r'|examination|what\s+constitutes\s+a\s+change)\b',
     QueryIntent.SELECT_COMPLIANCE),
    (r'\bcan\s+i\b|\bcan\s+(a\s+)?customer\b|\bam\s+i\s+(entitled|allowed|permitted)\b'
     r'|\bdo\s+i\s+have\s+(?:to|the\s+right)\b',
     QueryIntent.SELECT_ENTITLEMENT),
    (r'\bbank\s+(must|shall|should|is\s+required)\b'
     r'|\binstitution[s]?\s+(must|shall|should)\b'
     r'|\bobligat(ion|ed)\b|\bduty\s+of\b',
     QueryIntent.SELECT_OBLIGATION),
    (r'\bhow\s+(do|does|can|should|to)\b|\bwhat\s+(steps?|process|procedure)\b'
     r'|\bwhat\s+should\s+(i|a\s+bank)\s+(take|do|follow)\b',
     QueryIntent.SELECT_PROCEDURE),
    (r'\b(require[sd]?|requirement|must|shall|necessary|mandatory|prohibited'
     r'|permitted|obligation)\b',
     QueryIntent.SELECT_REQUIREMENT),
    (r'\bexists?\b|\bis\s+there\b|\bare\s+there\b',      QueryIntent.ASK_EXISTS),
    (r'^(can|could|does|do|is|are|was|were|will|would|has|have|had|should|must)\b',
     QueryIntent.ASK_BOOLEAN),
    (r'^(describe|what\s+is\s+(a|an|the)\b|tell\s+me\s+about)\b',
     QueryIntent.DESCRIBE_ENTITY),
]


def classify_intent(question: str) -> QueryIntent:
    tl = question.lower().strip()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, tl):
            return intent
    return QueryIntent.SELECT_GENERAL


def _infer_focus_var(qt: QuestionType, intent: QueryIntent) -> str:
    focus_by_qt: Dict[QuestionType, str] = {
        QuestionType.WH_WHY:      "?reason",
        QuestionType.WH_WHEN:     "?date",
        QuestionType.WH_WHERE:    "?location",
        QuestionType.WH_WHO:      "?entity",
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
        QueryIntent.SELECT_REQUIREMENT  : "?requirement",
        QueryIntent.SELECT_DEFINITION   : "?definition",
        QueryIntent.SELECT_PROCEDURE    : "?procedure",
        QueryIntent.SELECT_ENTITLEMENT  : "?entitlement",
        QueryIntent.SELECT_OBLIGATION   : "?obligation",
        QueryIntent.SELECT_RISK_METRICS : "?riskMetric",
        QueryIntent.SELECT_MARKET_DATA  : "?marketData",
        QueryIntent.SELECT_REPORTING    : "?financialData",
        QueryIntent.SELECT_COMPLIANCE   : "?complianceStatus",
        QueryIntent.SELECT_TECHNOLOGY   : "?techMetric",
        QueryIntent.SELECT_GENERAL      : "?result",
    }
    return focus_by_intent.get(intent, "?result")


def _infer_projected_fields(fi_query_type: str, question: str) -> List[str]:
    """
    Select up to 3 fields from QUERY_TYPE_FIELDS scored against the question.
    The primary field always appears first.
    """
    all_fields = QUERY_TYPE_FIELDS.get(fi_query_type, ["result"])
    tl         = question.lower()

    scored = sorted(
        [(sum(2 for kw in _FIELD_KEYWORDS.get(f, []) if re.search(kw, tl)), f)
         for f in all_fields],
        reverse=True,
    )
    primary = all_fields[0]
    result  = [primary]
    for _, f in scored:
        if f != primary and len(result) < 3:
            result.append(f)
    return result


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


def extract_spo_triples(text: str, entities: List[Entity]) -> List[SPOTriple]:
    doc        = NLP(text)
    triples:   List[SPOTriple]   = []
    tok_to_uri: Dict[int, str]   = {}
    seen:       set               = set()

    for ent in entities:
        for tok in doc:
            if tok.text.lower() in ent.surface or ent.surface in tok.text.lower():
                tok_to_uri.setdefault(tok.i, ent.uri)

    for tok in doc:
        if tok.pos_ != "VERB" or tok.dep_ not in _CLAUSE_HEAD_DEPS:
            continue
        lemma    = tok.lemma_.lower()
        pred_uri = _FI_VERB_MAP.get(lemma)
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
    if qt in (QuestionType.POLAR, QuestionType.EXISTENCE):
        return "presupposition"

    def _v(val: str, is_uri: bool) -> str:
        return _varname(val) if is_uri else (val if val.startswith("?") else _tok_to_var(val))

    if focus_var and focus_var in (_v(triple.subj, triple.subj_is_uri),
                                   _v(triple.obj,  triple.obj_is_uri)):
        return "restriction"
    return "presupposition"


def build_logical_form(question: str,
                       institute: str = "",
                       document:  str = "") -> QuestionLogicalForm:
    fi_meta       = extract_fi_metadata(question)
    fi_query_type = classify_fi_query_type(question)
    inst          = institute or _infer_institute(question, document)
    doc           = document  or _infer_document(question, inst, fi_meta)

    all_entities  = extract_entities(question)
    data_props    = extract_data_properties(question, fi_meta)
    ner_hits      = _run_spacy_ner(question)

    qt            = classify_question_type(question)
    intent        = classify_intent(question)
    focus_var     = _infer_focus_var(qt, intent)
    proj_fields   = _infer_projected_fields(fi_query_type, question)

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
    conjunctions = ([t for t in spo_triples if not t.negated]
                    if re.search(r"\bboth\b|\band\b|\bas well as\b", tl) else [])

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

    surface_rels  = _extract_surface_relations(question)

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
        fi_meta              = fi_meta,
        fi_query_type        = fi_query_type,
        institute            = inst,
        document             = doc,
        projected_fields     = proj_fields,
    )


# ---------------------------------------------------------------------------
# Stage 5 — DL/OWL2 expression
# ---------------------------------------------------------------------------

def lf_to_dl(lf: QuestionLogicalForm) -> DLExpression:
    dl    = DLExpression()
    added: set = set()

    for ent in lf.entities + lf.nary_args:
        dl.concept_assertions.append((_varname(ent.uri), ent.uri))

    def _sv(t: SPOTriple) -> Tuple[str, Optional[str], str]:
        def _v(val: str, is_uri: bool) -> str:
            return _varname(val) if is_uri else (val if val.startswith("?") else _tok_to_var(val))
        pred = (t.pred if t.pred_is_uri else None)
        return _v(t.subj, t.subj_is_uri), pred, _v(t.obj, t.obj_is_uri)

    for t in lf.spo_triples:
        s, p, o = _sv(t)
        if p is None:
            p = f"fi:relatedTo  # verb: {t.pred}"
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
            (_varname(lf.entities[0].uri), "fi:relatedTo", _varname(lf.entities[1].uri))
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
# Stage 6 — SPARQL generation
# ---------------------------------------------------------------------------

class SPARQLBuilder:

    def build(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        try:
            return self._fi_select(lf, dl)
        except Exception as exc:
            return self._fallback(lf, exc)

    def _nlq_annotation(self, lf: QuestionLogicalForm, indent: str = "  ") -> str:
        """
        Embed NLQ as rdfs:comment on _:queryMeta inside an OPTIONAL block.
        Records intent, question type, query type, institute, and focus variable
        as typed literals for downstream tooling.
        """
        q    = _escape(lf.raw_question)
        base = _ns(lf.institute).rstrip("/")
        it   = lf.intent.name if lf.intent else "UNKNOWN"
        qt   = lf.question_type.name
        fqt  = lf.fi_query_type
        fv   = lf.focus_var or ""
        return (
            f"{indent}OPTIONAL {{\n"
            f"{indent}  _:queryMeta rdf:type          <{base}/SPARQLQuery> ;\n"
            f'{indent}              rdfs:comment      "{q}"@en ;\n'
            f'{indent}              <{base}/property/intent>        "{it}"^^xsd:string ;\n'
            f'{indent}              <{base}/property/questionType>  "{qt}"^^xsd:string ;\n'
            f'{indent}              <{base}/property/queryType>     "{fqt}"^^xsd:string ;\n'
            f'{indent}              <{base}/property/institute>     "{lf.institute}"^^xsd:string ;\n'
            f'{indent}              <{base}/property/focusVariable> "{fv}"^^xsd:string .\n'
            f"{indent}}}"
        )

    def _fi_select(self, lf: QuestionLogicalForm, dl: DLExpression) -> str:
        """
        Build a SPARQL SELECT matching the fi_queries dataset pattern:

            SELECT ?result [?field2 ?field3] WHERE {
              ?service fi:query_type    "<query_type>" .
              ?service fi:institute     "<institute>" .
              ?service fi:document      "<document>" .
              ?service fi:description   "<question>"@en .
              ?service fi:<primary>     ?result .
              [OPTIONAL { ?service fi:<field2> ?field2 . }]
              ...
              [FILTER NOT EXISTS { ... }]       # negations
              [... UNION ...]                   # disjunctions
              [OPTIONAL { _:queryMeta ... }]    # NLQ annotation
            }
        """
        q         = _escape(lf.raw_question)
        fv        = lf.focus_var or "?result"
        pri_field = lf.projected_fields[0] if lf.projected_fields else "result"
        extra     = lf.projected_fields[1:3]

        select_vars = (fv + " " + " ".join(f"?{f}" for f in extra)).strip()

        lines = [_build_prefixes(lf.institute)]
        lines.append(f"SELECT {select_vars} WHERE {{")
        lines.append(f'  ?service fi:query_type  "{lf.fi_query_type}" .')
        lines.append(f'  ?service fi:institute   "{lf.institute}" .')
        if lf.document:
            lines.append(f'  ?service fi:document    "{lf.document}" .')
        lines.append(f'  ?service fi:description "{q}"@en .')
        lines.append(f'  ?service fi:{pri_field}  {fv} .')

        for f in extra:
            lines.append(f"  OPTIONAL {{ ?service fi:{f} ?{f} . }}")

        for sec in lf.fi_meta.get("sections", [])[:2]:
            lines.append(f'  OPTIONAL {{ ?service fi:hasSection "{_escape(sec)}" . }}')

        if "reg_names" in lf.fi_meta:
            lines.append(
                f'  OPTIONAL {{ ?service fi:hasRegulation "{_escape(lf.fi_meta["reg_names"][0])}" . }}'
            )

        if "days" in lf.fi_meta:
            lines.append(
                f'  OPTIONAL {{ ?service fi:hasDays "{lf.fi_meta["days"][0]}"^^xsd:integer . }}'
            )

        if "percentages" in lf.fi_meta:
            lines.append(
                f'  OPTIONAL {{ ?service fi:hasPercentage "{lf.fi_meta["percentages"][0]}"^^xsd:decimal . }}'
            )

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
            _build_prefixes(lf.institute)
            + f"# ERROR: {exc}\n"
            + "SELECT ?result WHERE {\n"
            + f'  ?service fi:description "{q}"@en .\n'
            + "  ?service fi:result ?result .\n"
            + self._nlq_annotation(lf) + "\n"
            + "}\n"
        )


_BUILDER = SPARQLBuilder()


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def nlq_to_sparql(question: str,
                  institute: str = "",
                  document:  str = "") -> Dict[str, Any]:
    """
    Run the full NLQ -> SPARQL pipeline for a single question.

    Returns a flat dict suitable for direct use as a DataFrame row,
    including the generated SPARQL, full logical form diagnostics,
    and NER breakdowns per label.
    """
    lf     = build_logical_form(question, institute, document)
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
        "FI Query Type":      lf.fi_query_type,
        "Institute":          lf.institute,
        "Document":           lf.document,
        "Projected Fields":   str(lf.projected_fields),
        "Sections":           str(lf.fi_meta.get("sections",   [])),
        "Regulation Names":   str(lf.fi_meta.get("reg_names",  [])),
        "Days":               str(lf.fi_meta.get("days",       [])),
        "Percentages":        str(lf.fi_meta.get("percentages", [])),
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
        "NER_ORG":            str(ner_by_label.get("ORG",      [])),
        "NER_LAW":            str(ner_by_label.get("LAW",      [])),
        "NER_DATE":           str(ner_by_label.get("DATE",     [])),
        "NER_MONEY":          str(ner_by_label.get("MONEY",    [])),
        "NER_PERCENT":        str(ner_by_label.get("PERCENT",  [])),
        "NER_CARDINAL":       str(ner_by_label.get("CARDINAL", [])),
        "NER_GPE":            str(ner_by_label.get("GPE",      [])),
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _find_question_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in ("original_query", "question", "query"):
            return col
    raise ValueError(f"No question column found. Available columns: {list(df.columns)}")


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

        institute = str(row.get("institute", "")).strip()
        document  = str(row.get("document",  "")).strip()
        if institute.lower() == "nan":
            institute = ""
        if document.lower() == "nan":
            document  = ""

        try:
            r = nlq_to_sparql(q, institute=institute, document=document)
            r["S.No"]              = row.get("S.No", idx)
            r["source_query_type"] = row.get("query_type", "")
            r["source_fields"]     = row.get("fields", "")
            r["ground_truth"]      = str(row.get("ground_truth", ""))[:300]
            r["reference"]         = str(row.get("reference",    ""))[:300]
        except Exception as exc:  # noqa: BLE001
            r = {
                "Question":    q,
                "S.No":        row.get("S.No", idx),
                "Intent":      "ERROR",
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
        description="Convert natural language financial queries to SPARQL."
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
        print(f"\nQ   : {r['Question'][:95]}")
        print(f"  QType   : {r['Question Type']}  |  FocusVar: {r['Focus Variable']}")
        print(f"  Intent  : {r['Intent']}")
        print(f"  FIQType : {r['FI Query Type']}  |  Institute: {r['Institute']}")
        print(f"  Doc     : {r['Document']}")
        print(f"  Fields  : {r['Projected Fields']}")
        print(f"\n  SPARQL:\n{r['SPARQL Query']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
