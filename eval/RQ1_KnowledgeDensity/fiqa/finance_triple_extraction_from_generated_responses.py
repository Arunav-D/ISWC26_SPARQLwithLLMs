"""
pipeline2a_finance.py
=====================
Knowledge Graph Triple Extraction — Financial Industry Domain
Rule-Based + spaCy Extractors, TKF Evaluation (KD / Triple Counts)

Extractors
----------
  rule_based     Five syntactic passes over spaCy dependency trees:
                   Pass 1 — relational  (nsubj/dobj verb triples)
                   Pass 2 — structural  (governed-by, subject-to, etc.)
                   Pass 3 — attribute modifiers (amod/nummod)
                   Pass 4 — conjunctions (owl:intersectionOf / unionOf)
                   Pass 5 — causal chains (leads-to, results-in, etc.)

  spacy_general  General SVO extraction over all VERB/AUX tokens.

Metrics computed (this module)
-------------------------------
  KD            Knowledge Density  =  high-confidence triples / total tokens
  TKF           Composite score    =  KD only (W_KD = 1.0 in this module)
  triple_count_c Candidate triple count (extractor output)
  triple_count_g Gold triple count (rule-based on ground truth)

  GED / SF / PAR / MA / triple-F1 / predicate metrics are out of scope
  for this module.

Excel output sheets
-------------------
  Leaderboard          Global ranking (model × method × extractor)
  Summary__{model}     Aggregated TKF stats per method × extractor
  Detail_{model}_{ext} Row-level metrics (one row per query × extractor)
  Triples_C__{model}   Candidate triples per row
  Triples_G            Ground-truth triples (deduped)
  TripleCounts         C/G count ratios per row

Memory strategy
---------------
  Each model's rows stream to compressed CSV files in a temporary
  directory as they are produced.  Only aggregation rows and GT triples
  live in RAM across models.  Per-model Excel sheets are assembled from
  those CSVs one model at a time with explicit del + gc.collect() calls.

Usage
-----
  # default: input/output resolve relative to this script's directory
  python pipeline2a_finance.py

  # explicit paths
  python pipeline2a_finance.py \\
      --input  /path/to/ALL_MODELS_responses_finance.xlsx \\
      --output /path/to/pipeline2a_finance_output.xlsx

  # override NLP model
  python pipeline2a_finance.py --nlp-model en_core_web_trf

Dependencies
------------
  pip install spacy numpy pandas openpyxl
  python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import collections
import gc
import math
import re
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
# Edit the constants below only if your data uses different sheet/column names.

GROUND_TRUTH_SHEET = "ground_truth"
RESPONSE_SUFFIX    = "_response"
SKIP_SHEETS: List[str] = [GROUND_TRUTH_SHEET]

DEFAULT_INPUT_FILENAME  = "ALL_MODELS_responses_finance.xlsx"
DEFAULT_OUTPUT_FILENAME = "pipeline2a_finance_output.xlsx"

# spaCy model — override at runtime with --nlp-model
NLP_MODEL = "en_core_web_sm"

# TKF composite: this module uses KD only
W_KD: float = 1.0

# Confidence gate: triples below this threshold are dropped
CONF_THRESHOLD: float = 0.60

# KD confidence threshold: triples must meet this to count as "typed"
KD_CONFIDENCE_THRESHOLD: float = 0.75

# Flush streaming buffer to CSV after this many input rows
_FLUSH_EVERY: int = 50

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
# Utilities
# ---------------------------------------------------------------------------

def safe_div(a: float, b: float, d: float = 0.0) -> float:
    """Return a / b, or *d* when b is zero."""
    return a / b if b else d


def _to_camel_case(s: str) -> str:
    """Convert a whitespace-separated surface string to lowerCamelCase."""
    parts = re.sub(r"[^a-zA-Z0-9\s]", " ", str(s)).split()
    if not parts:
        return "x"
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


# ---------------------------------------------------------------------------
# Financial industry ontology
# ---------------------------------------------------------------------------

FI_CLASSES: Dict[str, str] = {
    # Institutions
    "bank":                         "fi:Bank",
    "banks":                        "fi:Bank",
    "member bank":                  "fi:MemberBank",
    "state member bank":            "fi:StateMemberBank",
    "foreign banking organization": "fi:ForeignBankingOrganization",
    "bank holding company":         "fi:BankHoldingCompany",
    "holding company":              "fi:HoldingCompany",
    "financial institution":        "fi:FinancialInstitution",
    "institution":                  "fi:Institution",
    "institutions":                 "fi:Institution",
    "affiliate":                    "fi:Affiliate",
    "subsidiary":                   "fi:Subsidiary",
    "operating subsidiary":         "fi:OperatingSubsidiary",
    "broker":                       "fi:Broker",
    "dealer":                       "fi:Dealer",
    "custodian":                    "fi:Custodian",
    "fintech":                      "fi:Fintech",
    "federal reserve":              "fi:FederalReserve",
    "central bank":                 "fi:CentralBank",
    "regulator":                    "fi:Regulator",
    "regulators":                   "fi:Regulator",
    "supervisor":                   "fi:Supervisor",
    "supervisors":                  "fi:Supervisor",
    "authority":                    "fi:Authority",
    "authorities":                  "fi:Authority",
    "commission":                   "fi:Commission",
    "government":                   "fi:Government",
    "hkma":                         "fi:HKMA",
    # Parties
    "customer":                     "fi:Customer",
    "customers":                    "fi:Customer",
    "consumer":                     "fi:Consumer",
    "consumers":                    "fi:Consumer",
    "client":                       "fi:Client",
    "clients":                      "fi:Client",
    "investor":                     "fi:Investor",
    "investors":                    "fi:Investor",
    "shareholder":                  "fi:Shareholder",
    "shareholders":                 "fi:Shareholder",
    "director":                     "fi:Director",
    "directors":                    "fi:Director",
    "executive officer":            "fi:ExecutiveOfficer",
    "officer":                      "fi:Officer",
    "insider":                      "fi:Insider",
    "insiders":                     "fi:Insider",
    "beneficial owner":             "fi:BeneficialOwner",
    "lender":                       "fi:Lender",
    "lenders":                      "fi:Lender",
    "borrower":                     "fi:Borrower",
    "borrowers":                    "fi:Borrower",
    "creditor":                     "fi:Creditor",
    "counterparty":                 "fi:Counterparty",
    "counterparties":               "fi:Counterparty",
    "business":                     "fi:Business",
    "businesses":                   "fi:Business",
    "merchant":                     "fi:Merchant",
    "merchants":                    "fi:Merchant",
    "retailer":                     "fi:Retailer",
    "retailers":                    "fi:Retailer",
    "corporate customer":           "fi:CorporateCustomer",
    "individual":                   "fi:Individual",
    "individuals":                  "fi:Individual",
    "person":                       "fi:Person",
    "persons":                      "fi:Person",
    # Products and instruments
    "account":                      "fi:Account",
    "accounts":                     "fi:Account",
    "loan":                         "fi:Loan",
    "loans":                        "fi:Loan",
    "credit card":                  "fi:CreditCard",
    "credit":                       "fi:Credit",
    "deposit":                      "fi:Deposit",
    "deposits":                     "fi:Deposit",
    "mortgage":                     "fi:Mortgage",
    "mortgages":                    "fi:Mortgage",
    "fund":                         "fi:Fund",
    "funds":                        "fi:Fund",
    "mutual fund":                  "fi:MutualFund",
    "etf":                          "fi:ETF",
    "portfolio":                    "fi:Portfolio",
    "securities":                   "fi:Securities",
    "bond":                         "fi:Bond",
    "bonds":                        "fi:Bond",
    "equity":                       "fi:Equity",
    "derivative":                   "fi:Derivative",
    "derivatives":                  "fi:Derivative",
    "transaction":                  "fi:Transaction",
    "transactions":                 "fi:Transaction",
    "transfer":                     "fi:Transfer",
    "transfers":                    "fi:Transfer",
    "payment":                      "fi:Payment",
    "payments":                     "fi:Payment",
    "trade":                        "fi:Trade",
    "trades":                       "fi:Trade",
    "investment":                   "fi:Investment",
    "investments":                  "fi:Investment",
    "insurance":                    "fi:Insurance",
    "annuity":                      "fi:Annuity",
    "overdraft":                    "fi:Overdraft",
    "letter of credit":             "fi:LetterOfCredit",
    "statement":                    "fi:Statement",
    "statements":                   "fi:Statement",
    "report":                       "fi:Report",
    "reports":                      "fi:Report",
    "notice":                       "fi:Notice",
    "contract":                     "fi:Contract",
    "contracts":                    "fi:Contract",
    "agreement":                    "fi:Agreement",
    "agreements":                   "fi:Agreement",
    # Regulatory and compliance
    "regulation":                   "fi:Regulation",
    "regulations":                  "fi:Regulation",
    "rule":                         "fi:Rule",
    "rules":                        "fi:Rule",
    "requirement":                  "fi:Requirement",
    "requirements":                 "fi:Requirement",
    "policy":                       "fi:Policy",
    "policies":                     "fi:Policy",
    "guideline":                    "fi:Guideline",
    "guidelines":                   "fi:Guideline",
    "standard":                     "fi:Standard",
    "standards":                    "fi:Standard",
    "law":                          "fi:Law",
    "laws":                         "fi:Law",
    "statute":                      "fi:Statute",
    "statutes":                     "fi:Statute",
    "compliance":                   "fi:Compliance",
    "audit":                        "fi:Audit",
    "audits":                       "fi:Audit",
    "examination":                  "fi:Examination",
    "enforcement":                  "fi:EnforcementAction",
    "stress test":                  "fi:StressTest",
    "capital requirement":          "fi:CapitalRequirement",
    "capital ratio":                "fi:CapitalRatio",
    "leverage ratio":               "fi:LeverageRatio",
    "liquidity coverage":           "fi:LiquidityCoverageRatio",
    "tier 1":                       "fi:Tier1Capital",
    "tier 2":                       "fi:Tier2Capital",
    "regulation h":                 "fi:RegulationH",
    "regulation k":                 "fi:RegulationK",
    "regulation l":                 "fi:RegulationL",
    "regulation o":                 "fi:RegulationO",
    "regulation w":                 "fi:RegulationW",
    "regulation y":                 "fi:RegulationY",
    "regulation yy":                "fi:RegulationYY",
    "mifid":                        "fi:MiFID",
    "basel":                        "fi:Basel",
    "crd":                          "fi:CRD",
    "crr":                          "fi:CRR",
    "dodd-frank":                   "fi:DoddFrank",
    "section 23a":                  "fi:Section23A",
    "section 23b":                  "fi:Section23B",
    "gdpr":                         "fi:GDPR",
    "ccpa":                         "fi:CCPA",
    "kyc":                          "fi:KYC",
    "aml":                          "fi:AML",
    "know your customer":           "fi:KYC",
    "anti-money laundering":        "fi:AML",
    "customer due diligence":       "fi:CustomerDueDiligence",
    "beneficial ownership":         "fi:BeneficialOwnership",
    # Risk
    "risk":                         "fi:Risk",
    "credit risk":                  "fi:CreditRisk",
    "market risk":                  "fi:MarketRisk",
    "operational risk":             "fi:OperationalRisk",
    "liquidity risk":               "fi:LiquidityRisk",
    "systemic risk":                "fi:SystemicRisk",
    "value at risk":                "fi:ValueAtRisk",
    "var":                          "fi:ValueAtRisk",
    "capital":                      "fi:Capital",
    # Financial metrics
    "interest rate":                "fi:InterestRate",
    "exchange rate":                "fi:ExchangeRate",
    "fee":                          "fi:Fee",
    "fees":                         "fi:Fee",
    "penalty":                      "fi:Penalty",
    "penalties":                    "fi:Penalty",
    "charge":                       "fi:Charge",
    "charges":                      "fi:Charge",
    "revenue":                      "fi:Revenue",
    "net income":                   "fi:NetIncome",
    "earnings":                     "fi:Earnings",
    "profit":                       "fi:Profit",
    "loss":                         "fi:Loss",
    "cash flow":                    "fi:CashFlow",
    "balance sheet":                "fi:BalanceSheet",
    "total assets":                 "fi:TotalAssets",
    "shareholders equity":          "fi:ShareholdersEquity",
    "market cap":                   "fi:MarketCapitalization",
    "dividend":                     "fi:Dividend",
    "dividends":                    "fi:Dividend",
    "interest income":              "fi:InterestIncome",
    "net interest margin":          "fi:NetInterestMargin",
    "loan loss provision":          "fi:LoanLossProvision",
    "write-off":                    "fi:WriteOff",
    "impairment":                   "fi:Impairment",
    "outage":                       "fi:ServiceOutage",
    "fault":                        "fi:Fault",
    "error":                        "fi:Error",
    "issue":                        "fi:Issue",
    "problem":                      "fi:Problem",
    # Technology
    "internet banking":             "fi:InternetBanking",
    "two-factor authentication":    "fi:TwoFactorAuthentication",
    "authentication":               "fi:Authentication",
    "cybersecurity":                "fi:Cybersecurity",
    "api":                          "fi:API",
    "online banking":               "fi:OnlineBanking",
    "mobile banking":               "fi:MobileBanking",
    "system":                       "fi:System",
    "systems":                      "fi:System",
    "platform":                     "fi:Platform",
    "platforms":                    "fi:Platform",
    "service":                      "fi:Service",
    "services":                     "fi:Service",
    "process":                      "fi:Process",
    "procedure":                    "fi:Procedure",
    "procedures":                   "fi:Procedure",
    "framework":                    "fi:Framework",
    "frameworks":                   "fi:Framework",
    # AML and financial crime
    "money laundering":             "fi:MoneyLaundering",
    "terrorist financing":          "fi:TerroristFinancing",
    "suspicious transaction":       "fi:SuspiciousTransaction",
    "fraud":                        "fi:Fraud",
    # Structural and legal
    "disclosure":                   "fi:Disclosure",
    "disclosures":                  "fi:Disclosure",
    "privacy":                      "fi:Privacy",
    "consent":                      "fi:Consent",
    "notification":                 "fi:Notification",
    "notifications":                "fi:Notification",
    "access":                       "fi:Access",
    "protection":                   "fi:Protection",
    "protections":                  "fi:Protection",
    "right":                        "fi:Right",
    "rights":                       "fi:Rights",
    "obligation":                   "fi:Obligation",
    "obligations":                  "fi:Obligation",
    "exemption":                    "fi:Exemption",
    "exemptions":                   "fi:Exemption",
    "exception":                    "fi:Exception",
    "exceptions":                   "fi:Exception",
    "sanction":                     "fi:Sanction",
    "sanctions":                    "fi:Sanction",
    "restriction":                  "fi:Restriction",
    "restrictions":                 "fi:Restriction",
    "ownership":                    "fi:Ownership",
    "control":                      "fi:Control",
    "management":                   "fi:Management",
    "oversight":                    "fi:Oversight",
    "governance":                   "fi:Governance",
    "discretion":                   "fi:Discretion",
    "approval":                     "fi:Approval",
    "authorization":                "fi:Authorization",
    "waiver":                       "fi:Waiver",
    "interest":                     "fi:Interest",
    "jurisdiction":                 "fi:Jurisdiction",
    "jurisdictions":                "fi:Jurisdiction",
    "country":                      "fi:Country",
    "countries":                    "fi:Country",
    "terms and conditions":         "fi:TermsAndConditions",
    "terms":                        "fi:Terms",
    "conditions":                   "fi:Conditions",
    "identification":               "fi:Identification",
    "proof of identification":      "fi:ProofOfIdentification",
    "advance notice":               "fi:AdvanceNotice",
    "notice period":                "fi:NoticePeriod",
    "account closure":              "fi:AccountClosure",
    "closure":                      "fi:AccountClosure",
    "permission":                   "fi:Permission",
}

FI_OBJ_PROPS: Dict[str, str] = {
    "requires":               "fi:requires",
    "needs":                  "fi:requires",
    "regulated by":           "fi:regulatedBy",
    "subject to":             "fi:subjectTo",
    "governed by":            "fi:governedBy",
    "complies with":          "fi:compliesWith",
    "comply with":            "fi:compliesWith",
    "required to":            "fi:requiredTo",
    "prohibited from":        "fi:prohibitedFrom",
    "permitted to":           "fi:permittedTo",
    "applies to":             "fi:appliesTo",
    "issued by":              "fi:issuedBy",
    "owned by":               "fi:ownedBy",
    "controlled by":          "fi:controlledBy",
    "affiliated with":        "fi:affiliatedWith",
    "invested in":            "fi:investedIn",
    "exposed to":             "fi:exposedTo",
    "backed by":              "fi:backedBy",
    "insured by":             "fi:insuredBy",
    "provided by":            "fi:providedBy",
    "maintained by":          "fi:maintainedBy",
    "reported to":            "fi:reportedTo",
    "supervised by":          "fi:supervisedBy",
    "authorized by":          "fi:authorizedBy",
    "established by":         "fi:establishedBy",
    "depends on":             "fi:dependsOn",
    "supports":               "fi:supports",
    "provides":               "fi:provides",
    "offers":                 "fi:offers",
    "enables":                "fi:enables",
    "disables":               "fi:disables",
    "causes":                 "fi:hasCause",
    "affects":                "fi:affects",
    "uses":                   "fi:uses",
    "includes":               "fi:includes",
    "replaces":               "fi:replaces",
    "cancels":                "fi:cancels",
    "can":                    "fi:supports",
    "cannot":                 "fi:doesNotSupport",
    "prohibits":              "fi:prohibits",
    "exempts":                "fi:exempts",
    "waives":                 "fi:waives",
    "constitutes":            "fi:constitutes",
    "defines":                "fi:defines",
    "classifies":             "fi:classifies",
    "consolidates":           "fi:consolidates",
    "exceeds":                "fi:exceeds",
    "violates":               "fi:violates",
    "monitors":               "fi:monitors",
    "manages":                "fi:manages",
    "mitigates":              "fi:mitigates",
    "assesses":               "fi:assesses",
    "calculates":             "fi:calculates",
    "reports":                "fi:reports",
    "discloses":              "fi:discloses",
    "notifies":               "fi:notifies",
    "authorizes":             "fi:authorizes",
    "approves":               "fi:approves",
    "denies":                 "fi:denies",
    "issues":                 "fi:issues",
    "grants":                 "fi:grants",
    "charges":                "fi:charges",
    "invests in":             "fi:investsIn",
    "trades in":              "fi:tradesIn",
    "engages in":             "fi:engagesIn",
    "leads to":               "fi:leadsTo",
    "results in":             "fi:resultsIn",
    "triggers":               "fi:triggers",
    "drives":                 "fi:drives",
    "may":                    "fi:isPermittedTo",
    "must":                   "fi:isMandatedTo",
    "shall":                  "fi:isMandatedTo",
    "should":                 "fi:shouldComplyWith",
    "close":                  "fi:closes",
    "closes":                 "fi:closes",
    "protect":                "fi:protects",
    "protects":               "fi:protects",
    "ensure":                 "fi:ensures",
    "ensures":                "fi:ensures",
    "limit":                  "fi:limits",
    "limits":                 "fi:limits",
    "impose":                 "fi:imposes",
    "imposes":                "fi:imposes",
    "waive":                  "fi:waives",
    "implement":              "fi:implements",
    "implements":             "fi:implements",
    "maintain":               "fi:maintains",
    "maintains":              "fi:maintains",
    "obtain":                 "fi:obtains",
    "obtains":                "fi:obtains",
    "request":                "fi:requests",
    "requests":               "fi:requests",
    "verify":                 "fi:verifies",
    "verifies":               "fi:verifies",
    "have":                   "fi:has",
    "has":                    "fi:has",
    "hold":                   "fi:holds",
    "holds":                  "fi:holds",
}

FI_DATA_PROPS: Dict[str, str] = {
    "interest rate":          "fi:hasInterestRate",
    "maturity":               "fi:hasMaturity",
    "term":                   "fi:hasTerm",
    "amount":                 "fi:hasAmount",
    "limit":                  "fi:hasLimit",
    "threshold":              "fi:hasThreshold",
    "ratio":                  "fi:hasRatio",
    "capital ratio":          "fi:hasCapitalRatio",
    "notice period":          "fi:hasNoticePeriod",
    "days":                   "fi:hasDays",
    "fee":                    "fi:hasFee",
    "charge":                 "fi:hasCharge",
    "rate":                   "fi:hasRate",
    "section":                "fi:hasSection",
    "regulation":             "fi:hasRegulation",
    "requirement":            "fi:hasRequirement",
    "date":                   "fi:hasDate",
    "period":                 "fi:hasPeriod",
    "frequency":              "fi:hasFrequency",
    "percentage":             "fi:hasPercentage",
    "status":                 "fi:hasStatus",
    "score":                  "fi:hasScore",
    "rating":                 "fi:hasRating",
    "version":                "fi:hasVersion",
    "model":                  "fi:hasModel",
    "number":                 "fi:hasNumber",
    "priority":               "fi:hasPriority",
    "weight":                 "fi:hasWeight",
    "tier":                   "fi:hasTier",
    "level":                  "fi:hasLevel",
    "value":                  "fi:hasValue",
    "price":                  "fi:hasPrice",
    "volume":                 "fi:hasVolume",
    "quantity":               "fi:hasQuantity",
    "balance":                "fi:hasBalance",
    "exposure":               "fi:hasExposure",
    "risk weight":            "fi:hasRiskWeight",
    "maturity date":          "fi:hasMaturityDate",
    "effective date":         "fi:hasEffectiveDate",
    "contract end":           "fi:hasContractEndDate",
}

FI_MEREOLOGY: Dict[str, str] = {
    "part of":      "fi:partOf",
    "located in":   "fi:locatedIn",
    "inside":       "fi:locatedIn",
    "within":       "fi:locatedIn",
    "contained in": "fi:containedIn",
    "belongs to":   "fi:belongsTo",
    "component of": "fi:componentOf",
    "section of":   "fi:sectionOf",
    "subset of":    "fi:subsetOf",
    "under":        "fi:underJurisdictionOf",
    "included in":  "fi:includedIn",
    "subject to":   "fi:subjectTo",
    "governed by":  "fi:governedBy",
    "covered by":   "fi:coveredBy",
    "falls under":  "fi:fallsUnder",
}

FI_LOGICAL: Dict[str, str] = {
    "and":    "owl:intersectionOf",
    "both":   "owl:intersectionOf",
    "or":     "owl:unionOf",
    "either": "owl:unionOf",
    "not":    "owl:complementOf",
    "except": "owl:complementOf",
    "unless": "owl:complementOf",
}

# ---------------------------------------------------------------------------
# Subject blacklist and attribute qualifiers (extraction quality filters)
# ---------------------------------------------------------------------------

SUBJECT_BLACKLIST: Set[str] = {
    "i","me","my","we","our","you","your","he","him","his","she","her","it","its",
    "they","them","their","who","whom","whose","which","that","this","these","those",
    "there","here","one","ones","each","both","all","none","any","anyone","anything",
    "someone","something","everyone","everything","nobody","nothing",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","must","can","shall","ought","need",
    "dare","let","make","get","go","come","take","give","keep","put","seem","appear",
    "no","none","any","all","each","every","few","many","much","some","several","most",
    "such","other","another","same","different","more","less","enough","as","than",
    "certain","various","particular","respective","general","specific","applicable",
    "relevant","appropriate","available","possible","likely","unable","due","given",
    "however","therefore","thus","hence","moreover","furthermore","also","additionally",
    "consequently","nevertheless","nonetheless","still","yet","meanwhile","otherwise",
    "accordingly","subsequently","whether","although","though","whereas","when","where",
    "why","how","so","then","already","often","generally","typically","usually",
    "sometimes","always","never","once","since","while","after","before","until",
    "unless","except","even","only","just","not","nor","either","neither",
    "thing","things","way","ways","type","types","kind","kinds","form","forms",
    "aspect","aspects","matter","matters","case","cases","factor","factors",
    "reason","reasons","basis","point","points","note","notes","regard","regards",
    "purpose","purposes","mean","means","etc","example","examples","context",
    "use","uses","result","results","ability","fact","facts","sense","part","parts",
    "view","views","information","data",
    "mandated","depends","apply","applies","require","requires","provides","based",
    "noted","using","used","done","made","given","taken","become","include","includes",
    "involve","involves","ensure","ensures","affect","affects","consider","determines",
    "allow","allows","perform","performs","conduct","conducts","establish","address",
    "addresses","relate","relates","specify","specifies","constitute","constitutes",
    "contain","contains","follow","follows","indicate","indicates","suggest","suggests",
}

ATTRIBUTE_QUALIFIERS: Set[str] = {
    "annual","monthly","weekly","daily","quarterly",
    "minimum","maximum","mandatory","optional",
    "written","oral","electronic","domestic","foreign","international",
    "federal","state","local","regulatory","legal","statutory","contractual",
    "prior","advance","formal","informal","immediate","initial","final",
    "current","original","standard","default","approved","required",
    "permitted","prohibited","restricted","unlimited","limited","fixed",
    "variable","floating","secured","unsecured","senior","subordinated",
    "primary","secondary","joint","individual","separate","combined",
    "consolidated","gross","net","total","partial",
}

NUMERIC_RE = re.compile(
    r"^\d[\d,\.]*\s*(?:%|percent|basis\s*points?|bps?|days?|months?|years?|"
    r"weeks?|hours?|usd|eur|gbp|\$|€|£|million|billion|thousand)?$",
    re.I,
)

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
# Entity and property resolution helpers
# ---------------------------------------------------------------------------

def _resolve_entity(surface: str) -> Tuple[str, str, float]:
    """
    Map a surface-form string to the closest ontology class URI.

    Returns (label, uri, confidence): 0.95 exact phrase, 0.85 substring,
    0.75/0.70 token-level, 0.40 fallback coinage.
    Entities in SUBJECT_BLACKLIST or shorter than 3 characters score 0.0.
    """
    key = re.sub(r"\s+", " ", str(surface).lower().strip())
    if key in SUBJECT_BLACKLIST or len(key) < 3:
        return key, f"fi:{_to_camel_case(key)}", 0.0
    if key in FI_CLASSES:
        return key, FI_CLASSES[key], 0.95
    for term in sorted(FI_CLASSES, key=len, reverse=True):
        if len(term) >= 4 and term in key:
            return term, FI_CLASSES[term], 0.85
    tokens = key.split()
    for tok in tokens:
        if len(tok) >= 4 and tok not in SUBJECT_BLACKLIST:
            if tok in FI_CLASSES:
                return tok, FI_CLASSES[tok], 0.75
            if tok.endswith("s") and tok[:-1] in FI_CLASSES:
                return tok[:-1], FI_CLASSES[tok[:-1]], 0.70
    slug = _to_camel_case(surface[:24])
    return surface.strip()[:40], f"fi:{slug}", 0.40


def _resolve_obj_prop(surface: str, negated: bool) -> Tuple[str, str]:
    """Map a verb lemma to its object-property URI, applying negation if needed."""
    normalised = str(surface).lower().strip()
    for term in sorted(FI_OBJ_PROPS, key=len, reverse=True):
        if term in normalised:
            uri = FI_OBJ_PROPS[term]
            if negated:
                ns, loc = uri.split(":")
                uri = f"{ns}:doesNot{loc[0].upper()}{loc[1:]}"
            return term, uri
    slug = _to_camel_case(surface[:20])
    return surface[:30], f"fi:{slug}"


def _resolve_data_prop(surface: str) -> Tuple[str, str]:
    """Map a modifier token to its data-property URI."""
    normalised = str(surface).lower().strip()
    for term in sorted(FI_DATA_PROPS, key=len, reverse=True):
        if term in normalised:
            return term, FI_DATA_PROPS[term]
    slug = _to_camel_case(surface[:20]).capitalize()
    return surface[:30], f"fi:has{slug}"


def _token_is_negated(token: spacy.tokens.Token) -> bool:
    """Return True if *token* has a negation child dependency."""
    return any(child.dep_ == "neg" for child in token.children)


def _np_surface(token: spacy.tokens.Token) -> str:
    """Build a surface string for the noun-phrase headed by *token*."""
    lefts  = [t.text for t in token.lefts
               if t.dep_ in ("compound", "flat", "amod")
               and t.pos_ in ("NOUN", "PROPN")]
    rights = [t.text for t in token.rights
               if t.dep_ in ("compound", "flat")
               and t.pos_ in ("NOUN", "PROPN")]
    return " ".join(lefts + [token.text] + rights)


# ---------------------------------------------------------------------------
# Extractor A — Rule-Based (five syntactic passes)
# ---------------------------------------------------------------------------

def _pass1_relational(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 1: nsubj/dobj verb triples via dependency arcs."""
    triples: List[Triple] = []
    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue
        if token.dep_ not in ("ROOT", "xcomp", "ccomp", "relcl", "advcl", "conj"):
            continue
        subj_tok = None
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass") and subj_tok is None:
                subj_tok = child
        # Raise subject from head verb when xcomp has none of its own
        if subj_tok is None and token.dep_ == "xcomp":
            for child in token.head.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj_tok = child
                    break
        if subj_tok is None:
            continue
        ss, su, sc = _resolve_entity(_np_surface(subj_tok))
        if sc < CONF_THRESHOLD:
            continue
        obj_toks = [c for c in token.children
                    if c.dep_ in ("dobj", "attr", "pobj", "oprd")]
        for child in token.children:
            if child.dep_ == "prep":
                obj_toks += [c for c in child.children if c.dep_ == "pobj"]
        neg = _token_is_negated(token)
        ps, pu = _resolve_obj_prop(token.lemma_, neg)
        for ot in obj_toks:
            os_, ou, oc = _resolve_entity(_np_surface(ot))
            if oc < CONF_THRESHOLD:
                continue
            conf = round((sc + oc) / 2, 3)
            if conf < CONF_THRESHOLD:
                continue
            triples.append(Triple(
                subject=ss, predicate=ps, object=os_,
                subj_uri=su, pred_uri=pu, obj_uri=ou,
                dep_type=DepType.NOUN_ARG, negated=neg, confidence=conf,
                source=token.sent.text, extractor="rule_based",
            ))
    return triples


def _pass2_structural(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 2: regex patterns for governed-by, subject-to, and similar structures."""
    STRUCT_RE = [
        (re.compile(r"\bgoverned\s+by\b",         re.I), "fi:governedBy"),
        (re.compile(r"\bregulated\s+by\b",         re.I), "fi:regulatedBy"),
        (re.compile(r"\bsupervised\s+by\b",        re.I), "fi:supervisedBy"),
        (re.compile(r"\bsubject\s+to\b",           re.I), "fi:subjectTo"),
        (re.compile(r"\bcovered\s+by\b",           re.I), "fi:coveredBy"),
        (re.compile(r"\bowned\s+by\b",             re.I), "fi:ownedBy"),
        (re.compile(r"\bcontrolled\s+by\b",        re.I), "fi:controlledBy"),
        (re.compile(r"\baffiliated\s+with\b",      re.I), "fi:affiliatedWith"),
        (re.compile(r"\bprovided\s+by\b",          re.I), "fi:providedBy"),
        (re.compile(r"\bissued\s+by\b",            re.I), "fi:issuedBy"),
        (re.compile(r"\bpart\s+of\b",              re.I), "fi:partOf"),
        (re.compile(r"\bfalls?\s+under\b",         re.I), "fi:fallsUnder"),
        (re.compile(r"\bbelongs?\s+to\b",          re.I), "fi:belongsTo"),
        (re.compile(r"\bdefined\s+as\b",           re.I), "fi:definedAs"),
        (re.compile(r"\bclassified\s+as\b",        re.I), "rdf:type"),
        (re.compile(r"\bconsidered\s+(?:a|an)\b",  re.I), "rdf:type"),
        (re.compile(r"\bis\s+(?:a|an)\b",          re.I), "rdf:type"),
        (re.compile(r"\bare\s+(?:a|an)\b",         re.I), "rdf:type"),
    ]
    triples: List[Triple] = []
    for sent in doc.sents:
        text = sent.text
        for pat, pred_uri in STRUCT_RE:
            m = pat.search(text)
            if not m:
                continue
            before = text[:m.start()].strip()
            after  = text[m.end():].strip().split(",")[0].split(";")[0]
            sl, su, sc = "", "", 0.0
            for n in (3, 2, 1):
                sl, su, sc = _resolve_entity(" ".join(before.split()[-n:]))
                if sc >= CONF_THRESHOLD:
                    break
            ol, ou, oc = "", "", 0.0
            for n in (3, 2, 1):
                ol, ou, oc = _resolve_entity(" ".join(after.split()[:n]))
                if oc >= CONF_THRESHOLD:
                    break
            if sc < CONF_THRESHOLD or oc < CONF_THRESHOLD:
                continue
            conf = round((sc + oc) / 2, 3)
            triples.append(Triple(
                subject=sl, predicate=m.group(0).strip().lower(), object=ol,
                subj_uri=su, pred_uri=pred_uri, obj_uri=ou,
                dep_type=DepType.VERB_PROP, confidence=conf,
                source=sent.text.strip(), extractor="rule_based",
            ))
    return triples


def _pass3_attributes(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 3: numeric and qualifier attribute modifiers → data-property triples."""
    triples: List[Triple] = []
    for token in doc:
        if token.dep_ not in ("amod", "nummod", "npadvmod", "quantmod"):
            continue
        hs, hu, hc = _resolve_entity(_np_surface(token.head))
        if hc < CONF_THRESHOLD:
            continue
        mod_text = token.text.lower()
        if not (bool(NUMERIC_RE.match(mod_text)) or mod_text in ATTRIBUTE_QUALIFIERS):
            continue
        dp_s, dp_uri = _resolve_data_prop(mod_text)
        conf = round(hc * 0.85, 3)
        if conf < CONF_THRESHOLD:
            continue
        triples.append(Triple(
            subject=hs, predicate=dp_s, object=f'"{token.text}"',
            subj_uri=hu, pred_uri=dp_uri, obj_uri=f'"{token.text}"',
            dep_type=DepType.MOD_DATA, data_value=token.text, confidence=conf,
            source=token.sent.text, extractor="rule_based",
        ))
    return triples


def _pass4_conjunctions(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 4: coordinate conjunctions → owl:intersectionOf / unionOf triples."""
    triples: List[Triple] = []
    for token in doc:
        if token.dep_ != "conj":
            continue
        ls, lu, lc = _resolve_entity(_np_surface(token.head))
        rs, ru, rc = _resolve_entity(_np_surface(token))
        if lc < CONF_THRESHOLD or rc < CONF_THRESHOLD or lu == ru:
            continue
        cc_text = "and"
        for child in token.head.children:
            if child.dep_ == "cc":
                cc_text = child.text.lower()
                break
        op_uri = FI_LOGICAL.get(cc_text, "owl:intersectionOf")
        conf   = round((lc + rc) / 2, 3)
        if conf < CONF_THRESHOLD:
            continue
        triples.append(Triple(
            subject=ls, predicate=cc_text, object=rs,
            subj_uri=lu, pred_uri=op_uri, obj_uri=ru,
            dep_type=DepType.CONJ_LOGIC, logical_op=op_uri, confidence=conf,
            source=token.sent.text, extractor="rule_based",
        ))
    return triples


def _pass5_causal(doc: spacy.tokens.Doc) -> List[Triple]:
    """Pass 5: causal-chain patterns (leads-to, results-in, triggers, etc.)."""
    CAUSAL_RE = re.compile(
        r"(?P<subj>[A-Za-z ]{3,45}?)\s+"
        r"(?P<verb>leads?\s+to|results?\s+in|triggers?|causes?|drives?|produces?|"
        r"increases?|decreases?|reduces?)\s+"
        r"(?P<obj>[A-Za-z ]{3,45})",
        re.I,
    )
    CAUSAL_URI: Dict[str, str] = {
        "lead":    "fi:leadsTo",   "leads":    "fi:leadsTo",
        "result":  "fi:resultsIn", "results":  "fi:resultsIn",
        "trigger": "fi:triggers",  "triggers": "fi:triggers",
        "cause":   "fi:hasCause",  "causes":   "fi:hasCause",
        "drive":   "fi:drives",    "drives":   "fi:drives",
        "produce": "fi:produces",  "produces": "fi:produces",
        "increase":"fi:increases", "increases":"fi:increases",
        "decrease":"fi:decreases", "decreases":"fi:decreases",
        "reduce":  "fi:reduces",   "reduces":  "fi:reduces",
    }
    triples: List[Triple] = []
    for sent in doc.sents:
        for m in CAUSAL_RE.finditer(sent.text):
            sl, su, sc = _resolve_entity(m.group("subj").strip())
            ol, ou, oc = _resolve_entity(m.group("obj").strip())
            if sc < CONF_THRESHOLD or oc < CONF_THRESHOLD:
                continue
            verb_key = m.group("verb").split()[0].lower()
            pred_uri = CAUSAL_URI.get(verb_key, "fi:leadsTo")
            conf     = round((sc + oc) / 2, 3)
            triples.append(Triple(
                subject=sl, predicate=m.group("verb").strip().lower(), object=ol,
                subj_uri=su, pred_uri=pred_uri, obj_uri=ou,
                dep_type=DepType.PREP_MERO, confidence=conf,
                source=sent.text.strip(), extractor="rule_based",
            ))
    return triples


def extract_rule_based(text: str) -> List[Triple]:
    """
    Run all five rule-based passes over *text* and return a deduplicated,
    confidence-sorted list of triples.  Input is capped at 10,000 characters.
    """
    if not str(text).strip():
        return []
    doc  = get_nlp()(str(text)[:10000])
    seen: Set[str] = set()
    out:  List[Triple] = []
    for pass_fn in (
        _pass1_relational,
        _pass2_structural,
        _pass3_attributes,
        _pass4_conjunctions,
        _pass5_causal,
    ):
        for triple in pass_fn(doc):
            if triple.key() not in seen:
                seen.add(triple.key())
                out.append(triple)
    del doc
    out.sort(key=lambda t: t.confidence, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Extractor B — spaCy General SVO
# ---------------------------------------------------------------------------

def extract_spacy_general(text: str) -> List[Triple]:
    """
    General SVO extraction over all VERB/AUX tokens in the dependency tree.
    Predicate URIs are raw lemmas prefixed with ``rel:``; no ontology mapping.
    """
    if not str(text).strip():
        return []
    doc  = get_nlp()(str(text)[:10000])
    seen: Set[str] = set()
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
                dep_type=DepType.GENERAL, negated=neg, confidence=0.70,
                source=token.sent.text, extractor="spacy_general",
            )
            if triple.key() not in seen:
                seen.add(triple.key())
                out.append(triple)
    del doc
    return out


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------

EXTRACTORS: Dict[str, callable] = {
    "rule_based":    extract_rule_based,
    "spacy_general": extract_spacy_general,
}

EXTRACTOR_HEADER_COLORS: Dict[str, str] = {
    "rule_based":    "1F497D",
    "spacy_general": "4A235A",
}


def extract_all(text: str) -> Dict[str, List[Triple]]:
    """Run all registered extractors over *text* and return results by name."""
    return {name: fn(text) for name, fn in EXTRACTORS.items()}


# ---------------------------------------------------------------------------
# TKF metrics (KD and triple counts only — this module)
# ---------------------------------------------------------------------------

def knowledge_density(triples: List[Triple], text: str) -> float:
    """
    Knowledge Density (KD) = high-confidence triples / total tokens.

    A triple is considered typed when its confidence meets or exceeds
    KD_CONFIDENCE_THRESHOLD.  Token count uses simple whitespace splitting.
    """
    tokens = [w for w in str(text).lower().split() if w.strip()]
    typed  = sum(1 for t in triples if t.confidence >= KD_CONFIDENCE_THRESHOLD)
    return round(safe_div(typed, len(tokens)), 4) if tokens else 0.0


# Metric keys produced by this module
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
    Compute KD, TKF (= KD in this module), and triple counts.

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
# Intermediate file helpers (streaming to disk)
# ---------------------------------------------------------------------------

def _tmp_path(tmp_dir: Path, model: str, kind: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", model)[:40]
    return tmp_dir / f"{safe}__{kind}.csv.gz"


def _append_rows(path: Path, rows: List[Dict], first_write: bool) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if first_write:
        df.to_csv(path, index=False, compression="gzip", mode="w")
    else:
        df.to_csv(path, index=False, compression="gzip", mode="a", header=False)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, compression="gzip", low_memory=False)


def _cleanup_tmp(tmp_dir: Path) -> None:
    """Remove all compressed CSV files and the temporary directory."""
    if tmp_dir.exists():
        for f in tmp_dir.glob("*.csv.gz"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_ground_truth(xl: pd.ExcelFile) -> Dict[str, str]:
    """
    Read the ground-truth sheet and return a mapping of Query_ID → answer text.

    The sheet must contain columns matching:
        Query_ID    (regex: query.?id | qid | ^id$ | s.no | sno)
        ground_truth (regex: ground_truth | groundtruth)
    """
    if GROUND_TRUTH_SHEET not in xl.sheet_names:
        raise FileNotFoundError(
            f"Sheet '{GROUND_TRUTH_SHEET}' not found. "
            f"Available: {xl.sheet_names}"
        )
    df = pd.read_excel(xl, sheet_name=GROUND_TRUTH_SHEET, header=0)
    qid_col = next(
        (c for c in df.columns
         if re.search(r"query.?id|qid|^id$|s\.no|sno", str(c).lower())),
        None,
    )
    gt_col = next(
        (c for c in df.columns
         if "ground_truth" in str(c).lower() or "groundtruth" in str(c).lower()),
        None,
    )
    if qid_col is None or gt_col is None:
        raise ValueError(
            f"Ground-truth sheet requires Query_ID and ground_truth columns. "
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


def resolve_method_cols(df: pd.DataFrame) -> Tuple[List, List[str]]:
    """
    Identify response columns (those ending with RESPONSE_SUFFIX) and return
    (column_names, method_labels).  Falls back to all non-metadata columns.
    """
    suffix     = RESPONSE_SUFFIX.lower()
    meta_lower = {
        str(c).lower() for c in df.columns
        if re.search(r"query.?id|qid|^id$|question|prompt|input|s\.no|sno",
                     str(c).lower())
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
# Row-level evaluator
# ---------------------------------------------------------------------------

def evaluate_row(
    ground_truth: str,
    method_data: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, Dict[str, List[Triple]]], List[Triple]]:
    """
    Extract gold triples from *ground_truth* and candidate triples from each
    method response, then compute KD/TKF/count metrics.

    Returns
    -------
    per_method : method → extractor → metric dict
    inventory  : method → extractor → list of extracted triples
    gold       : gold triples (rule-based on ground truth)
    """
    gold      = extract_rule_based(ground_truth)
    zero      = {k: 0.0 for k in ALL_METRICS}
    per_method: Dict[str, Dict[str, Dict]] = {}
    inventory:  Dict[str, Dict[str, List[Triple]]] = {}

    for method, resp_raw in method_data.items():
        resp = str(resp_raw).strip()
        if resp.lower() in ("", "nan", "none", "n/a"):
            per_method[method] = {ext: {**zero, "missing": True} for ext in EXTRACTORS}
            inventory[method]  = {ext: [] for ext in EXTRACTORS}
            continue
        ext_triples = extract_all(resp)
        per_method[method] = {}
        for ext_name, candidates in ext_triples.items():
            per_method[method][ext_name] = {
                **compute_metrics(gold, candidates, resp, ext_name),
                "missing": False,
            }
        inventory[method] = ext_triples

    return per_method, inventory, gold


# ---------------------------------------------------------------------------
# Sheet processor (streams rows to disk)
# ---------------------------------------------------------------------------

def process_sheet(
    model: str,
    df: pd.DataFrame,
    gt_map: Dict[str, str],
    tmp_dir: Path,
) -> Tuple[List[Dict], List[Dict], List[str], List[Tuple]]:
    """
    Process one model sheet.  Streams detail, triple, and count rows to
    compressed CSV files in *tmp_dir*.

    Returns in-memory only:
        agg_rows      — aggregated TKF stats (method × extractor)
        gt_triple_rows — ground-truth triples (deduped by caller)
        method_names  — list of method names for console reporting
        pm_log        — per-row (qid, question, gt, methods, per_method) tuples
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    method_cols, method_names = resolve_method_cols(df)
    q_col = next(
        (c for c in df.columns
         if "question" in str(c).lower() or "original_query" in str(c).lower()),
        df.columns[1],
    )
    qid_col = next(
        (c for c in df.columns
         if re.search(r"query.?id|qid|^id$|s\.no|sno", str(c).lower())),
        None,
    )

    print(f"\n    Model   : {model}")
    print(f"    Methods : {method_names}")
    print(f"    Rows    : {len(df)}")

    accum = {
        m: {ext: collections.defaultdict(list) for ext in EXTRACTORS}
        for m in method_names
    }

    gt_triple_rows: List[Dict]  = []
    agg_rows:       List[Dict]  = []
    pm_log:         List[Tuple] = []
    gt_saved:       Set[str]    = set()
    missing_gt = 0

    buf_det: List[Dict] = []
    buf_tr:  List[Dict] = []
    buf_cnt: List[Dict] = []
    _first = {"det": True, "tr": True, "cnt": True}

    def _flush(force: bool = False) -> None:
        nonlocal buf_det, buf_tr, buf_cnt
        if not force and len(buf_det) < _FLUSH_EVERY:
            return
        for kind, buf in (("det", buf_det), ("tr", buf_tr), ("cnt", buf_cnt)):
            if buf:
                _append_rows(_tmp_path(tmp_dir, model, kind), buf, _first[kind])
                _first[kind] = False
        buf_det = []; buf_tr = []; buf_cnt = []
        gc.collect()

    for ri, row in df.iterrows():
        question = str(row.get(q_col, "")).strip()
        if not question or question.lower() == "nan":
            continue
        qid = (
            str(row[qid_col]).strip()
            if qid_col and pd.notna(row.get(qid_col))
            else str(ri + 1)
        )
        gt = gt_map.get(qid, "")
        if not gt:
            missing_gt += 1

        method_data = {
            name: (str(row[col]).strip() if pd.notna(row.get(col)) else "")
            for col, name in zip(method_cols, method_names)
        }

        per_method, inventory, gold = evaluate_row(gt, method_data)

        # Collect GT triples once per query_id
        if qid not in gt_saved:
            for t in gold:
                gt_triple_rows.append({"query_id": qid, **t.as_dict()})
            gt_saved.add(qid)

        pm_log.append((qid, question, gt, method_names, per_method))

        for method in method_names:
            for ext_name in EXTRACTORS:
                m_ = per_method.get(method, {}).get(ext_name, {})
                clean = {k: v for k, v in m_.items() if k != "missing"}

                buf_det.append({
                    "model":        model,
                    "query_id":     qid,
                    "question":     question[:120],
                    "ground_truth": gt[:120],
                    "method":       method,
                    "extractor":    ext_name,
                    "response":     method_data.get(method, "")[:150],
                    "gt_available": bool(gt),
                    "missing":      m_.get("missing", False),
                    **clean,
                })

                for k, v in clean.items():
                    if isinstance(v, (int, float)):
                        accum[method][ext_name][k].append(float(v))

                candidates = inventory.get(method, {}).get(ext_name, [])
                buf_cnt.append({
                    "model":          model,
                    "query_id":       qid,
                    "method":         method,
                    "extractor":      ext_name,
                    "triple_count_c": len(candidates),
                    "triple_count_g": len(gold),
                    "ratio_c_over_g": (
                        round(len(candidates) / len(gold), 4) if gold else None
                    ),
                })

            # Candidate triple rows (all extractors)
            for ext_name, triples in inventory.get(method, {}).items():
                for t in triples:
                    buf_tr.append({
                        "model":    model,
                        "query_id": qid,
                        "method":   method,
                        **t.as_dict(),
                    })

        del gold, inventory, per_method
        _flush()

    _flush(force=True)

    if missing_gt:
        print(f"    WARNING: {missing_gt} rows had no ground-truth entry.")

    # Build aggregation rows from in-memory float accumulators
    for method in method_names:
        for ext_name in EXTRACTORS:
            acc = accum[method][ext_name]
            agg: Dict = {"model": model, "method": method, "extractor": ext_name}
            for k in ALL_METRICS:
                vals = acc.get(k, [])
                if vals:
                    agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                    agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
                    agg[f"{k}_min"]  = round(float(np.min(vals)),  4)
                    agg[f"{k}_max"]  = round(float(np.max(vals)),  4)
                else:
                    agg[f"{k}_mean"] = agg[f"{k}_std"] = \
                    agg[f"{k}_min"]  = agg[f"{k}_max"] = 0.0
            agg_rows.append(agg)

    del accum
    gc.collect()

    return agg_rows, gt_triple_rows, method_names, pm_log


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
_CENT_ALIGN  = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _write_ws(
    ws,
    df: pd.DataFrame,
    header_hex: str = "1F3864",
    score_cols: Optional[List[str]] = None,
) -> None:
    """Write *df* to openpyxl worksheet *ws* with header styling and score colouring."""
    hfill  = PatternFill("solid", fgColor=header_hex)
    cols   = list(df.columns)
    sc_set = set(score_cols or [])

    for ci, col in enumerate(cols, 1):
        cell = ws.cell(1, ci, str(col))
        cell.fill, cell.font, cell.alignment = hfill, _HEADER_FONT, _CENT_ALIGN

    for ri, row in enumerate(df.itertuples(index=False), 2):
        is_mean = str(row[0]) == "MEAN"
        for ci, val in enumerate(row, 1):
            display  = val if not (isinstance(val, float) and math.isnan(val)) else ""
            cell     = ws.cell(ri, ci, display)
            cell.font = _BOLD_FONT if is_mean else _BODY_FONT
            if is_mean:
                cell.fill = _SCORE_FILLS["mean"]
                continue
            if cols[ci - 1] in sc_set:
                try:
                    v = float(val)
                    cell.fill = (
                        _SCORE_FILLS["high"] if v >= 0.70 else
                        _SCORE_FILLS["mid"]  if v >= 0.40 else
                        _SCORE_FILLS["low"]
                    )
                except (TypeError, ValueError):
                    pass

    for ci, col in enumerate(cols, 1):
        width = max(
            len(str(col)),
            int(df[col].astype(str).str.len().max()) if len(df) else 8,
        )
        ws.column_dimensions[get_column_letter(ci)].width = min(width + 2, 40)
    ws.freeze_panes = "D2"


def _ext_pivot(
    df_det: pd.DataFrame,
    model: str,
    method: str,
    metric: str,
) -> Optional[pd.DataFrame]:
    """
    Build a query_id × extractor pivot for one (model, method, metric).
    Appends a MEAN summary row.
    """
    rows = df_det[(df_det["model"] == model) & (df_det["method"] == method)]
    if rows.empty:
        return None

    data: Dict = {}
    for _, r in rows.iterrows():
        qid = r["query_id"]
        if qid not in data:
            data[qid] = {"query_id": qid, "question": str(r["question"])[:80]}
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

    summary: Dict = {"query_id": "MEAN", "question": "-- column mean --"}
    for c in ext_cols:
        try:    summary[c] = round(float(numeric[c].mean()), 4)
        except: summary[c] = ""
    for c in ("best_score", "extractor_spread"):
        try:    summary[c] = round(float(df[c].mean()), 4)
        except: summary[c] = ""
    summary["best_extractor"] = ""

    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def write_excel(
    output_path: Path,
    all_ag:      List[Dict],
    models:      List[str],
    all_gt_tr:   List[Dict],
    tmp_dir:     Path,
) -> None:
    """
    Write the full evaluation workbook.

    Sheet structure
    ---------------
    Pivot sheets        — query_id × extractor for each key metric
    Detail sheets       — row-level metrics per model × extractor
    Triples_C__{model}  — candidate triples per model
    Triples_G           — ground-truth triples (deduped)
    TripleCounts        — C/G count ratios
    Leaderboard         — global TKF ranking
    Summary__{model}    — aggregated TKF stats per method × extractor
    """
    wb = Workbook()
    wb.remove(wb.active)

    KEY_METRICS = ["tkf_score", "kd_tkf", "triple_count_c", "triple_count_g"]

    # Build model → methods lookup from aggregation rows
    model_methods: Dict[str, List[str]] = {}
    for a in all_ag:
        model_methods.setdefault(a["model"], [])
        if a["method"] not in model_methods[a["model"]]:
            model_methods[a["model"]].append(a["method"])

    # Per-model sheets
    for model in models:
        sm = re.sub(r"[^a-zA-Z0-9]", "_", model)[:10]

        df_det = _read_csv(_tmp_path(tmp_dir, model, "det"))
        if not df_det.empty:
            # Pivot sheets per method × metric
            for method in model_methods.get(model, []):
                smth = re.sub(r"[^a-zA-Z0-9]", "_", method)[:14]
                for metric in KEY_METRICS:
                    piv = _ext_pivot(df_det, model, method, metric)
                    if piv is None:
                        continue
                    ws = wb.create_sheet(f"{sm}_{smth}__{metric}"[:31])
                    ext_cols = [
                        c for c in piv.columns
                        if c not in ("query_id", "question",
                                     "best_extractor", "best_score", "extractor_spread")
                    ]
                    _write_ws(
                        ws, piv, header_hex="1F3864",
                        score_cols=ext_cols + ["best_score", "extractor_spread"],
                    )

            # Detail sheet per extractor
            for ext in list(EXTRACTORS.keys()):
                df_ext = df_det[df_det["extractor"] == ext]
                if df_ext.empty:
                    continue
                ws = wb.create_sheet(f"Detail_{sm}_{ext[:14]}"[:31])
                _write_ws(
                    ws, df_ext.reset_index(drop=True),
                    header_hex=EXTRACTOR_HEADER_COLORS.get(ext, "1F3864"),
                    score_cols=["tkf_score", "kd_tkf"],
                )

        del df_det
        gc.collect()

        # Candidate triples per model
        df_tr = _read_csv(_tmp_path(tmp_dir, model, "tr"))
        if not df_tr.empty:
            ws = wb.create_sheet(f"Triples_C__{sm}"[:31])
            _write_ws(ws, df_tr.reset_index(drop=True), header_hex="375623")
        del df_tr
        gc.collect()

        # Triple counts per model
        df_cnt = _read_csv(_tmp_path(tmp_dir, model, "cnt"))
        if not df_cnt.empty:
            for method in df_cnt["method"].unique():
                df_m    = df_cnt[df_cnt["method"] == method].copy()
                smth    = re.sub(r"[^a-zA-Z0-9]", "_", str(method))[:12]
                piv_cnt = df_m.pivot_table(
                    index="query_id", columns="extractor",
                    values="triple_count_c", aggfunc="first",
                ).reset_index()
                piv_cnt.columns.name = None
                g_vals = (
                    df_m.groupby("query_id")["triple_count_g"]
                    .first().reset_index()
                )
                piv_cnt = piv_cnt.merge(g_vals, on="query_id", how="left")
                piv_cnt.rename(columns={"triple_count_g": "count_G"}, inplace=True)
                ext_cols = [c for c in piv_cnt.columns
                            if c not in ("query_id", "count_G")]
                piv_cnt.rename(
                    columns={c: f"count_C_{c}" for c in ext_cols}, inplace=True
                )
                ws_p = wb.create_sheet(f"Cnt_{sm}_{smth}"[:31])
                _write_ws(
                    ws_p, piv_cnt.reset_index(drop=True), header_hex="843C0C",
                    score_cols=[c for c in piv_cnt.columns if c.startswith("count_")],
                )
        del df_cnt
        gc.collect()

        # Aggregation summary
        df_s = pd.DataFrame([r for r in all_ag if r["model"] == model])
        if not df_s.empty:
            ws = wb.create_sheet(f"Summary__{sm}")
            _write_ws(ws, df_s, header_hex="17375E",
                      score_cols=["tkf_score_mean", "kd_tkf_mean"])
        del df_s
        gc.collect()

    # Global sheets

    # Triples_G — ground truth (deduped)
    if all_gt_tr:
        df_gt = (
            pd.DataFrame(all_gt_tr)
            .drop_duplicates(
                subset=["query_id", "subject_uri", "predicate_uri", "object_uri"]
            )
            .sort_values("query_id")
        )
        ws_gt = wb.create_sheet("Triples_G")
        _write_ws(ws_gt, df_gt.reset_index(drop=True), header_hex="1F497D")
        del df_gt
        gc.collect()

    # TripleCounts — all models combined
    cnt_frames = []
    for model in models:
        df_c = _read_csv(_tmp_path(tmp_dir, model, "cnt"))
        if not df_c.empty:
            cnt_frames.append(df_c)
    if cnt_frames:
        df_cnt_all = pd.concat(cnt_frames, ignore_index=True)
        ws_cnt = wb.create_sheet("TripleCounts")
        _write_ws(
            ws_cnt, df_cnt_all.reset_index(drop=True), header_hex="17375E",
            score_cols=["triple_count_c", "triple_count_g", "ratio_c_over_g"],
        )
        del df_cnt_all
    del cnt_frames
    gc.collect()

    # Leaderboard
    df_lb = (
        pd.DataFrame(all_ag)
        .sort_values("tkf_score_mean", ascending=False)
        .reset_index(drop=True)
    )
    df_lb.insert(0, "rank", range(1, len(df_lb) + 1))
    ws = wb.create_sheet("Leaderboard")
    _write_ws(ws, df_lb, header_hex="1F3864",
              score_cols=["tkf_score_mean", "kd_tkf_mean"])
    del df_lb
    gc.collect()

    wb.save(str(output_path))
    print(f"\n    Output saved : {output_path}  ({len(wb.worksheets)} sheets)")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for a value in [0, 1]."""
    filled = int(max(0.0, min(1.0, float(value))) * width)
    return "=" * filled + "-" * (width - filled)


def print_row_report(
    qid: str,
    question: str,
    gt: str,
    method_names: List[str],
    per_method: Dict,
) -> None:
    """Print a formatted per-row evaluation summary to stdout."""
    EXT_NAMES = list(EXTRACTORS.keys())
    COL_W, EXT_W = 24, 16
    ROWS = [
        ("KD (typed/tokens)", "kd_tkf"),
        ("TKF score",         "tkf_score"),
        ("Triple count C",    "triple_count_c"),
        ("Triple count G",    "triple_count_g"),
    ]

    print(f"\n  Q{qid} {'_' * 60}")
    print(f"  Question     : {question[:110]}")
    print(f"  Ground truth : {gt[:110]}")

    for method in method_names:
        print(f"\n  Method: {method}")
        hdr = f"  {'Metric':<{COL_W}}" + "".join(f"{e:>{EXT_W}}" for e in EXT_NAMES)
        print(hdr)
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
                f"    #{rank} {ext:<22} TKF={tkf:.4f} [{_bar(tkf)}]"
                f"  KD={kd:.4f}{miss}"
            )


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Financial industry triple extraction and TKF evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / DEFAULT_INPUT_FILENAME,
        help=(
            "Path to the input Excel file or the directory containing it. "
            f"If a directory is given, looks for '{DEFAULT_INPUT_FILENAME}' inside it."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the output Excel file. "
            "Defaults to <input_directory>/pipeline2a_finance_output.xlsx."
        ),
    )
    parser.add_argument(
        "--nlp-model",
        default=NLP_MODEL,
        metavar="MODEL",
        help="spaCy model name to use for dependency parsing.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the temporary CSV files after writing the Excel output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve input path: accept a .xlsx file or a directory containing it
    input_path: Path = args.input.resolve()
    if input_path.is_dir():
        input_path = input_path / DEFAULT_INPUT_FILENAME
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Resolve output path: default alongside the input file
    output_path: Path = (
        args.output.resolve()
        if args.output
        else input_path.parent / DEFAULT_OUTPUT_FILENAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary streaming directory lives next to the output file
    tmp_dir: Path = output_path.parent / "_p2a_finance_tmp"

    # Allow CLI override of the spaCy model
    global NLP_MODEL
    NLP_MODEL = args.nlp_model

    print(f"\n{'=' * 80}")
    print("  Pipeline 2A — Triple Extraction [Financial Industry]")
    print(f"  Extractors : {list(EXTRACTORS.keys())}")
    print(f"  Input      : {input_path}")
    print(f"  Output     : {output_path}")
    print(f"  Metrics    : KD, TKF, triple counts (GED/SF/PAR/MA not in scope)")
    print(f"  W_KD       : {W_KD}")
    print(f"  Conf gate  : {CONF_THRESHOLD}")
    print(f"  Flush every: {_FLUSH_EVERY} rows  |  Tmp dir: {tmp_dir}")
    print(f"{'=' * 80}")

    print("\n  Loading spaCy model ...")
    get_nlp()
    print(f"    Loaded '{NLP_MODEL}'")

    xl     = pd.ExcelFile(input_path)
    gt_map = load_ground_truth(xl)
    sheets = [s for s in xl.sheet_names if s not in SKIP_SHEETS]
    print(f"\n  Model sheets: {sheets}")

    all_ag:    List[Dict] = []
    all_gt_tr: List[Dict] = []
    models:    List[str]  = []
    gt_saved_global: Set[str] = set()

    for sheet in sheets:
        model = sheet.strip()
        models.append(model)
        df = pd.read_excel(input_path, sheet_name=sheet, header=0)
        print(f"\n  {'_' * 78}")
        print(f"  Processing: {model}  ({len(df)} rows)")

        agg_rows, gt_triple_rows, method_names, pm_log = process_sheet(
            model, df, gt_map, tmp_dir
        )
        all_ag.extend(agg_rows)

        for r in gt_triple_rows:
            key = (
                r["query_id"], r.get("subject_uri", ""),
                r.get("predicate_uri", ""), r.get("object_uri", ""),
            )
            if key not in gt_saved_global:
                gt_saved_global.add(key)
                all_gt_tr.append(r)

        for qid, q, gt, mnames, pm in pm_log:
            print_row_report(qid, q, gt, mnames, pm)

        del pm_log, gt_triple_rows, df
        gc.collect()

        # Per-model aggregate summary
        print(f"\n  Aggregate [{model}]:")
        print(
            f"  {'Method':<22}{'Extractor':<18}"
            f"{'TKF':>10}{'KD':>10}{'Cnt_C':>8}{'Cnt_G':>8}"
        )
        print(f"  {'_' * 68}")
        for a in agg_rows:
            print(
                f"  {a['method']:<22}{a['extractor']:<18}"
                f"{a.get('tkf_score_mean', 0):>10.4f}"
                f"{a.get('kd_tkf_mean',   0):>10.4f}"
                f"{a.get('triple_count_c_mean', 0):>8.1f}"
                f"{a.get('triple_count_g_mean', 0):>8.1f}"
            )

    # Global leaderboard
    print(f"\n\n{'=' * 80}")
    print("  GLOBAL LEADERBOARD  —  Pipeline 2A Finance  (TKF score)")
    print(f"{'=' * 80}")
    df_lb = (
        pd.DataFrame(all_ag)
        .sort_values("tkf_score_mean", ascending=False)
        .reset_index(drop=True)
    )
    print(
        f"  {'#':<4}{'Model':<14}{'Method':<22}{'Extractor':<18}"
        f"{'TKF':>10}{'KD':>10}{'Cnt_C':>8}{'Cnt_G':>8}"
    )
    print(f"  {'_' * 86}")
    for i, row in df_lb.iterrows():
        print(
            f"  {i + 1:<4}{str(row['model']):<14}{str(row['method']):<22}"
            f"{str(row['extractor']):<18}"
            f"{row.get('tkf_score_mean', 0):>10.4f}"
            f"{row.get('kd_tkf_mean',   0):>10.4f}"
            f"{row.get('triple_count_c_mean', 0):>8.1f}"
            f"{row.get('triple_count_g_mean', 0):>8.1f}"
        )

    print("\n  Writing Excel output ...")
    write_excel(output_path, all_ag, models, all_gt_tr, tmp_dir)

    if not args.keep_tmp:
        print("  Cleaning up temporary files ...")
        _cleanup_tmp(tmp_dir)

    print(f"\n{'=' * 80}")
    print("  Pipeline 2A Finance complete.")
    print(f"  Output : {output_path}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
