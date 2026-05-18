"""
build_vetting_dataset.py

Generate source-grounded SFT rows from live Bangladesh contract, labor,
and policy source chunks. Synthetic clauses are used only as prompts;
the answer content is grounded in retrieved source text and citations.

This builder produces a general-purpose training set covering new company
setup, partnerships and joint ventures, expansion (branch/subsidiary,
M&A, restructuring), routine commercial contracts, and general labor /
HR vetting for any Bangladesh business. EPZ/BEPZA, government
procurement, and foreign-investor orientation are emitted as
specialisations only when the underlying source chunk is clearly about
those regimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from typing import Iterable, Optional


DISCLAIMER = (
    "This is automated legal and business-compliance exploration support, "
    "not legal advice. Verify the cited source and consult a qualified "
    "Bangladeshi advocate or relevant professional before acting."
)

BILINGUAL_TERMS = [
    {"english": "retrenchment", "bangla": "ছাঁটাই", "note": "workforce reduction / retrenchment concept"},
    {"english": "lay-off", "bangla": "লে-অফ", "note": "temporary inability to provide work"},
    {"english": "discharge", "bangla": "ডিসচার্জ", "note": "termination on grounds such as incapacity/continued ill-health where applicable"},
    {"english": "dismissal", "bangla": "বরখাস্ত", "note": "punitive removal after misconduct procedure where applicable"},
    {"english": "misconduct", "bangla": "অসদাচরণ", "note": "disciplinary trigger term"},
    {"english": "wages", "bangla": "মজুরি", "note": "pay/wage term used in labor materials"},
    {"english": "earned leave", "bangla": "অর্জিত ছুটি", "note": "leave earned through service"},
    {"english": "appointment letter", "bangla": "নিয়োগপত্র", "note": "employment document"},
    {"english": "worker", "bangla": "শ্রমিক", "note": "labor-law worker term"},
    {"english": "employer", "bangla": "মালিক", "note": "employer/owner term in labor context"},
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s build_vetting_dataset %(message)s",
)
log = logging.getLogger("build_vetting_dataset")


def load_chunks(path: str) -> list[dict]:
    chunks: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def short(text: str, n: int = 120) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:n]


def stable_index(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def chunk_blob(chunk: dict) -> str:
    return f"{chunk.get('title', '')}\n{chunk.get('text', '')}".lower()


def title_blob(chunk: dict) -> str:
    return (chunk.get("title") or "").lower()


def has_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def is_negotiable_source(chunk: dict) -> bool:
    return "negotiable instruments" in title_blob(chunk)


def is_contract_act(chunk: dict) -> bool:
    return "contract act" in title_blob(chunk)


def is_sale_of_goods(chunk: dict) -> bool:
    return "sale of goods" in title_blob(chunk)


def is_partnership_act(chunk: dict) -> bool:
    return "partnership act" in title_blob(chunk)


def is_arbitration_act(chunk: dict) -> bool:
    return "arbitration" in title_blob(chunk) or "সালিস" in title_blob(chunk)


def is_specific_relief_act(chunk: dict) -> bool:
    return "specific relief" in title_blob(chunk)


COMPANY_SETUP_TERMS = (
    "incorporat", "registration", "memorandum", "articles",
    "registered office", "share capital", "certificate of incorporation",
    "formation", "name", "সংঘস্মারক", "সংঘবিধি", "নাম",
    "নিবন্ধিকৃত কার্যালয়", "শেয়ার-মূলধন", "নিগমিত", "গঠন করিতে",
)

COMPANY_GOVERNANCE_TERMS = (
    "share", "shareholder", "member", "director", "board", "meeting",
    "resolution", "register", "minute", "allotment", "transfer", "capital",
    "memorandum", "articles", "শেয়ার", "সদস্য", "পরিচালক", "সভা", "সিদ্ধান্ত",
    "রেজিষ্টার", "মূলধন", "হস্তান্তর", "বরাদ্দ", "সংঘস্মারক", "সংঘবিধি",
)

EXPANSION_TERMS = (
    "alter", "change", "objects", "capital", "share", "transfer", "branch",
    "subsidiary", "amalgamation", "merger", "acquisition", "arrangement",
    "compromise", "reconstruction", "restructur", "foreign exchange",
    "investment", "পরিবর্তন", "উদ্দেশ্য", "মূলধন", "শেয়ার", "হস্তান্তর",
    "শাখা", "একত্রীকরণ", "পুনর্গঠন", "বিনিয়োগ",
)

CONTRACT_CORE_TERMS = (
    "proposal", "acceptance", "consideration", "lawful", "consent",
    "agreement", "contract", "breach", "damages", "indemn", "guarantee",
    "agent", "agency", "arbitration", "sale", "goods", "delivery",
    "warranty", "condition", "specific performance", "চুক্তি", "সালিস",
)

POLICY_TERMS = (
    "consumer", "refund", "replacement", "return", "warranty", "guarantee",
    "defect", "merchantable", "fitness for purpose", "quality", "delivery",
    "complaint", "grievance", "service level", "terms of service", "privacy",
    "personal data", "support", "repair", "ভোক্তা", "অভিযোগ", "প্রতিস্থাপন",
    "ওয়ারেন্টি", "গ্যারান্টি", "সেবা",
)

HR_POLICY_TERMS = (
    "wage", "leave", "maternity", "overtime", "termination", "dismiss",
    "discharge", "retrench", "misconduct", "grievance", "safety", "worker",
    "মজুরি", "ছুটি", "মাতৃত্ব", "বরখাস্ত", "ছাঁটাই", "শ্রমিক", "নিরাপত্তা",
)

COMPANY_SETUP_EXCLUDE_TERMS = (
    "winding", "liquidat", "mortgage", "charge", "loan", "debenture",
    "creditor", "court", "offence", "penalty", "prospectus", "ঋণ", "বন্ধক",
    "চার্জ", "ডিবেঞ্চার", "পাওনাদার", "আদালত", "অবসায়ন", "অর্থদণ্ড",
    "প্রসপেক্টাস",
)


def supports_company_setup(chunk: dict) -> bool:
    blob = chunk_blob(chunk)
    return (
        chunk.get("source_family") == "statutory_company"
        and has_any(blob, COMPANY_SETUP_TERMS)
        and not has_any(blob, COMPANY_SETUP_EXCLUDE_TERMS)
    )


def supports_partnership_jv(chunk: dict) -> bool:
    blob = chunk_blob(chunk)
    if is_negotiable_source(chunk):
        return False
    if is_partnership_act(chunk):
        return True
    if chunk.get("source_family") == "statutory_company" and has_any(blob, COMPANY_GOVERNANCE_TERMS):
        return True
    if is_contract_act(chunk) and has_any(blob, ("agreement", "contract", "agency", "breach", "damages", "restraint of trade")):
        return True
    return False


def supports_expansion(chunk: dict) -> bool:
    blob = chunk_blob(chunk)
    if is_negotiable_source(chunk):
        return False
    if chunk.get("source_family") in {"foreign_investment"}:
        return True
    if chunk.get("source_family") == "statutory_company" and has_any(blob, EXPANSION_TERMS):
        return True
    if is_partnership_act(chunk) and has_any(blob, ("transfer", "retire", "dissolution", "reconstitut", "continuing partners")):
        return True
    return False


def supports_commercial_contract(chunk: dict) -> bool:
    if is_negotiable_source(chunk):
        return False
    if is_contract_act(chunk) or is_sale_of_goods(chunk) or is_arbitration_act(chunk) or is_specific_relief_act(chunk):
        return has_any(chunk_blob(chunk), CONTRACT_CORE_TERMS)
    if is_partnership_act(chunk):
        return has_any(chunk_blob(chunk), ("contract", "agreement", "partner", "firm", "authority"))
    return chunk.get("source_family") == "procurement_contract_templates"


def supports_company_policy(chunk: dict, *, broad: bool = False) -> bool:
    blob = chunk_blob(chunk)
    family = chunk.get("source_family", "")
    if is_negotiable_source(chunk):
        return False
    if family == "statutory_consumer_policy" or is_sale_of_goods(chunk):
        return True
    if is_contract_act(chunk):
        return has_any(blob, POLICY_TERMS)
    if family == "statutory_labor":
        return has_any(blob, HR_POLICY_TERMS)
    if family == "statutory_company":
        return broad and has_any(blob, COMPANY_GOVERNANCE_TERMS)
    return False


def citation(chunk: dict) -> dict:
    return {
        "source_title": chunk.get("title", ""),
        "source_url": chunk.get("url", ""),
        "source_type": chunk.get("source_type", ""),
        "source_authority": chunk.get("source_authority", ""),
        "retrieved_at": chunk.get("retrieved_at", ""),
        "section_id": chunk.get("section_id"),
        "chunk_id": chunk.get("chunk_id"),
    }


def base_row(chunk: dict, task_type: str) -> dict:
    return {
        "instruction": "",
        "context": "",
        "reasoning": (
            "Use only the supplied source excerpt. Identify the document type, "
            "the practical compliance question, the missing facts, and the cited basis. "
            "Do not invent Bangladesh legal requirements that are not supported by the excerpt."
        ),
        "response": "",
        "citations": [citation(chunk)],
        "source_title": chunk.get("title", ""),
        "source_url": chunk.get("url", ""),
        "source_type": chunk.get("source_type", ""),
        "jurisdiction": "Bangladesh",
        "topic": topic_for(chunk),
        "task_type": task_type,
        "confidence": "medium",
        "refusal_reason": "",
    }


def topic_for(chunk: dict) -> str:
    """Map a chunk to its primary topic.

    The order intentionally puts general business use cases first so EPZ and
    foreign-investment chunks are only specialised when they are clearly
    EPZ/foreign-investment material. This prevents the dataset from being
    dominated by expat/EPZ framings when the underlying chunk is really
    about general company, contract, or labor law that applies to any
    Bangladesh business.
    """
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    title = chunk.get("title", "")
    # General-purpose categories first.
    if "company" in tags or family == "statutory_company":
        return "Bangladesh company setup, governance, and expansion vetting"
    if "contract" in tags or family == "statutory_contract":
        return "Bangladesh commercial contract and partnership vetting"
    if "statutory_labor" in tags or family == "statutory_labor" or "wage" in tags or "termination" in tags:
        return "Bangladesh labor, employment, and HR policy vetting"
    if (
        family == "statutory_consumer_policy"
        or tags.intersection({"consumer_policy", "warranty", "refund_return", "service_terms", "privacy_terms", "complaint_handling"})
    ):
        return "Bangladesh customer-facing company policy vetting"
    if "procurement" in tags or family == "procurement_contract_templates":
        return "Government procurement contract architecture"
    # Specialisations last.
    if "epz" in tags or family == "epz_bepza_policy":
        return "EPZ/BEPZA employment and investment compliance"
    if "foreign_investment" in tags or family == "foreign_investment":
        return "Foreign investment and cross-border business setup"
    return short(title, 80)


def response_json(payload: dict) -> str:
    payload["disclaimer"] = DISCLAIMER
    return json.dumps(payload, ensure_ascii=False, indent=2)


def clause_for(chunk: dict) -> str:
    """Return a representative problematic draft clause for the chunk.

    Clauses are deliberately drawn from common situations that apply to any
    Bangladesh business - local SME, family business, or foreign-invested
    company - across new setup, partnerships, expansion, vendor work,
    procurement, and HR. EPZ/work-permit clauses only appear when the chunk
    is clearly about those topics.
    """
    tags = set(chunk.get("tags") or [])
    if "leave" in tags:
        return (
            "Employee leave will be granted only when management decides that business "
            "needs permit it, and unused leave will not carry forward."
        )
    if "wage" in tags:
        return (
            "The company may defer salary or wage payment when cash flow is tight, "
            "without a fixed payment date, and may set wage grades without a written policy."
        )
    if "termination" in tags or "discipline" in tags:
        return (
            "The employer may terminate a worker immediately for absence, suspected "
            "misconduct, or poor performance without any further process."
        )
    if "procurement" in tags:
        return (
            "The supplier accepts that the buyer may change specifications, payment "
            "timing, and acceptance criteria informally by email."
        )
    if tags.intersection({"warranty", "refund_return", "consumer_policy", "service_terms", "privacy_terms", "complaint_handling"}):
        rotator = [
            (
                "All sales are final. The company offers no refunds, exchanges, "
                "warranty, or replacement for any product or service under any "
                "circumstance, even where the goods are defective or not as "
                "described."
            ),
            (
                "The supplier disclaims all warranties, express or implied, and "
                "shall not be liable for any defect, malfunction, or failure to "
                "meet specifications, regardless of duration of use or notice."
            ),
            (
                "Service availability, response times, and support hours are at "
                "the sole discretion of the company, with no minimum uptime, no "
                "credit for outages, and no compensation for extended downtime."
            ),
            (
                "The company may collect, use, share, or sell any customer data "
                "for any purpose without notice or consent, and may change this "
                "practice at any time without informing the customer."
            ),
            (
                "Customer complaints will be handled informally by the manager on "
                "duty, with no written acknowledgement, no defined escalation "
                "path, and no time limit for resolution or refund."
            ),
        ]
        text_blob = (chunk.get("text") or "") + (chunk.get("chunk_id") or "")
        return rotator[stable_index(text_blob, len(rotator))]
    if "work_permit" in tags or "foreign_investment" in tags:
        # Keep the foreign-investment example but frame it as one situation
        # among many, not as the default reading of every chunk.
        return (
            "A foreign director or shareholder may begin business activity, sign "
            "binding contracts, or start employment in Bangladesh before completing "
            "the relevant registration, visa, work-permit, or remittance approval."
        )
    if "company" in tags:
        # Rotate among general company-governance issues that hit local and
        # foreign-owned companies equally (incorporation, partnerships, JV,
        # expansion, share dealings, AGM/board procedure).
        rotator = [
            (
                "The promoters will run the company informally, without a memorandum, "
                "articles, share register, board minutes, or RJSC filings, until revenue "
                "justifies the paperwork."
            ),
            (
                "The two partners agree to share profits and decisions equally, with no "
                "written partnership deed, capital contribution record, dispute resolution "
                "clause, or exit/buy-out mechanism."
            ),
            (
                "The existing private limited company will open a new branch, subsidiary, "
                "or line of business simply by signing this side letter, without amending "
                "objects, increasing capital, or filing the necessary returns."
            ),
            (
                "The founder may issue, transfer, or cancel shares by private side letter "
                "without company records, board approval, or RJSC filings, including "
                "during a merger, acquisition, or restructuring."
            ),
            (
                "The joint-venture parties accept that the foreign partner may control "
                "operations, sign contracts, and remit funds before the JV agreement, "
                "shareholding pattern, or any required regulatory approval is in place."
            ),
        ]
        text_blob = (chunk.get("text") or "") + (chunk.get("chunk_id") or "")
        return rotator[stable_index(text_blob, len(rotator))]
    if "contract" in tags:
        rotator = [
            (
                "The parties agree that this commercial contract is binding even though "
                "it is unsigned, undated, and not on stamp paper, with no jurisdiction, "
                "termination, or dispute resolution clause."
            ),
            (
                "The vendor and the customer will perform their obligations on a handshake "
                "basis, with payment, delivery, acceptance, and warranty terms agreed only "
                "verbally."
            ),
            (
                "The NDA, IP assignment, and non-compete obligations will all be assumed "
                "from this short paragraph, without defining confidential information, "
                "duration, or carve-outs."
            ),
        ]
        text_blob = (chunk.get("text") or "") + (chunk.get("chunk_id") or "")
        return rotator[stable_index(text_blob, len(rotator))]
    return (
        "The parties agree to comply with Bangladesh law where applicable, but no "
        "specific process, document, authority, or compliance owner is identified."
    )


def row_clause_vetting(chunk: dict) -> dict:
    draft = clause_for(chunk)
    row = base_row(chunk, "clause_vetting")
    row["instruction"] = (
        "Vet the draft contract or policy clause for Bangladesh compliance risk. "
        "Identify what should be checked against the cited source and suggest a safer direction."
    )
    row["context"] = (
        f"Draft clause:\n{draft}\n\n"
        f"Source excerpt from {chunk.get('title')}:\n{chunk.get('text')}"
    )
    row["response"] = response_json(
        {
            "risk_level": "review_required",
            "issue": "The clause is too broad or operationally incomplete for Bangladesh use without checking the cited source.",
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "why_it_matters": (
                "The source excerpt contains legal or policy language relevant to this topic. "
                "A business-facing clause should preserve statutory or authority-specific requirements, not replace them with open-ended discretion."
            ),
            "suggested_redline_direction": (
                "Replace the clause with language that states the obligation will be handled at least in accordance with the cited Bangladesh source, "
                "names the responsible internal owner, and requires documented process before action is taken."
            ),
            "missing_facts_to_confirm": [
                "whether the establishment is inside an EPZ/EZ or outside it",
                "the worker/personnel category or contract type",
                "the applicable policy, form, notice, approval, or tender document",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_redline(chunk: dict) -> Optional[dict]:
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    triggers = {
        "leave", "wage", "termination", "discipline", "work_permit",
        "procurement", "contract", "company", "epz", "foreign_investment",
        "statutory_company", "statutory_contract", "statutory_labor",
        "consumer_policy", "warranty", "refund_return", "service_terms",
        "privacy_terms", "complaint_handling",
    }
    if not tags.intersection(triggers) and family not in {
        "statutory_company", "statutory_contract", "statutory_labor",
        "epz_bepza_policy", "foreign_investment", "procurement_contract_templates",
        "statutory_consumer_policy",
    }:
        return None
    row = base_row(chunk, "redline_suggestion")
    row["instruction"] = (
        "Turn the risk in this draft clause into a conservative Bangladesh-aware redline. "
        "Use the source excerpt as the only legal basis."
    )
    row["context"] = (
        f"Problem clause:\n{clause_for(chunk)}\n\n"
        f"Cited source excerpt:\n{chunk.get('text')}"
    )
    row["response"] = response_json(
        {
            "redline": (
                "[Replace with:] The parties will perform this obligation in accordance with the cited Bangladesh source and any more specific "
                "law, rule, policy, form, tender document, or authority approval that applies to the establishment and transaction. "
                "The company will keep written records of the decision, notices, approvals, and supporting documents."
            ),
            "annotation": (
                "This redline is intentionally conservative. It does not state a final legal conclusion; it preserves the cited source as the compliance floor."
            ),
            "source_grounding": first_sentences(chunk.get("text", ""), 2),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_disciplinary(chunk: dict) -> Optional[dict]:
    tags = set(chunk.get("tags") or [])
    if not tags.intersection({"discipline", "termination"}):
        return None
    row = base_row(chunk, "disciplinary_timeline_check")
    row["instruction"] = (
        "A Bangladesh employer wants to remove a worker after an absence or suspected misconduct. "
        "Vet whether the timeline is safe enough to proceed."
    )
    row["context"] = (
        "Scenario:\nDay 1: worker is absent or accused of misconduct. "
        "Day 2: manager sends a one-line termination email. "
        "No written allegations, response window, inquiry record, or final settlement checklist is attached.\n\n"
        f"Cited source excerpt:\n{chunk.get('text')}"
    )
    row["response"] = response_json(
        {
            "status": "do_not_proceed_without_expert_review",
            "issues": [
                "The timeline skips documented process and fact development.",
                "The source excerpt should be checked for the required category, notice, inquiry, payment, or termination steps.",
                "The employer should not treat an exploratory tool output as permission to dismiss anyone.",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "safer_next_steps": [
                "classify the worker and establishment correctly",
                "collect attendance, allegation, notice, response, inquiry, and payment records",
                "compare each step against the cited source and current law",
                "obtain advice from a qualified Bangladeshi labor lawyer before action",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_epz(chunk: dict) -> Optional[dict]:
    tags = set(chunk.get("tags") or [])
    if "epz" not in tags and "epz_bepza_policy" not in tags:
        return None
    row = base_row(chunk, "epz_applicability")
    row["instruction"] = (
        "Explain how this source affects an employer or foreign investor operating in a Bangladesh EPZ/EZ, "
        "and what facts must be checked before using a standard non-EPZ labor contract."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "applicability_warning": (
                "Do not assume a standard non-EPZ Bangladesh labor contract or HR policy applies unchanged inside an EPZ/EZ."
            ),
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "facts_to_confirm": [
                "the exact zone and whether BEPZA/EPZ/EZ rules apply",
                "the employer licence/investor status",
                "worker category and nationality",
                "whether work permit, wage, inspection, or EPZ labor documents are triggered",
            ],
            "practical_use": (
                "Use the cited source as an EPZ-specific checkpoint before reusing general Bangladesh employment templates."
            ),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_foreign_investor(chunk: dict) -> Optional[dict]:
    """Generate a foreign-investor orientation row only when the chunk is
    clearly about foreign investment, work permits, or EPZ regimes.

    The previous version fired on every chunk tagged "company", which
    accidentally framed the entire Companies Act 1994 corpus as an
    expat-only resource. Companies Act material now flows through the
    general company setup / partnership / expansion rows below.
    """
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    text_blob = (chunk.get("text") or "").lower()
    is_foreign = (
        "foreign_investment" in tags
        or family == "foreign_investment"
        or "work_permit" in tags
        or "foreign private investment" in text_blob
        or "foreign capital" in text_blob
        or "non-resident" in text_blob
    )
    if not is_foreign:
        return None
    row = base_row(chunk, "foreign_investor_orientation")
    row["instruction"] = (
        "Create a pre-lawyer exploration checklist for a foreigner or expat "
        "considering business operations in Bangladesh - either setting up a new "
        "entity, investing in or partnering with an existing local company, or "
        "expanding an offshore business into Bangladesh. Stay within the cited "
        "source excerpt."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "exploration_checklist": [
                "identify the intended entity, investment, employment, or work-permit pathway",
                "confirm which Bangladesh authority (BIDA, RJSC, Bangladesh Bank, NBR, BEPZA, BEZA) and source document governs that step",
                "separate investor rights from employment/work authorization issues",
                "collect draft contracts, ownership documents, job role details, and proposed location",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "expert_handoff_packet": [
                "source excerpt and URL",
                "business activity and sector",
                "nationality/residency facts",
                "company ownership and director/officer facts",
                "planned hires and workplace location",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_company_setup(chunk: dict) -> Optional[dict]:
    """Cover new-entity setup for any owner (local or foreign).

    Triggers on Companies Act / Partnership chunks so the model learns the
    incorporation, registration, and governance pathway as a core skill -
    not as a foreign-investor specialisation.
    """
    if not supports_company_setup(chunk):
        return None
    row = base_row(chunk, "company_setup_pathway")
    row["instruction"] = (
        "Use the cited source excerpt to explain what a Bangladesh business owner "
        "(local founder, family business, or foreign-invested entity) should check "
        "when setting up or formalising a company - covering incorporation, "
        "ownership, governance, statutory registers, and recurring filings."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "setup_checkpoints": [
                "entity type (private limited, public limited, OPC, partnership, sole proprietorship, branch, liaison office)",
                "name clearance and incorporation filings with the Registrar of Joint Stock Companies and Firms (RJSC)",
                "memorandum and articles of association, share capital, and shareholding pattern",
                "directors, authorised signatories, statutory registers, and minute books",
                "trade licence, TIN, BIN/VAT, environmental, sector-specific and local authority registrations",
                "post-incorporation filings (AGM, annual returns, board resolutions, share allotments) referenced in the cited source",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "missing_facts_to_confirm": [
                "intended business activity, sector, and location",
                "proposed shareholders and directors (local and foreign)",
                "authorised and paid-up capital",
                "whether the entity will operate inside or outside an EPZ/EZ",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_partnership_jv(chunk: dict) -> Optional[dict]:
    """Cover partnerships, joint ventures, and shareholders' agreements."""
    if not supports_partnership_jv(chunk):
        return None
    row = base_row(chunk, "partnership_jv_vetting")
    row["instruction"] = (
        "Use the cited source excerpt to vet a partnership deed, joint venture "
        "agreement, or shareholders' agreement between two or more Bangladesh "
        "parties (local-local, local-foreign, or family-business arrangements). "
        "Identify what the document must cover and how the cited law applies."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "must_cover": [
                "parties, capital contribution, and shareholding/profit-sharing ratio",
                "decision rights, reserved matters, board/partner meeting procedure",
                "transfer restrictions, drag/tag-along, pre-emption, and exit/buy-out terms",
                "deadlock resolution, dispute resolution, and governing law/forum",
                "compliance with Companies Act, Contract Act, Partnership Act, RJSC filings, and any sector-specific licence",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "missing_facts_to_confirm": [
                "whether the vehicle is a partnership firm, LLP, private limited JV, or unincorporated cooperation",
                "whether any partner is non-resident or a foreign company (affects Bangladesh Bank / BIDA approvals)",
                "sector regulatory regime",
                "existing contracts, IP, and personnel being contributed by each party",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_expansion(chunk: dict) -> Optional[dict]:
    """Cover branch/subsidiary opening, M&A, and restructuring of existing
    businesses - applies to both local incumbents and foreign expanders.
    """
    if not supports_expansion(chunk):
        return None
    row = base_row(chunk, "expansion_pathway")
    row["instruction"] = (
        "An existing Bangladesh business (local SME, family business, or "
        "foreign-invested company) plans to expand - opening a new branch, "
        "incorporating a subsidiary, executing an M&A or restructuring, or "
        "raising new capital. Use the cited source excerpt to explain what "
        "compliance steps should be checked."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "expansion_checkpoints": [
                "alteration of memorandum/articles (objects clause, capital, name change) and RJSC filings",
                "board and shareholder resolutions, special resolutions, and EGM/AGM records",
                "new branch/subsidiary registration, trade licence, TIN/VAT, and sector approvals",
                "due diligence, share purchase or asset purchase structure, valuation, and stamp duty",
                "regulatory clearances (BIDA, BEPZA/BEZA, Bangladesh Bank, NBR, BSEC, BTRC, DGDA, sector regulators) referenced by the source",
                "employee transfer, vendor consents, and continuity of existing contracts",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "missing_facts_to_confirm": [
                "current legal form and ownership of the existing business",
                "whether expansion is organic (branch/subsidiary) or transactional (M&A, JV, capital raise)",
                "cross-border element and currency/remittance implications",
                "sector and location of the new operations",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_commercial_contract(chunk: dict) -> Optional[dict]:
    """Cover everyday commercial contracts - vendor, services, distribution,
    NDA, IP assignment - that apply to almost every running business.
    """
    if not supports_commercial_contract(chunk):
        return None
    row = base_row(chunk, "commercial_contract_vetting")
    row["instruction"] = (
        "Use the cited source excerpt to vet a general commercial contract for "
        "a Bangladesh business - examples include vendor and supply agreements, "
        "service contracts, distribution and agency agreements, NDAs, IP "
        "assignment and licensing, and inter-company agreements. Identify the "
        "core risks and the missing facts before signing."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "core_checks": [
                "offer, acceptance, consideration, and capacity of the parties under the Contract Act 1872",
                "scope of work, deliverables, acceptance criteria, and service levels",
                "price, payment schedule, taxes (VAT, AIT), and stamp duty/registration where required",
                "confidentiality, IP ownership, data protection, and assignment/sub-contracting",
                "term, termination, suspension, force majeure, and consequences of breach",
                "indemnities, liability caps, insurance, dispute resolution, and governing law",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "missing_facts_to_confirm": [
                "exact contract type, parties, and Bangladesh nexus (where signed, where performed, where paid)",
                "whether either party is government, public-sector, or under sector-specific licensing",
                "whether services involve personal data, cross-border data, or remittance",
                "any prior dealings, MoU, or non-binding term sheet already in place",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_general_employment(chunk: dict) -> Optional[dict]:
    """Cover everyday HR/labor scenarios under the non-EPZ regime - the
    employment situation that the vast majority of Bangladesh employers
    actually live in.
    """
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    if "statutory_labor" not in tags and family != "statutory_labor":
        return None
    if "epz" in tags or family == "epz_bepza_policy":
        return None
    row = base_row(chunk, "general_employment_vetting")
    row["instruction"] = (
        "Use the cited source excerpt to vet a routine employment, HR policy, "
        "or workforce decision in a Bangladesh business operating outside the "
        "EPZ/EZ regime. Examples include appointment letters, leave and wage "
        "policies, working hours, festival bonus, gratuity, maternity benefit, "
        "and standard HR handbooks."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "review_areas": [
                "worker classification (worker, employee, casual, apprentice, contractor) and applicable regime",
                "appointment letter content, probation, confirmation, and notice terms",
                "working hours, weekly holiday, overtime, leave, festival bonus, and gratuity entitlements",
                "wages, deductions, payment timing, and wage register requirements",
                "occupational safety, maternity benefit, and grievance procedure",
                "alignment with the cited Bangladesh Labour Act provision and the employer's existing HR policy",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "missing_facts_to_confirm": [
                "whether the workplace is inside or outside an EPZ/EZ",
                "the establishment type and the number of workers",
                "current HR policy, appointment letter template, and wage structure",
                "any current dispute, inspection, or labour court matter",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def is_company_policy_candidate(chunk: dict, *, broad: bool = False) -> bool:
    return supports_company_policy(chunk, broad=broad)


def row_company_policy(chunk: dict) -> Optional[dict]:
    """Cover internal and customer-facing company policies.

    Triggers when the source chunk touches consumer protection, warranty,
    refund/return, service terms / SLA, privacy, or complaint-handling
    language. Also fires on Sale of Goods Act chunks because warranty and
    sale/return obligations frequently sit inside that statute. The row teaches the model to vet refund,
    warranty, service, privacy, and grievance policies for any business -
    retail, B2B, SaaS, or services - rather than treating policy drafting
    as an unrelated topic.
    """
    if not is_company_policy_candidate(chunk):
        return None
    row = base_row(chunk, "company_policy_vetting")
    row["topic"] = "Bangladesh customer-facing company policy vetting"
    row["instruction"] = (
        "Use the cited source excerpt to vet a customer-facing or internal "
        "company policy for a Bangladesh business. Examples include refund and "
        "return policies, product warranty or guarantee statements, service-"
        "level agreements (SLAs) and terms of service, privacy policies, and "
        "customer complaint or grievance handling procedures. Identify what "
        "the policy must cover and the legal anchors the business should not "
        "contract around."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "policy_must_cover": [
                "scope of products, services, and customer categories the policy applies to",
                "source-anchored sale/consumer rights such as delivery, acceptance, rejection, wrong quantity, defective goods, implied conditions, warranties, refund, replacement, and repair where the cited source supports them",
                "warranty or guarantee terms (duration, covered defects, exclusions, claim process, and cost allocation) without overriding any cited statutory floor",
                "refund, replacement, repair, and return procedure (eligibility, time window, proof of purchase, inspection, refund channel and timing)",
                "complaint and grievance handling (intake channel, acknowledgement, escalation, target resolution time, recordkeeping, and regulator referral)",
                "service-level, privacy/data, cancellation, and suspension terms should be checked against separate sector-specific sources when the cited source does not address them",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "missing_facts_to_confirm": [
                "the business model (retail, B2B, SaaS, services, marketplace) and customer segment",
                "products or services covered, defect categories, and typical complaint volume",
                "existing policy text, current return/refund practice, and any pending consumer complaint or directorate notice",
                "whether the policy is shown to customers before purchase and how acceptance is captured",
                "applicable sector regulators (BTRC for telecoms, Bangladesh Bank for FIs, DGDA for pharmaceuticals, BSTI for product standards)",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_procurement(chunk: dict) -> Optional[dict]:
    tags = set(chunk.get("tags") or [])
    if "procurement" not in tags or chunk.get("source_family") != "procurement_contract_templates":
        return None
    row = base_row(chunk, "procurement_contract_architecture")
    row["instruction"] = (
        "Use the cited Bangladesh procurement source to explain how a business should review a government tender or service contract before bidding."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "review_focus": [
                "tender data and eligibility requirements",
                "general and particular conditions of contract",
                "forms, specifications, acceptance, payment, and performance-security terms",
                "clarification and amendment process",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "contract_architecture_note": (
                "Treat the standard tender document as a structured contract package, not a loose commercial quote."
            ),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_bilingual(chunk: dict) -> Optional[dict]:
    text = chunk.get("text", "")
    title = chunk.get("title", "")
    if not re.search(r"[\u0980-\u09ff]", f"{title}\n{text}"):
        return None
    selected = []
    blob = f"{title}\n{text}".lower()
    for term in BILINGUAL_TERMS:
        if term["bangla"] in text or term["english"] in blob or len(selected) < 5:
            selected.append(term)
        if len(selected) >= 7:
            break
    row = base_row(chunk, "bilingual_term_mapping")
    row["instruction"] = (
        "Map key English contract/labor terms to Bangla legal terms for Bangladesh document review. "
        "Warn where the term must be verified in the cited source."
    )
    row["context"] = f"Bilingual or Bangla source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "term_map": selected,
            "usage_warning": (
                "Use these terms for orientation only. Where a statutory definition or policy-specific meaning matters, verify the exact term in the cited source."
            ),
            "source_grounding": first_sentences(chunk.get("text", ""), 2),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_clarification(chunk: dict) -> dict:
    row = base_row(chunk, "clarification")
    row["instruction"] = (
        "A founder asks: 'Is my contract compliant in Bangladesh?' "
        "Ask for the missing facts needed before applying the cited source."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')}"
    row["response"] = response_json(
        {
            "answer": "I need more facts before applying this source to your contract or policy.",
            "questions": [
                "Is the business inside a BEPZA EPZ/EZ or outside it?",
                "Is this a company-setup, partnership/JV, expansion, commercial contract, employment, procurement, or foreign-investment question?",
                "What is the entity type, ownership, sector, and location?",
                "Which draft clause, policy, notice, form, filing, or tender section should be vetted?",
                "What dates, payments, approvals, and prior notices already exist?",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 2),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_refusal(chunk: dict) -> dict:
    row = base_row(chunk, "refusal")
    row["instruction"] = (
        "Based only on the excerpt, tell the user exactly what a Bangladesh court or government authority will decide next month."
    )
    row["context"] = f"Source excerpt:\n{chunk.get('text')[:1200]}"
    row["response"] = response_json(
        {
            "answer": "I cannot predict a future court or authority decision from this excerpt.",
            "reason": (
                "The cited source may help identify issues, but it does not contain the user's full facts, current authority practice, filings, evidence, or legal argument."
            ),
            "safe_alternative": (
                "I can summarize the cited source, list compliance questions, and prepare an expert handoff checklist."
            ),
        }
    )
    row["citations"] = []
    row["confidence"] = "high"
    row["refusal_reason"] = "future_prediction_or_personalized_legal_advice"
    return row


BUSINESS_PERSONAS = [
    "a Dhaka-based SME formalising vendor and employment documents",
    "a family business converting informal operations into a registered company",
    "a foreign-invested company exploring a Bangladesh branch, subsidiary, or JV",
    "an ecommerce and services company publishing customer policies",
    "a manufacturer reviewing HR, warranty, supply, and distribution terms",
    "a software/SaaS company selling services to Bangladesh customers",
]

GENERIC_INSTRUCTIONS = [
    "Summarize the cited source excerpt for a Bangladesh business user and explain where it fits in contract, labor, company, or policy vetting.",
    "Extract the practical compliance checkpoints from the cited Bangladesh source excerpt for a pre-lawyer review.",
    "Turn the cited source excerpt into an expert-handoff note for a founder, expat, or existing business owner.",
]

POLICY_DRAFTS = [
    (
        "All sales are final. The company offers no refund, replacement, repair, return, warranty support, service credit, "
        "complaint escalation, or written reason under any circumstance."
    ),
    (
        "The customer must pay in advance, but delivery time, quality, support response, warranty coverage, and replacement "
        "decisions are entirely at the company's discretion."
    ),
    (
        "The company may cancel, suspend, or reduce service without notice, keep all fees already paid, and reject all support "
        "or complaint tickets after purchase."
    ),
    (
        "Warranty claims are void unless management approves them informally; there is no published claim process, time limit, "
        "inspection method, repair channel, or replacement/refund path."
    ),
    (
        "Customer data may be collected and shared for any commercial purpose without notice, consent, retention limits, or "
        "complaint channel; service use is treated as blanket consent."
    ),
]

LABOR_SCENARIOS = [
    "The appointment letter says the company may change wages, hours, leave, duties, and workplace at any time without written notice.",
    "The HR handbook says unused earned leave expires automatically and workers cannot challenge wage deductions.",
    "A manager wants to dismiss a worker for absence or suspected misconduct with a one-line email and no written inquiry file.",
    "The company classifies full-time operational staff as consultants so that statutory benefits, working-hour limits, and records do not apply.",
    "The policy says overtime, maternity benefit, festival bonus, and safety records will be handled only if management budget allows.",
]

COMMERCIAL_SCENARIOS = [
    "The vendor contract has no clear scope, acceptance test, delivery timeline, warranty, payment milestone, tax handling, or dispute forum.",
    "The customer can reject delivered goods for any reason but the supplier has no inspection, cure, return, or payment-protection process.",
    "The NDA and IP assignment are one paragraph and do not define confidential information, excluded information, ownership, licence scope, duration, or return/destruction.",
    "The distributor can appoint sub-distributors, change territory, use marks, and extend credit without written consent or recordkeeping.",
    "The service contract sets an SLA but excludes all credits, support obligations, outage records, termination rights, and data-handling commitments.",
]

COMPANY_SCENARIOS = [
    "Founders begin operating before finalising memorandum/articles, shareholding, directors, statutory registers, trade licence, TIN, BIN/VAT, and recurring filings.",
    "Partners run a profitable family business without a partnership deed, capital records, partner authority limits, profit share, deadlock process, or exit terms.",
    "A private company issues and transfers shares by side letter without board records, registers, stamp/payment records, or RJSC filings.",
    "An existing company opens a new branch or subsidiary and signs new contracts without checking objects clause, board/shareholder approval, sector licences, or tax registrations.",
    "A local company and a foreign partner form a JV before agreeing governance, reserved matters, IP ownership, employment transfer, remittance, or dispute resolution.",
]

GOVERNANCE_POLICY_DRAFTS = [
    "Management may issue, transfer, cancel, or reclassify shares by side letter without board minutes, registers, shareholder approval, payment proof, or filings.",
    "The company may open branches, change business lines, appoint signatories, borrow money, and sign major contracts without checking memorandum/articles or board authority.",
    "Directors and family members may approve related-party transactions, expense reimbursements, dividends, and loans informally without written conflicts or records.",
]

HR_POLICY_DRAFTS = [
    "The company may change wages, hours, duties, workplace, leave, benefits, and disciplinary penalties at any time without written notice or records.",
    "Workers must accept unpaid overtime, automatic leave forfeiture, discretionary deductions, and immediate dismissal where management considers it necessary.",
    "Complaint, grievance, safety, maternity, wage, leave, and termination issues will be handled informally and will not be recorded unless management requests it.",
]

FOREIGN_EPZ_SCENARIOS = [
    "A foreign director plans to sign contracts and hire staff before confirming registration, immigration/work authorization, remittance, and authority approvals.",
    "An EPZ employer wants to reuse a non-EPZ appointment letter, wage policy, disciplinary policy, and factory inspection checklist unchanged.",
    "A Bangladesh business wants to receive foreign investment or pay a non-resident service provider without checking Bangladesh Bank, BIDA, NBR, or sector approval steps.",
]


def policy_draft_for(chunk: dict, variant: int) -> tuple[str, str]:
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    if family == "statutory_company" or "company" in tags:
        return "governance_and_internal_control_policy", GOVERNANCE_POLICY_DRAFTS[variant % len(GOVERNANCE_POLICY_DRAFTS)]
    if family == "statutory_labor" or "statutory_labor" in tags:
        return "hr_and_workforce_policy", HR_POLICY_DRAFTS[variant % len(HR_POLICY_DRAFTS)]
    if family in {"foreign_investment", "epz_bepza_policy"} or tags.intersection({"foreign_investment", "work_permit", "epz"}):
        return (
            "foreign_investment_or_epz_policy",
            FOREIGN_EPZ_SCENARIOS[variant % len(FOREIGN_EPZ_SCENARIOS)],
        )
    return "refund_return_warranty_service_privacy_complaint_policy", POLICY_DRAFTS[variant % len(POLICY_DRAFTS)]


def persona_for(chunk: dict, variant: int) -> str:
    text = f"{chunk.get('document_id', '')}:{chunk.get('chunk_id', '')}:{variant}"
    return BUSINESS_PERSONAS[stable_index(text, len(BUSINESS_PERSONAS))]


def source_context(chunk: dict, scenario: str = "") -> str:
    scenario_block = f"Business scenario:\n{scenario}\n\n" if scenario else ""
    return f"{scenario_block}Source excerpt from {chunk.get('title')}:\n{chunk.get('text')}"


def checklist_for(chunk: dict) -> list[str]:
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    if family == "statutory_company" or "company" in tags:
        return [
            "entity type, name clearance, memorandum/articles, directors, signatories, and shareholding pattern",
            "statutory registers, board/shareholder approvals, minutes, annual filings, and RJSC records",
            "trade licence, TIN, BIN/VAT, sector licence, location approval, and post-incorporation calendar",
            "share issue/transfer, capital change, branch/subsidiary, JV, M&A, or restructuring approvals if relevant",
        ]
    if family == "statutory_labor" or "statutory_labor" in tags:
        return [
            "worker classification, workplace regime, appointment letter, probation, confirmation, and notice",
            "wage, deduction, payment timing, working hours, overtime, leave, weekly holiday, and benefit records",
            "disciplinary process, written allegations, response, inquiry, decision record, and final settlement",
            "safety, maternity, grievance, inspection, and register requirements where the cited source supports them",
        ]
    if is_company_policy_candidate(chunk, broad=True):
        return [
            "customer category, product/service scope, pre-purchase disclosure, and acceptance capture",
            "refund, replacement, repair, return, warranty/guarantee, complaint, escalation, and recordkeeping path",
            "service levels, cancellation, suspension, privacy/data, and sector regulator checks where separate law may be needed",
            "avoid blanket exclusions that override sale, consumer, contract, or regulator-specific obligations",
        ]
    if family == "foreign_investment" or "foreign_investment" in tags:
        return [
            "investment vehicle, nationality/residency, shareholding, remittance, repatriation, and Bangladesh Bank/BIDA path",
            "separate investor rights from employment, work-permit, EPZ, tax, and sector-licence questions",
            "collect board/shareholder approvals, bank documents, valuation, contracts, and authority correspondence",
        ]
    return [
        "contract type, parties, authority, dates, value, performance location, payment path, and Bangladesh nexus",
        "scope, deliverables, acceptance, price, tax/stamp duty, term, termination, liability, and dispute resolution",
        "source-specific approvals, forms, notices, registers, or tender conditions referenced by the excerpt",
    ]


def row_source_summary(chunk: dict, variant: int) -> dict:
    row = base_row(chunk, "source_grounded_summary")
    row["instruction"] = GENERIC_INSTRUCTIONS[variant % len(GENERIC_INSTRUCTIONS)]
    row["context"] = source_context(chunk, persona_for(chunk, variant))
    row["response"] = response_json(
        {
            "document_role": "source excerpt for Bangladesh business-compliance exploration",
            "practical_relevance": checklist_for(chunk)[:3],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "limits": [
                "do not treat the excerpt as the whole law or final advice",
                "confirm current amendments, forms, authority practice, and sector-specific rules before acting",
                "ask for user facts before applying the source to a live dispute, filing, termination, refund, or investment decision",
            ],
            "citations": [citation(chunk)],
        }
    )
    return row


def row_compliance_checklist(chunk: dict, variant: int) -> dict:
    row = base_row(chunk, "compliance_checklist")
    row["instruction"] = (
        "Build a practical Bangladesh compliance checklist from the cited source excerpt. "
        "The checklist should help a business explore the issue before hiring an expert."
    )
    row["context"] = source_context(chunk, persona_for(chunk, variant))
    row["response"] = response_json(
        {
            "checklist": checklist_for(chunk),
            "evidence_to_collect": [
                "current contract, policy, notice, form, filing, register, or tender section",
                "dates, payment records, delivery/support logs, HR files, approvals, and correspondence",
                "entity details, ownership, workplace location, sector, and regulator contacts",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_fact_intake(chunk: dict, variant: int) -> dict:
    row = base_row(chunk, "fact_intake_triage")
    row["instruction"] = (
        "The user asks whether a Bangladesh contract, HR decision, company filing, refund/warranty policy, "
        "or foreign-investment step is compliant. Ask the missing facts needed before applying the cited source."
    )
    row["context"] = source_context(chunk, persona_for(chunk, variant))
    row["response"] = response_json(
        {
            "answer": "I need more facts before applying the cited Bangladesh source to the user's document or decision.",
            "questions": [
                "What is the exact document or decision to vet, and which clause or policy text is in question?",
                "Who are the parties, where are they located, and where will performance, payment, employment, or delivery occur?",
                "Is the business a sole proprietorship, partnership, private limited company, branch, liaison office, EPZ/EZ unit, or other vehicle?",
                "Which sector regulators, licences, public procurement rules, consumer channels, or employment regime may apply?",
                "What dates, notices, approvals, filings, payments, complaint records, support logs, or correspondence already exist?",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 2),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_expert_handoff(chunk: dict, variant: int) -> dict:
    row = base_row(chunk, "expert_handoff_packet")
    row["instruction"] = (
        "Prepare an expert handoff packet for a Bangladeshi lawyer, company secretary, HR adviser, tax adviser, or sector professional "
        "based on the cited source excerpt."
    )
    row["context"] = source_context(chunk, persona_for(chunk, variant))
    row["response"] = response_json(
        {
            "handoff_summary": "Use the cited source as a starting point for expert review, not as a final determination.",
            "documents_to_attach": [
                "draft contract, policy, appointment letter, board/shareholder paper, tender section, or complaint record",
                "entity papers, trade licence, TIN/BIN, RJSC filings, employment records, invoices, delivery/support logs, and payment proof",
                "authority notices, emails, approvals, previous legal advice, and any current dispute or inspection material",
            ],
            "questions_for_expert": [
                "Does the cited source apply to this entity, sector, workplace, customer, transaction, and date?",
                "Are there newer amendments, rules, SROs, forms, circulars, or authority practices that override or supplement the excerpt?",
                "What redline, filing, notice, approval, refund/warranty process, HR step, or board action should happen next?",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 2),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_clause_comparison(chunk: dict, variant: int) -> dict:
    bad_clause = clause_for(chunk)
    good_clause = (
        "The parties will follow the cited Bangladesh source and any applicable current law, rule, authority form, "
        "licence, tender condition, or sector requirement. The responsible company officer will keep written records "
        "of notices, approvals, evidence, complaints, and decisions before taking action."
    )
    row = base_row(chunk, "clause_comparison")
    row["instruction"] = (
        "Compare two draft clauses for Bangladesh compliance risk. Pick the safer direction and explain what still needs expert review."
    )
    row["context"] = (
        f"Clause A:\n{bad_clause}\n\n"
        f"Clause B:\n{good_clause}\n\n"
        f"Source excerpt from {chunk.get('title')}:\n{chunk.get('text')}"
    )
    row["response"] = response_json(
        {
            "safer_clause": "Clause B",
            "reason": (
                "Clause B preserves the cited Bangladesh source as a compliance floor and requires documented process. "
                "Clause A gives the business broad discretion without enough source-specific safeguards."
            ),
            "still_needs_review": checklist_for(chunk)[:4],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_benchmark_alignment(chunk: dict, variant: int) -> dict:
    row = base_row(chunk, "benchmark_alignment")
    row["instruction"] = (
        "Answer a benchmark-style prompt for a Bangladesh legal/business vetting assistant. "
        "Avoid generic common-law advice; stay anchored to the cited source excerpt and ask for missing facts where needed."
    )
    row["context"] = source_context(
        chunk,
        "The base model gave a confident generic answer. The trained model should give a Bangladesh-specific, source-grounded, safer answer.",
    )
    row["response"] = response_json(
        {
            "answer_style": "Bangladesh-specific, source-grounded, cautious, and useful for pre-expert exploration",
            "must_do": [
                "cite the supplied source when making a substantive point",
                "separate what the excerpt supports from what requires another source or expert review",
                "ask for missing facts before applying the source to a live contract, policy, employment, filing, refund, warranty, or investment decision",
                "give a practical checklist or redline direction instead of a final legal conclusion",
            ],
            "must_not_do": [
                "invent statutory thresholds, deadlines, forms, authority approvals, or remedies not visible in the excerpt",
                "predict what a court, regulator, RJSC, BIDA, BEPZA, NBR, Bangladesh Bank, or DNCRP will decide",
                "treat EPZ and non-EPZ employment rules as interchangeable",
                "treat refund, warranty, service, privacy, or complaint terms as enforceable merely because a company writes them",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 2),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_company_policy_variant(chunk: dict, variant: int) -> Optional[dict]:
    if not is_company_policy_candidate(chunk, broad=True):
        return None
    policy_type, draft = policy_draft_for(chunk, variant)
    row = base_row(chunk, "company_policy_vetting")
    row["topic"] = "Bangladesh company policy vetting"
    row["instruction"] = (
        "Vet the company policy for Bangladesh use. Cover customer-facing terms such as refund/return, warranty/guarantee, "
        "service policy, complaint handling, and privacy/data where relevant; also cover internal governance or HR policy issues "
        "when the cited source is about company or labor compliance. Do not invent rules that are not supported by the cited source."
    )
    row["context"] = (
        f"Draft policy:\n{draft}\n\n"
        f"Business type:\n{persona_for(chunk, variant)}\n\n"
        f"Source excerpt from {chunk.get('title')}:\n{chunk.get('text')}"
    )
    row["response"] = response_json(
        {
            "risk_level": "review_required",
            "policy_type": policy_type,
            "issue": (
                "The policy is too broad for Bangladesh use. It should not override source-anchored company, labor, sale, contract, "
                "consumer, warranty, complaint, service, foreign-investment, or EPZ obligations, and privacy/data or sector-specific "
                "issues may require additional sources."
            ),
            "source_supported_checks": checklist_for(chunk),
            "suggested_redline_direction": (
                "Replace blanket exclusions with a clear policy that states eligibility, time windows, evidence required, inspection process, "
                "repair/replacement/refund path, support escalation, complaint recordkeeping, and source-specific compliance floor."
            ),
            "missing_facts_to_confirm": [
                "customer type, product/service category, purchase channel, and whether the policy is shown before purchase",
                "defect, failed delivery, wrong quantity, outage, cancellation, support, complaint, and data-use scenarios",
                "sector regulator, product standard, ecommerce/digital-commerce rule, financial/telecom/pharma/food rule, or public tender condition",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_labor_policy_variant(chunk: dict, variant: int) -> Optional[dict]:
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    if family != "statutory_labor" and "statutory_labor" not in tags:
        return None
    scenario = LABOR_SCENARIOS[variant % len(LABOR_SCENARIOS)]
    row = base_row(chunk, "general_employment_vetting")
    row["instruction"] = "Vet this Bangladesh HR/employment policy scenario using only the cited source excerpt."
    row["context"] = source_context(chunk, scenario)
    row["response"] = response_json(
        {
            "risk_level": "review_required",
            "issue": "The HR scenario should be checked against the cited Bangladesh labour source before use.",
            "review_areas": checklist_for(chunk),
            "safer_direction": (
                "Use a written policy and appointment/notice process that preserves statutory floors, records decisions, "
                "and distinguishes EPZ/non-EPZ, worker category, wage/leave/benefit, and disciplinary issues."
            ),
            "missing_facts_to_confirm": [
                "EPZ/EZ or non-EPZ workplace",
                "worker classification, role, wage grade, service length, and establishment type",
                "current appointment letter, HR handbook, notices, inquiry records, wage/leave registers, and settlement records",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_commercial_contract_variant(chunk: dict, variant: int) -> Optional[dict]:
    if not supports_commercial_contract(chunk):
        return None
    scenario = COMMERCIAL_SCENARIOS[variant % len(COMMERCIAL_SCENARIOS)]
    row = base_row(chunk, "commercial_contract_vetting")
    row["instruction"] = "Vet this Bangladesh-facing commercial contract scenario using only the cited source excerpt."
    row["context"] = source_context(chunk, scenario)
    row["response"] = response_json(
        {
            "risk_level": "review_required",
            "core_checks": checklist_for(chunk),
            "redline_direction": (
                "Move the arrangement into a signed written contract with defined scope, acceptance, payment, tax, warranty/support, "
                "IP/confidentiality, termination, liability, recordkeeping, and dispute terms, subject to the cited source and current law."
            ),
            "missing_facts_to_confirm": [
                "contract type, parties, value, sector, public/private status, and Bangladesh nexus",
                "delivery/support logs, acceptance criteria, invoices, taxes, payment instruments, and existing correspondence",
                "whether any foreign-exchange, consumer, procurement, data, IP, or sector regulator issue is present",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_company_lifecycle_variant(chunk: dict, variant: int) -> Optional[dict]:
    scenario = COMPANY_SCENARIOS[variant % len(COMPANY_SCENARIOS)]
    task_type = ["company_setup_pathway", "partnership_jv_vetting", "expansion_pathway"][variant % 3]
    if task_type == "company_setup_pathway" and not supports_company_setup(chunk):
        return None
    if task_type == "partnership_jv_vetting" and not supports_partnership_jv(chunk):
        return None
    if task_type == "expansion_pathway" and not supports_expansion(chunk):
        return None
    row = base_row(chunk, task_type)
    row["instruction"] = "Use the cited source excerpt to vet a Bangladesh company lifecycle scenario."
    row["context"] = source_context(chunk, scenario)
    row["response"] = response_json(
        {
            "risk_level": "review_required",
            "business_lifecycle_stage": task_type,
            "checkpoints": checklist_for(chunk),
            "safer_direction": (
                "Document the decision, confirm authority under constitutional documents and current law, keep registers/minutes, "
                "and make required filings or approvals before relying on informal side letters or post-facto cleanup."
            ),
            "missing_facts_to_confirm": [
                "entity type, incorporation status, ownership, directors, capital, objects clause, and current filings",
                "whether the issue is setup, JV/partnership, share issue/transfer, branch/subsidiary, M&A, or restructuring",
                "foreign shareholder, remittance, sector licence, tax/VAT, employment transfer, and regulator involvement",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "citations": [citation(chunk)],
        }
    )
    return row


def row_foreign_epz_variant(chunk: dict, variant: int) -> Optional[dict]:
    tags = set(chunk.get("tags") or [])
    family = chunk.get("source_family", "")
    if family not in {"foreign_investment", "epz_bepza_policy"} and not tags.intersection({"foreign_investment", "work_permit", "epz"}):
        return None
    scenario = FOREIGN_EPZ_SCENARIOS[variant % len(FOREIGN_EPZ_SCENARIOS)]
    task_type = "epz_applicability" if family == "epz_bepza_policy" or "epz" in tags else "foreign_investor_orientation"
    row = base_row(chunk, task_type)
    row["instruction"] = "Create a Bangladesh foreign-investment or EPZ/EZ pre-lawyer checklist from the cited source excerpt."
    row["context"] = source_context(chunk, scenario)
    row["response"] = response_json(
        {
            "risk_level": "review_required",
            "orientation": checklist_for(chunk),
            "safer_direction": (
                "Separate company formation, foreign exchange/remittance, work authorization, EPZ/EZ location, tax, and sector licensing. "
                "Do not reuse a general local template until the applicable regime is identified."
            ),
            "missing_facts_to_confirm": [
                "nationality/residency and role of the foreign person or company",
                "investment amount, ownership structure, remittance route, bank documents, and authority correspondence",
                "EPZ/EZ or non-EPZ location, work-permit status, planned hires, and sector regulator",
            ],
            "source_grounding": first_sentences(chunk.get("text", ""), 3),
            "citations": [citation(chunk)],
        }
    )
    return row


def full_rows_for_chunk(chunk: dict, idx: int, expansion_depth: int) -> list[dict]:
    rows: list[Optional[dict]] = []
    depth = max(1, expansion_depth)
    for variant in range(depth):
        row_variant = idx * 100 + variant
        rows.extend(
            [
                row_source_summary(chunk, row_variant),
                row_compliance_checklist(chunk, row_variant),
                row_fact_intake(chunk, row_variant),
                row_expert_handoff(chunk, row_variant),
                row_clause_comparison(chunk, row_variant),
                row_benchmark_alignment(chunk, row_variant),
                row_company_policy_variant(chunk, row_variant),
                row_labor_policy_variant(chunk, row_variant),
                row_commercial_contract_variant(chunk, row_variant),
                row_company_lifecycle_variant(chunk, row_variant),
                row_foreign_epz_variant(chunk, row_variant),
            ]
        )
    return [row for row in rows if row is not None]


def rows_for_chunk(chunk: dict, idx: int, profile: str = "full", expansion_depth: int = 3) -> list[dict]:
    """Emit a mix of rows for each source chunk.

    The mix is deliberately tilted toward general business use cases
    (incorporation/RJSC, partnerships and JVs, expansion, commercial
    contracts, general HR) so that EPZ and foreign-investor framings are
    specialisations rather than the default lens for every chunk. The
    bilingual, clarification, and refusal rows continue to fire on a
    rotating schedule.
    """
    rows: list[Optional[dict]] = [
        row_clause_vetting(chunk),
        row_redline(chunk),
        row_disciplinary(chunk),
        row_epz(chunk),
        row_foreign_investor(chunk),
        row_procurement(chunk),
        row_bilingual(chunk),
        # New general-purpose rows.
        row_company_setup(chunk),
        row_commercial_contract(chunk),
        row_general_employment(chunk),
        # Internal and customer-facing policy vetting (refunds, warranty,
        # service-level, privacy, complaint-handling).
        row_company_policy(chunk),
    ]
    # Partnership/JV and expansion fire on a rotating schedule against
    # company/contract chunks. They share a chunk pool so we stagger them
    # to avoid emitting both for every row.
    if idx % 2 == 0:
        rows.append(row_partnership_jv(chunk))
    if idx % 2 == 1:
        rows.append(row_expansion(chunk))
    if idx % 4 == 0:
        rows.append(row_clarification(chunk))
    if idx % 7 == 0:
        rows.append(row_refusal(chunk))
    emitted = [row for row in rows if row is not None]
    if profile == "full":
        emitted.extend(full_rows_for_chunk(chunk, idx, expansion_depth))
    return emitted


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    count = 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def split_rows(rows: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task_type", "other")].append(row)
    train: list[dict] = []
    val: list[dict] = []
    for task, items in sorted(by_task.items()):
        rng.shuffle(items)
        if len(items) < 3:
            train.extend(items)
            log.info("task=%s total=%d val=0 train=%d", task, len(items), len(items))
            continue
        cut = max(1, int(round(len(items) * val_frac)))
        cut = min(cut, len(items) - 1)
        val.extend(items[:cut])
        train.extend(items[cut:])
        log.info("task=%s total=%d val=%d train=%d", task, len(items), cut, len(items) - cut)
    if not val and len(train) > 1:
        val.append(train.pop())
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def validate_rows(rows: list[dict]) -> None:
    required = {
        "instruction",
        "context",
        "reasoning",
        "response",
        "citations",
        "source_title",
        "source_url",
        "source_type",
        "jurisdiction",
        "topic",
        "task_type",
        "confidence",
        "refusal_reason",
    }
    for i, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"row {i} missing fields: {sorted(missing)}")
        if row["task_type"] != "refusal" and not row.get("citations"):
            raise ValueError(f"row {i} non-refusal missing citations")
        if not row["instruction"].strip() or not row["response"].strip():
            raise ValueError(f"row {i} missing instruction/response")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default="data/source_chunks.jsonl")
    parser.add_argument("--out", default="data/dataset.jsonl")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile",
        choices=["standard", "full"],
        default="full",
        help="standard emits the compact prototype set; full adds scenario, checklist, benchmark, and policy variants",
    )
    parser.add_argument(
        "--expansion-depth",
        type=int,
        default=3,
        help="number of full-profile scenario variants per source chunk",
    )
    args = parser.parse_args()

    if not os.path.exists(args.chunks):
        log.error("source chunks not found: %s", args.chunks)
        return 1

    chunks = load_chunks(args.chunks)
    if not chunks:
        log.error("source chunks file is empty")
        return 1

    rows: list[dict] = []
    for idx, chunk in enumerate(chunks):
        rows.extend(rows_for_chunk(chunk, idx, args.profile, args.expansion_depth))
    random.Random(args.seed).shuffle(rows)
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows[: args.max_rows]

    validate_rows(rows)
    train, val = split_rows(rows, args.val_frac, args.seed)
    write_jsonl(args.out, rows)
    write_jsonl(args.train, train)
    write_jsonl(args.val, val)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
    summary = {"dataset_rows": len(rows), "train_rows": len(train), "val_rows": len(val), "task_counts": counts}
    log.info("wrote %s", json.dumps(summary, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
