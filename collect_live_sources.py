"""
collect_live_sources.py

Bounded live-source collector for a general-purpose Bangladesh contract,
labor, and policy vetting SFT dataset. It fetches selected official and
supplementary sources, extracts text, stores provenance, and writes
source chunks for downstream instruction generation.

The collector is designed so the training corpus serves any business
operating in Bangladesh - local SMEs, family firms, partnerships, joint
ventures, and foreign-invested entities - across new setup, partnerships,
expansion, commercial contracts, and HR. EPZ/BEPZA, government procurement
(BPPA/CPTU), and the Foreign Private Investment Act remain in the mix as
specialised regimes rather than the dominant share.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import logging
import os
import re
import ssl
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Iterable, Optional
from urllib.parse import quote, urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    import requests
except Exception:  # pragma: no cover - optional dependency fallback
    requests = None

try:
    from bs4 import BeautifulSoup, Tag
except Exception:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None
    Tag = Any

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional at import time
    pdfplumber = None


BEPZA_ACTS_URL = "https://bepza.gov.bd/acts-policies"
CPTU_STD_URL = "https://bangla.cptu.gov.bd/standard-documents/standard-tender-document.html"
CPTU_STD_FALLBACK_URLS = [
    "https://cptu.gov.bd/standard-documents/standard-tender-document.html",
    "https://www.cptu.gov.bd/standard-documents/standard-tender-document.html",
]
BDLAWS_ACT_PRINT = "http://bdlaws.minlaw.gov.bd/act-print-{act_id}.html"

DEFAULT_BDLAWS_ACTS = [
    # Core statutes that apply to every Bangladesh business - local SME,
    # family firm, partnership, or foreign-invested entity. Verified act
    # IDs on bdlaws.minlaw.gov.bd.
    {
        "act_id": "26",
        "title": "Contract Act, 1872",
        "family": "statutory_contract",
        "reason": "core source for contract formation, obligations, and remedies for any commercial contract",
    },
    {
        "act_id": "150",
        "title": "Sale of Goods Act, 1930",
        "family": "statutory_contract",
        "reason": "core source for sale contracts, implied conditions, warranties, delivery, acceptance, and breach of warranty",
    },
    {
        "act_id": "788",
        "title": "Companies Act, 1994",
        "family": "statutory_company",
        "reason": "incorporation, governance, share capital, board, and filings for any Bangladesh company",
    },
    {
        "act_id": "952",
        "title": "Bangladesh Labour Act, 2006",
        "family": "statutory_labor",
        "reason": "core labor and HR compliance source for non-EPZ employers (and a reference for EPZ employers)",
    },
    {
        "act_id": "597",
        "title": "Foreign Private Investment (Promotion and Protection) Act, 1980",
        "family": "foreign_investment",
        "reason": "foreign investor protection and treatment, used as a specialised source",
    },
    {
        "act_id": "1014",
        "title": "Consumer Rights Protection Act, 2009",
        "family": "statutory_consumer_policy",
        "reason": "consumer rights, complaint handling, unfair practice, refund/replacement/warranty policy, and customer-facing service policy vetting",
    },
]

# Additional Bangladesh statutes that are commonly relevant to business
# setup, partnerships, expansion, and commercial contracts. Act IDs on
# bdlaws.minlaw.gov.bd shift over time; these are best-effort candidates
# and the collector logs and continues on individual failures, so an
# unresolved ID does not block the run. Operators are encouraged to
# verify and extend this list before each fresh collection.
CANDIDATE_BDLAWS_ACTS = [
    {
        "act_id": "157",
        "title": "Partnership Act, 1932",
        "family": "statutory_contract",
        "reason": "partnership formation, partner liability, dissolution - relevant to partnerships, JVs, and family businesses",
    },
    {
        "act_id": "46",
        "title": "Negotiable Instruments Act, 1881 (candidate ID - verify before use)",
        "family": "statutory_contract",
        "reason": "cheques, promissory notes, and bills of exchange - common in B2B contracts and dishonour disputes",
    },
    {
        "act_id": "850",
        "title": "Arbitration Act, 2001",
        "family": "statutory_contract",
        "reason": "dispute resolution clauses and enforcement of arbitral awards",
    },
    {
        "act_id": "36",
        "title": "Specific Relief Act, 1877",
        "family": "statutory_contract",
        "reason": "specific performance, injunctions, and remedies for contract breach",
    },
    {
        "act_id": "218",
        "title": "Foreign Exchange Regulation Act, 1947",
        "family": "foreign_investment",
        "reason": "Bangladesh Bank / foreign exchange permission issues in cross-border contracts, remittance, and foreign-invested business operations",
    },
]


def selected_bdlaws_acts(include_candidates: bool) -> list[dict]:
    """Return the curated list of Bangladesh statutes to collect.

    By default only verified core acts are used. Operators can opt in to
    the broader candidate list via --include-candidate-acts; unresolved
    IDs are handled by the existing failure-logging path.
    """
    if include_candidates:
        return DEFAULT_BDLAWS_ACTS + CANDIDATE_BDLAWS_ACTS
    return DEFAULT_BDLAWS_ACTS

BEPZA_KEYWORDS = (
    "epz labour",
    "labour rules",
    "work permit",
    "minimum wage",
    "wage structure",
    "inspection checklist",
    "foreign private investment",
    "oss rules",
    "one stop service",
)

CPTU_KEYWORDS = (
    "procurement of goods",
    "procurement of works",
    "intellectual and professional",
    "physical service",
    "consultancy",
    "standard e-tender",
    "standard tender",
    "request for proposal",
    "request for quotation",
    "general conditions of contract",
    "particular conditions of contract",
)

HEADERS = {
    "User-Agent": (
        "BDContractLaborPolicyVettingCollector/0.1 "
        "(research dataset; polite bounded fetch)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s collect_live_sources %(message)s",
)
log = logging.getLogger("collect_live_sources")


@dataclass
class SourceRecord:
    document_id: str
    title: str
    url: str
    source_family: str
    source_type: str
    source_authority: str
    content_type: str
    retrieved_at: str
    retrieval_date: str
    sha256: str
    raw_path: str
    text_path: str
    http_status: int
    selected_reason: str
    source_language: str = "mixed"
    page_count_extracted: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class SourceChunk:
    chunk_id: str
    document_id: str
    title: str
    url: str
    source_family: str
    source_type: str
    source_authority: str
    retrieved_at: str
    retrieval_date: str
    section_id: Optional[str]
    heading: str
    text: str
    tokenish_length: int
    tags: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(text: str, fallback: str = "source") -> str:
    text = re.sub(r"[^\w\s.-]+", "", text or "", flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text.strip().lower())
    text = text.strip(".-")
    return text[:90] or fallback


def clean_title(text: str) -> str:
    text = normalize_text(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\d+\s+", "", text)
    return text.strip(" -") or "Untitled source"


def normalize_text(text: str) -> str:
    if "\u00e0\u00a6" in text or "\u00e0\u00a7" in text:
        try:
            text = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def visible_text_from_html(html: str) -> tuple[str, str]:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "lxml")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        title_node = soup.find(id="printheader") or soup.find(["h1", "h2", "h3"]) or soup.title
        title = normalize_text(title_node.get_text(" ", strip=True)) if title_node else "Untitled source"
        blocks = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "div"]):
            text = normalize_text(tag.get_text(" ", strip=True))
            if text and len(text) > 2:
                blocks.append(text)
        deduped = []
        seen = set()
        for block in blocks:
            key = block[:200]
            if key not in seen:
                seen.add(key)
                deduped.append(block)
        return title, normalize_text("\n".join(deduped))

    title_match = re.search(r"<(?:title|h1|h2|h3)[^>]*>(.*?)</(?:title|h1|h2|h3)>", html, re.I | re.S)
    title = strip_tags(title_match.group(1)) if title_match else "Untitled source"
    body = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
    body = re.sub(r"</(?:p|div|tr|li|h[1-4]|table)>", "\n", body, flags=re.I)
    return normalize_text(title), normalize_text(strip_tags(body))


def detect_language(text: str) -> str:
    bengali = len(re.findall(r"[\u0980-\u09ff]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    if bengali and latin:
        return "mixed"
    if bengali:
        return "bn"
    return "en"


def tags_for(title: str, text: str, family: str) -> list[str]:
    blob = f"{title}\n{text}".lower()
    tags: set[str] = {family}
    checks = {
        "leave": ("leave", "annual leave", "earned leave", "ছুটি"),
        "wage": ("wage", "salary", "minimum wage", "মজুরি", "বেতন"),
        "termination": (
            "termination of employment",
            "dismiss",
            "dismissal",
            "discharge from service",
            "retrench",
            "চাকুরীর অবসান",
            "বরখাস্ত",
            "ছাঁটাই",
            "ডিসচার্জ",
        ),
        "discipline": ("misconduct", "disciplinary", "absence", "theft", "অসদাচরণ", "শাস্তি"),
        "safety": ("safety", "health", "inspection", "factory inspection", "নিরাপত্তা"),
        "work_permit": ("work permit", "foreign employee", "expatriate"),
        "epz": ("epz", "export processing zone", "bepza"),
        "procurement": ("procurement", "tender", "contract forms", "gcc", "pcc", "cptu", "bppa"),
        "company": ("company", "shares", "share capital", "director", "registered", "কোম্পানী"),
        "company_setup": (
            "memorandum",
            "articles of association",
            "incorporation",
            "registrar",
            "rjsc",
            "name clearance",
            "trade licence",
            "trade license",
        ),
        "partnership": (
            "partnership",
            "partner",
            "firm",
            "deed of partnership",
            "joint venture",
            "shareholders' agreement",
            "shareholder agreement",
        ),
        "expansion": (
            "branch",
            "subsidiary",
            "merger",
            "amalgamation",
            "acquisition",
            "scheme of arrangement",
            "alter the memorandum",
            "increase of share capital",
        ),
        "ip_confidentiality": (
            "confidential",
            "intellectual property",
            "trademark",
            "copyright",
            "patent",
        ),
        "dispute": ("arbitration", "dispute resolution", "specific performance", "injunction"),
        "negotiable": ("cheque", "promissory note", "bill of exchange", "negotiable instrument"),
        "tax_filings": ("income tax", "vat", "value added tax", "tin", "bin", "withholding"),
        "consumer_policy": (
            "consumer right",
            "consumer protection",
            "unfair trade",
            "deceptive practice",
            "misleading advertisement",
            "ভোক্তা",
            "ভোক্তা অধিকার",
        ),
        "warranty": (
            "warranty",
            "guarantee",
            "merchantable",
            "fitness for purpose",
            "implied condition",
            "express condition",
            "defect",
        ),
        "refund_return": (
            "refund",
            "return policy",
            "return goods",
            "replacement",
            "money back",
            "credit note",
        ),
        "service_terms": (
            "terms of service",
            "service level",
            "service-level agreement",
            "sla",
            "uptime",
            "support hours",
            "response time",
        ),
        "privacy_terms": (
            "privacy policy",
            "personal data",
            "data protection",
            "data subject",
            "consent",
        ),
        "complaint_handling": (
            "complaint",
            "grievance",
            "redress",
            "ombudsman",
            "consumer affairs",
            "অভিযোগ",
        ),
        "foreign_investment": (
            "foreign private investment",
            "foreign capital",
            "non-resident",
            "remittance",
            "repatriation",
        ),
        "contract": ("contract", "agreement", "consideration", "offer", "acceptance"),
    }
    # "foreigner" alone is too noisy (appears in many statutes contextually);
    # require an explicit work-permit framing before tagging.
    for tag, needles in checks.items():
        if any(n in blob for n in needles):
            tags.add(tag)
    return sorted(tags)


def is_extraction_noise(text: str) -> bool:
    cid_count = text.count("(cid:")
    if cid_count >= 5:
        return True
    if cid_count >= 2 and len(text) < 600:
        return True
    return False


class SimpleResponse:
    def __init__(self, *, url: str, status_code: int, headers: dict[str, str], content: bytes):
        self.url = url
        self.status_code = status_code
        self.headers = headers
        self.content = content
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"


def make_session() -> object:
    if requests is not None:
        return requests.Session()
    return object()


def encode_url(url: str) -> str:
    parts = urlsplit(url)
    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&%/:+,-._~")
    return urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def http_get(session: object, url: str, timeout: int = 45) -> SimpleResponse:
    url = encode_url(url)
    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            if requests is not None and hasattr(session, "get"):
                resp = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
                if resp.status_code >= 500:
                    raise RuntimeError(f"server error {resp.status_code} for {url}")
                return resp
            req = Request(url, headers=HEADERS)
            context = ssl._create_unverified_context() if attempt >= 2 and url.lower().startswith("https://") else None
            with urlopen(req, timeout=timeout, context=context) as resp:  # noqa: S310 - controlled source URLs
                return SimpleResponse(
                    url=resp.geturl(),
                    status_code=getattr(resp, "status", 200),
                    headers=dict(resp.headers.items()),
                    content=resp.read(),
                )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if requests is not None and url.lower().startswith("https://"):
                try:
                    req = Request(url, headers=HEADERS)
                    context = ssl._create_unverified_context()
                    with urlopen(req, timeout=timeout, context=context) as resp:  # noqa: S310
                        return SimpleResponse(
                            url=resp.geturl(),
                            status_code=getattr(resp, "status", 200),
                            headers=dict(resp.headers.items()),
                            content=resp.read(),
                        )
                except Exception as fallback_exc:  # noqa: BLE001
                    last_error = fallback_exc
            if attempt < 3:
                time.sleep(min(12, 2 ** attempt))
    raise RuntimeError(f"fetch failed after retries for {url}: {last_error}")


def ensure_dirs(out_dir: str) -> dict[str, str]:
    paths = {
        "data": out_dir,
        "raw": os.path.join(out_dir, "raw"),
        "text": os.path.join(out_dir, "text"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def reset_failure_log(out_dir: str) -> None:
    path = os.path.join(out_dir, "failed_sources.log")
    if os.path.exists(path):
        os.remove(path)


def write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_failure(out_dir: str, url: str, reason: str) -> None:
    path = os.path.join(out_dir, "failed_sources.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{now_iso()} | {url} | {reason}\n")
    log.warning("failed %s: %s", url, reason)


def extract_pdf_text(pdf_bytes: bytes, max_pages: int) -> tuple[str, int, list[str]]:
    warnings: list[str] = []
    if pdfplumber is None:
        return "", 0, ["pdfplumber is not installed; PDF text extraction skipped"]
    pages_text: list[str] = []
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            page_limit = min(len(pdf.pages), max_pages)
            for page in pdf.pages[:page_limit]:
                pages_text.append(page.extract_text() or "")
            if len(pdf.pages) > page_limit:
                warnings.append(f"PDF truncated to first {page_limit} of {len(pdf.pages)} pages")
            return normalize_text("\n\n".join(pages_text)), page_limit, warnings
    except Exception as exc:  # noqa: BLE001
        return "", 0, [f"PDF extraction failed: {exc.__class__.__name__}: {exc}"]


def chunk_text(
    *,
    document_id: str,
    title: str,
    url: str,
    source_family: str,
    source_type: str,
    source_authority: str,
    retrieved_at: str,
    retrieval_date: str,
    text: str,
    chunk_chars: int,
    overlap_chars: int,
    section_id: Optional[str] = None,
    heading: str = "",
) -> list[SourceChunk]:
    text = normalize_text(text)
    if len(text) < 180:
        return []
    chunks: list[SourceChunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            split = text.rfind("\n", start, end)
            if split <= start + 400:
                split = text.rfind(". ", start, end)
            if split > start + 400:
                end = split + 1
        piece = normalize_text(text[start:end])
        if len(piece) >= 180 and not is_extraction_noise(piece):
            chunk_id = sha256_text(f"{document_id}:{section_id}:{idx}:{piece[:200]}")
            chunks.append(
                SourceChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=title,
                    url=url,
                    source_family=source_family,
                    source_type=source_type,
                    source_authority=source_authority,
                    retrieved_at=retrieved_at,
                    retrieval_date=retrieval_date,
                    section_id=section_id,
                    heading=heading,
                    text=piece,
                    tokenish_length=max(1, len(piece) // 4),
                    tags=tags_for(title, piece, source_family),
                )
            )
            idx += 1
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def record_document(
    *,
    paths: dict[str, str],
    title: str,
    url: str,
    source_family: str,
    source_type: str,
    source_authority: str,
    content_type: str,
    raw_bytes: bytes,
    extracted_text: str,
    http_status: int,
    selected_reason: str,
    page_count: int = 0,
    warnings: Optional[list[str]] = None,
) -> SourceRecord:
    digest = sha256_bytes(raw_bytes)
    parsed = urlparse(url)
    ext = ".pdf" if "pdf" in content_type.lower() or parsed.path.lower().endswith(".pdf") else ".html"
    title = clean_title(title)
    stem = f"{slugify(title)}-{digest[:12]}"
    raw_path = os.path.join(paths["raw"], stem + ext)
    text_path = os.path.join(paths["text"], stem + ".txt")
    write_bytes(raw_path, raw_bytes)
    write_text(text_path, extracted_text)
    retrieved_at = now_iso()
    return SourceRecord(
        document_id=digest[:24],
        title=title,
        url=url,
        source_family=source_family,
        source_type=source_type,
        source_authority=source_authority,
        content_type=content_type,
        retrieved_at=retrieved_at,
        retrieval_date=retrieved_at[:10],
        sha256=digest,
        raw_path=raw_path,
        text_path=text_path,
        http_status=http_status,
        selected_reason=selected_reason,
        source_language=detect_language(extracted_text),
        page_count_extracted=page_count,
        warnings=warnings or [],
    )


def fetch_document(
    session: requests.Session,
    *,
    paths: dict[str, str],
    url: str,
    title_hint: str,
    source_family: str,
    source_type: str,
    source_authority: str,
    selected_reason: str,
    max_pdf_pages: int,
) -> tuple[Optional[SourceRecord], str]:
    resp = http_get(session, url)
    content_type = resp.headers.get("Content-Type", "")
    raw = resp.content
    looks_pdf = "pdf" in content_type.lower() or urlparse(resp.url).path.lower().endswith(".pdf")
    if looks_pdf:
        text, page_count, warnings = extract_pdf_text(raw, max_pdf_pages)
        title = normalize_text(title_hint) or os.path.basename(urlparse(resp.url).path) or "PDF source"
        if len(text) < 100:
            warnings.append("extracted PDF text is short; source may be scanned or image-only")
        rec = record_document(
            paths=paths,
            title=title,
            url=resp.url,
            source_family=source_family,
            source_type=source_type,
            source_authority=source_authority,
            content_type="application/pdf",
            raw_bytes=raw,
            extracted_text=text,
            http_status=resp.status_code,
            selected_reason=selected_reason,
            page_count=page_count,
            warnings=warnings,
        )
        return rec, text
    html = resp.content.decode(resp.encoding or resp.apparent_encoding or "utf-8", errors="replace")
    title, text = visible_text_from_html(html)
    if title_hint and (not title or title == "Untitled source"):
        title = title_hint
    rec = record_document(
        paths=paths,
        title=title,
        url=resp.url,
        source_family=source_family,
        source_type=source_type,
        source_authority=source_authority,
        content_type="text/html",
        raw_bytes=raw,
        extracted_text=text,
        http_status=resp.status_code,
        selected_reason=selected_reason,
    )
    return rec, text


def strip_tags(fragment: str) -> str:
    fragment = re.sub(r"<[^>]+>", " ", fragment or "")
    return normalize_text(html_lib.unescape(fragment))


def link_title(anchor: Tag) -> str:
    row = anchor.find_parent("tr")
    if row:
        text = normalize_text(row.get_text(" ", strip=True))
        text = re.sub(r"\bView\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\d+\s+", "", text)
        return normalize_text(text)
    parent = anchor.parent
    if parent:
        text = normalize_text(parent.get_text(" ", strip=True))
        if text and text.lower() != "view":
            return re.sub(r"\bView\b", "", text, flags=re.IGNORECASE).strip()
    prev = anchor.find_previous(string=True)
    return normalize_text(str(prev or anchor.get_text(" ", strip=True)))


def discover_links(page_html: str, base_url: str, keywords: Iterable[str]) -> list[tuple[str, str]]:
    lowered_keywords = tuple(k.lower() for k in keywords)
    if BeautifulSoup is None:
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        pattern = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
        for match in pattern.finditer(page_html):
            href = urljoin(base_url, html_lib.unescape(match.group(1)))
            anchor_text = strip_tags(match.group(2))
            prefix = page_html[max(0, match.start() - 300):match.start()]
            prefix_text = strip_tags(prefix)
            title = anchor_text if anchor_text.lower() != "view" else short_tail(prefix_text)
            blob = f"{title} {anchor_text} {href}".lower()
            if any(k in blob for k in lowered_keywords) and href not in seen:
                seen.add(href)
                found.append((href, title or href))
        return found

    soup = BeautifulSoup(page_html, "lxml")
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"])
        title = link_title(anchor)
        blob = f"{title} {anchor.get_text(' ', strip=True)} {href}".lower()
        if not any(k in blob for k in lowered_keywords):
            continue
        if href in seen:
            continue
        seen.add(href)
        found.append((href, title or href))
    return found


def short_tail(text: str) -> str:
    text = normalize_text(text)
    if len(text) <= 180:
        return text
    return text[-180:]


def parse_section_number(section_text: str, fallback: int) -> str:
    digit_map = str.maketrans({
        "\u09e6": "0", "\u09e7": "1", "\u09e8": "2", "\u09e9": "3", "\u09ea": "4",
        "\u09eb": "5", "\u09ec": "6", "\u09ed": "7", "\u09ee": "8", "\u09ef": "9",
    })
    converted = (section_text or "").translate(digit_map)
    match = re.match(r"\s*(\d+[A-Za-z]?)\s*[\.\)\u0964-]", converted)
    return match.group(1) if match else str(fallback)


def parse_bdlaws_sections(html: str, fallback_title: str) -> tuple[str, list[dict[str, str]], str]:
    if BeautifulSoup is None:
        title, body = visible_text_from_html(html)
        return title or fallback_title, [], body

    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find(id="printheader") or soup.find("h3") or soup.title
    title = normalize_text(title_el.get_text(" ", strip=True)) if title_el else fallback_title
    body = normalize_text((soup.find(id="hide") or soup.body or soup).get_text(" ", strip=True))
    sections: list[dict[str, str]] = []
    for row in soup.select("div.row.lineremoves, div.row.lineremove"):
        details = row.select_one(".txt-details")
        if not details:
            continue
        text = normalize_text(details.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        heading_el = row.select_one(".txt-head")
        heading = normalize_text(heading_el.get_text(" ", strip=True)) if heading_el else ""
        sections.append(
            {
                "section_id": parse_section_number(text, len(sections) + 1),
                "heading": heading,
                "text": text,
            }
        )
    return title, sections, body


def collect_bdlaws(
    session: object,
    *,
    paths: dict[str, str],
    max_acts: int,
    max_pdf_pages: int,
    chunk_chars: int,
    overlap_chars: int,
    delay: float,
    out_dir: str,
    include_candidate_acts: bool = False,
) -> tuple[list[SourceRecord], list[SourceChunk]]:
    records: list[SourceRecord] = []
    chunks: list[SourceChunk] = []
    for item in selected_bdlaws_acts(include_candidate_acts)[:max_acts]:
        url = BDLAWS_ACT_PRINT.format(act_id=item["act_id"])
        try:
            resp = http_get(session, url)
            raw = resp.content
            html = raw.decode(resp.encoding or resp.apparent_encoding or "utf-8", errors="replace")
            title, sections, full_body = parse_bdlaws_sections(html, item["title"])
            rec = record_document(
                paths=paths,
                title=title or item["title"],
                url=resp.url,
                source_family=item["family"],
                source_type="bdlaws_act_print",
                source_authority="Laws of Bangladesh",
                content_type="text/html",
                raw_bytes=raw,
                extracted_text=full_body,
                http_status=resp.status_code,
                selected_reason=item["reason"],
            )
            records.append(rec)
            if sections:
                for section in sections:
                    chunks.extend(
                        chunk_text(
                            document_id=rec.document_id,
                            title=rec.title,
                            url=f"{rec.url}#section={section['section_id']}",
                            source_family=rec.source_family,
                            source_type=rec.source_type,
                            source_authority=rec.source_authority,
                            retrieved_at=rec.retrieved_at,
                            retrieval_date=rec.retrieval_date,
                            section_id=section["section_id"],
                            heading=section["heading"],
                            text=section["text"],
                            chunk_chars=chunk_chars,
                            overlap_chars=overlap_chars,
                        )
                    )
            else:
                chunks.extend(
                    chunk_text(
                        document_id=rec.document_id,
                        title=rec.title,
                        url=rec.url,
                        source_family=rec.source_family,
                        source_type=rec.source_type,
                        source_authority=rec.source_authority,
                        retrieved_at=rec.retrieved_at,
                        retrieval_date=rec.retrieval_date,
                        section_id=None,
                        heading="",
                        text=full_body,
                        chunk_chars=chunk_chars,
                        overlap_chars=overlap_chars,
                    )
                )
            log.info("bdlaws %s sections=%d chunks=%d", item["act_id"], len(sections), len(chunks))
            time.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            append_failure(out_dir, url, f"bdlaws_fetch_failed:{exc.__class__.__name__}:{exc}")
    return records, chunks


def collect_indexed_links(
    session: object,
    *,
    paths: dict[str, str],
    index_url: str,
    index_title: str,
    source_family: str,
    source_type: str,
    source_authority: str,
    keywords: Iterable[str],
    max_links: int,
    max_pdf_pages: int,
    chunk_chars: int,
    overlap_chars: int,
    delay: float,
    out_dir: str,
) -> tuple[list[SourceRecord], list[SourceChunk]]:
    records: list[SourceRecord] = []
    chunks: list[SourceChunk] = []
    try:
        resp = http_get(session, index_url)
        raw = resp.content
        html = raw.decode(resp.encoding or resp.apparent_encoding or "utf-8", errors="replace")
        title, text = visible_text_from_html(html)
        index_rec = record_document(
            paths=paths,
            title=title or index_title,
            url=resp.url,
            source_family=source_family,
            source_type=f"{source_type}_index",
            source_authority=source_authority,
            content_type="text/html",
            raw_bytes=raw,
            extracted_text=text,
            http_status=resp.status_code,
            selected_reason=f"index page for discovering {index_title}",
        )
        records.append(index_rec)
        chunks.extend(
            chunk_text(
                document_id=index_rec.document_id,
                title=index_rec.title,
                url=index_rec.url,
                source_family=index_rec.source_family,
                source_type=index_rec.source_type,
                source_authority=index_rec.source_authority,
                retrieved_at=index_rec.retrieved_at,
                retrieval_date=index_rec.retrieval_date,
                section_id=None,
                heading="index",
                text=text,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )[:3]
        )
        links = discover_links(html, resp.url, keywords)[:max_links]
        log.info("%s discovered selected links=%d", source_authority, len(links))
    except Exception as exc:  # noqa: BLE001
        append_failure(out_dir, index_url, f"index_fetch_failed:{exc.__class__.__name__}:{exc}")
        return records, chunks

    for url, title_hint in links:
        try:
            rec, text = fetch_document(
                session,
                paths=paths,
                url=url,
                title_hint=title_hint,
                source_family=source_family,
                source_type=source_type,
                source_authority=source_authority,
                selected_reason=f"matched source keyword for {index_title}: {title_hint}",
                max_pdf_pages=max_pdf_pages,
            )
            if rec is None:
                continue
            records.append(rec)
            chunks.extend(
                chunk_text(
                    document_id=rec.document_id,
                    title=rec.title,
                    url=rec.url,
                    source_family=rec.source_family,
                    source_type=rec.source_type,
                    source_authority=rec.source_authority,
                    retrieved_at=rec.retrieved_at,
                    retrieval_date=rec.retrieval_date,
                    section_id=None,
                    heading="",
                    text=text,
                    chunk_chars=chunk_chars,
                    overlap_chars=overlap_chars,
                )
            )
            log.info("fetched %s chunks=%d", rec.title[:80], len(chunks))
            time.sleep(delay)
        except Exception as exc:  # noqa: BLE001
            append_failure(out_dir, url, f"linked_fetch_failed:{exc.__class__.__name__}:{exc}")
    return records, chunks


def dedupe_chunks(chunks: Iterable[SourceChunk]) -> list[SourceChunk]:
    seen: set[str] = set()
    out: list[SourceChunk] = []
    for chunk in chunks:
        key = sha256_text(normalize_text(chunk.text).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def write_outputs(out_dir: str, records: list[SourceRecord], chunks: list[SourceChunk]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "description": "Live-source manifest for Bangladesh contract/labor/policy vetting SFT data.",
        "generated_at": now_iso(),
        "source_urls": {
            "bepza": BEPZA_ACTS_URL,
            "cptu": CPTU_STD_URL,
            "bdlaws_act_print": BDLAWS_ACT_PRINT,
        },
        "records": [asdict(record) for record in records],
        "counts": {
            "records": len(records),
            "chunks": len(chunks),
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "source_chunks.jsonl"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    lines = [
        "# Source Manifest Card",
        "",
        f"Generated: {manifest['generated_at']}",
        f"Documents: {len(records)}",
        f"Chunks: {len(chunks)}",
        "",
        "## Sources",
    ]
    for rec in records:
        lines.append(f"- {rec.title} ({rec.source_authority}) - {rec.url}")
    write_text(os.path.join(out_dir, "source_manifest_card.md"), "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data")
    # Defaults tilted toward general business statutes. BEPZA stays in the
    # mix as a specialisation but no longer dominates the budget; CPTU and
    # bdlaws were under-represented relative to their broad applicability.
    parser.add_argument("--max-bepza-links", type=int, default=6)
    parser.add_argument("--max-cptu-docs", type=int, default=6)
    parser.add_argument("--max-bdlaws-acts", type=int, default=9)
    parser.add_argument(
        "--include-candidate-acts",
        action="store_true",
        help=(
            "Include candidate bdlaws acts (Partnership, Arbitration, "
            "Negotiable Instruments, Specific Relief, Foreign Exchange Regulation). IDs are "
            "best-effort; unresolved acts are logged and skipped."
        ),
    )
    parser.add_argument("--max-pdf-pages", type=int, default=40)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=160)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    paths = ensure_dirs(args.out_dir)
    reset_failure_log(args.out_dir)
    session = make_session()

    all_records: list[SourceRecord] = []
    all_chunks: list[SourceChunk] = []

    records, chunks = collect_bdlaws(
        session,
        paths=paths,
        max_acts=args.max_bdlaws_acts,
        max_pdf_pages=args.max_pdf_pages,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        delay=args.delay,
        out_dir=args.out_dir,
        include_candidate_acts=args.include_candidate_acts,
    )
    all_records.extend(records)
    all_chunks.extend(chunks)

    if args.max_bepza_links > 0:
        records, chunks = collect_indexed_links(
            session,
            paths=paths,
            index_url=BEPZA_ACTS_URL,
            index_title="BEPZA acts, policies, and SROs",
            source_family="epz_bepza_policy",
            source_type="bepza_policy_document",
            source_authority="BEPZA",
            keywords=BEPZA_KEYWORDS,
            max_links=args.max_bepza_links,
            max_pdf_pages=args.max_pdf_pages,
            chunk_chars=args.chunk_chars,
            overlap_chars=args.overlap_chars,
            delay=args.delay,
            out_dir=args.out_dir,
        )
        all_records.extend(records)
        all_chunks.extend(chunks)

    if args.max_cptu_docs > 0:
        cptu_urls = [CPTU_STD_URL] + CPTU_STD_FALLBACK_URLS
        records = []
        chunks = []
        for cptu_url in cptu_urls:
            records, chunks = collect_indexed_links(
                session,
                paths=paths,
                index_url=cptu_url,
                index_title="BPPA/CPTU standard tender documents",
                source_family="procurement_contract_templates",
                source_type="cptu_standard_tender_document",
                source_authority="BPPA/CPTU",
                keywords=CPTU_KEYWORDS,
                max_links=args.max_cptu_docs,
                max_pdf_pages=args.max_pdf_pages,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
                delay=args.delay,
                out_dir=args.out_dir,
            )
            if records:
                break
        all_records.extend(records)
        all_chunks.extend(chunks)

    all_chunks = dedupe_chunks(all_chunks)
    write_outputs(args.out_dir, all_records, all_chunks)
    log.info("wrote records=%d chunks=%d to %s", len(all_records), len(all_chunks), args.out_dir)
    print(json.dumps({"records": len(all_records), "chunks": len(all_chunks)}, indent=2))
    return 0 if all_chunks else 2


if __name__ == "__main__":
    sys.exit(main())
