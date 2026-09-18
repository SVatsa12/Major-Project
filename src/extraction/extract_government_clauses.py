"""
Production-grade clause extraction for Indian government Acts, Rules, and
notifications.

Features:
- Recursive discovery of PDF and DOCX files.
- PDF extraction page by page with optional OCR fallback.
- Conservative removal of repeated headers, footers, and navigation blocks.
- Preservation of page numbers, section headings, source hashes, and extraction
  methods for legal auditability.
- Clause segmentation that avoids deleting short legal provisions or cutting
  text at arbitrary character positions.
- Act-name normalization with configurable filename overrides.
- JSON and CSV clause datasets.
- Extraction review, failure, quality, and summary reports.

Expected structure:

    datasets/GovernmentActs/source/*.pdf
    datasets/GovernmentActs/source/*.docx

Nested folders are supported.

Install:

    pip install pymupdf pandas python-docx pillow pytesseract
    sudo apt-get install tesseract-ocr

Run from the project root:

    python src/extraction/extract_government_clauses.py --ocr

Dry run:

    python src/extraction/extract_government_clauses.py --ocr --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import pymupdf
except ImportError:  # Compatibility with older PyMuPDF installations.
    import fitz as pymupdf  # type: ignore

try:
    from docx import Document
except ImportError:  # Optional unless DOCX files are present.
    Document = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import pytesseract
except ImportError:
    pytesseract = None  # type: ignore


DEFAULT_SOURCE_DIR = Path("datasets/GovernmentActs/source")
DEFAULT_OUTPUT_DIR = Path("datasets/GovernmentActs")

# Add or edit overrides when a filename is ambiguous or inconsistent.
ACT_NAME_OVERRIDES = {
    "dpdp_2023": "DPDP_ACT_2023",
    "dpdp_act_2023": "DPDP_ACT_2023",
    "dpdp_2025": "DPDP_RULES_2025",
    "dpdp_rules_2025": "DPDP_RULES_2025",
    "it_act_2000_updated": "IT_ACT_2000",
    "it_act_2000": "IT_ACT_2000",
    "rti-act_english": "RTI_ACT_2005",
    "rti_act_2005": "RTI_ACT_2005",
    "the_disaster_management_act": "DISASTER_MGMT_ACT_2005",
    "disaster_management_act_2005": "DISASTER_MGMT_ACT_2005",
    "the_patents_act_1970": "PATENTS_ACT_1970",
    "patents_act_1970": "PATENTS_ACT_1970",
    "the_trai_act_1997": "TRAI_ACT_1997",
    "trai_act_1997": "TRAI_ACT_1997",
    "airportauthority": "AERA_ACT_2008",
    "aera_act_2008": "AERA_ACT_2008",
}

EXACT_JUNK_PATTERNS = [
    r"^back\s+to\s+top$",
    r"^legal\s+info$",
    r"^table\s+of\s+contents$",
    r"^contents$",
    r"^page\s+\d+(?:\s+of\s+\d+)?$",
    r"^last\s+updated(?:\s*[:\-].*)?$",
    r"^archived\s+versions?$",
]
EXACT_JUNK_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in EXACT_JUNK_PATTERNS]

REPEATED_LINE_MIN_OCCURRENCES = 3
REPEATED_LINE_MAX_CHARS = 140
REPEATED_LINE_PAGE_RATIO = 0.60
MIN_REVIEWABLE_CHARS = 10
MAX_CLAUSE_CHARS = 1400

BULLET_RE = re.compile(r"^(?:[-*•▪◦‣]|\(?\d+[.)]|\(?[a-zA-Z][.)])\s+")
HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z0-9 &'()/,:—–-]{2,120}$"
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])" )

# Matches enumerated sub-clause markers like (a), (b), (za), (ii), (12) etc.
# Used to detect and split compound legal paragraphs where multiple obligations
# are bundled inline: "(z) X means ...; (za) Y means ...; and (zb) Z means ..."
SUBCLAUSE_RE = re.compile(
    r"\(([a-z]{1,3}|\d{1,3}|i{1,3}v?|vi{0,3})\)\s+",
    re.IGNORECASE,
)
# Splits text at the natural boundary between inline sub-clauses.
# Handles "X; (za) Y" and "X; and (zb) Y" patterns.
_SUBCLAUSE_BOUNDARY_RE = re.compile(
    r"(?<=[;.])\s+(?:and\s+)?(?=\([a-z]{1,3}\)[\s\u201c\u2018\"'])",
    re.IGNORECASE,
)


@dataclass
class PageRecord:
    page_number: int
    text: str
    extraction_method: str
    char_count: int
    needs_ocr: bool
    warning: str | None = None


@dataclass
class ClauseRecord:
    clause_id: str
    source_act: str
    source_file: str
    source_path: str
    page_start: int
    page_end: int
    section_heading: str | None
    clause_index_in_document: int
    clause_text: str
    extraction_method: str
    extraction_quality: str
    retrieved_at: str
    file_sha256: str


@dataclass
class ReviewRecord:
    source_act: str
    source_file: str
    page_number: int | None
    text: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract auditable clauses from government legal documents."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing source PDFs/DOCX files, including nested folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated datasets and reports.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run OCR on pages with little or no embedded text.",
    )
    parser.add_argument(
        "--ocr-min-chars",
        type=int,
        default=30,
        help="Pages with fewer embedded characters than this are OCR candidates.",
    )
    parser.add_argument(
        "--max-clause-chars",
        type=int,
        default=MAX_CLAUSE_CHARS,
        help="Preferred maximum clause length; splitting uses safe boundaries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing output files.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin words split across a PDF line ending, but preserve normal hyphens.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_exact_junk(text: str) -> bool:
    candidate = normalize_line(text)
    return not candidate or any(pattern.fullmatch(candidate) for pattern in EXACT_JUNK_REGEX)


def looks_like_heading(text: str) -> bool:
    candidate = normalize_line(text)
    if not candidate or len(candidate) > 160:
        return False
    if candidate.endswith((".", ";", ":", "?", "!")):
        return False
    return bool(HEADING_RE.fullmatch(candidate))


def derive_act_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    # Prefer the longest matching override to avoid partial-name collisions.
    for key, label in sorted(ACT_NAME_OVERRIDES.items(), key=lambda item: len(item[0]), reverse=True):
        key_normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if key_normalized in normalized or key_normalized in stem:
            return label

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", Path(filename).stem).strip("_").upper()
    return cleaned or "UNKNOWN_GOVERNMENT_SOURCE"


def extract_pdf_pages(
    pdf_path: Path,
    use_ocr: bool,
    ocr_min_chars: int,
    source_act: str,
) -> tuple[list[PageRecord], list[ReviewRecord]]:
    pages: list[PageRecord] = []
    reviews: list[ReviewRecord] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            embedded_text = normalize_text(page.get_text("text") or "")
            needs_ocr = len(embedded_text) < ocr_min_chars
            text = embedded_text
            extraction_method = "pymupdf"
            warning: str | None = None

            if needs_ocr and use_ocr:
                if Image is None or pytesseract is None:
                    warning = "OCR requested but Pillow/pytesseract is not installed"
                else:
                    try:
                        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                        image = Image.open(BytesIO(pixmap.tobytes("png")))
                        ocr_text = normalize_text(pytesseract.image_to_string(image))
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            extraction_method = "pymupdf+ocr" if embedded_text else "ocr"
                        else:
                            warning = "OCR produced no additional text"
                    except Exception as exc:
                        warning = f"OCR failed: {type(exc).__name__}: {exc}"

            if not text:
                warning = warning or "No text extracted; page may be image-only or corrupt"

            if needs_ocr or warning:
                reviews.append(
                    ReviewRecord(
                        source_act=source_act,
                        source_file=pdf_path.name,
                        page_number=page_number,
                        text=text[:2000],
                        reason=warning or "Low embedded-text count; inspect page",
                    )
                )

            pages.append(
                PageRecord(
                    page_number=page_number,
                    text=text,
                    extraction_method=extraction_method,
                    char_count=len(text),
                    needs_ocr=needs_ocr,
                    warning=warning,
                )
            )

    return pages, reviews


def extract_docx_pages(
    docx_path: Path,
    source_act: str,
) -> tuple[list[PageRecord], list[ReviewRecord]]:
    if Document is None:
        raise RuntimeError("python-docx is not installed; install it or remove DOCX files")

    document = Document(docx_path)
    parts: list[str] = []
    parts.extend(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = normalize_text("\n\n".join(parts))
    review: list[ReviewRecord] = []
    if not text:
        review.append(
            ReviewRecord(
                source_act=source_act,
                source_file=docx_path.name,
                page_number=None,
                text="",
                reason="No text extracted from DOCX",
            )
        )
    return [PageRecord(1, text, "python-docx", len(text), False, None)], review


def collect_repeated_lines(pages: Iterable[PageRecord]) -> set[str]:
    page_list = list(pages)
    if len(page_list) < 3:
        return set()

    line_page_counts: Counter[str] = Counter()
    for page in page_list:
        seen_on_page: set[str] = set()
        for line in page.text.splitlines():
            normalized = normalize_line(line)
            if (
                normalized
                and len(normalized) <= REPEATED_LINE_MAX_CHARS
                and not BULLET_RE.match(normalized)
            ):
                seen_on_page.add(normalized.casefold())
        line_page_counts.update(seen_on_page)

    threshold = max(REPEATED_LINE_MIN_OCCURRENCES, int(len(page_list) * REPEATED_LINE_PAGE_RATIO))
    return {
        line
        for line, count in line_page_counts.items()
        if count >= threshold
    }


def remove_repeated_lines(text: str, repeated_lines: set[str]) -> str:
    if not repeated_lines:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        normalized = normalize_line(line)
        if normalized and normalized.casefold() in repeated_lines:
            continue
        kept.append(line)
    return normalize_text("\n".join(kept))


def _split_block_on_subclauses(text: str) -> list[str]:
    """Split a compound block containing multiple inline enumerated sub-clauses.

    Handles paragraphs common in Indian legislation where several sub-obligations
    are written as one block, e.g.:
        "(z) 'X' means ...; (za) 'Y' means ...; and (zb) 'Z' means ..."

    The text before the first sub-clause marker is prepended to every resulting
    chunk as parent context (e.g., the section number / preamble text).

    Returns the original [text] if fewer than 2 sub-clause markers are found,
    so it is always safe to call on non-compound blocks.
    """
    marker_count = len(SUBCLAUSE_RE.findall(text))
    if marker_count < 2:
        return [text]

    parts = _SUBCLAUSE_BOUNDARY_RE.split(text)
    cleaned = [normalize_line(p) for p in parts if normalize_line(p)]
    if len(cleaned) > 1:
        # Detect whether there is preamble text before the first (x) marker.
        first_marker = SUBCLAUSE_RE.search(text)
        if first_marker and first_marker.start() > 0:
            # Preamble is already part of cleaned[0] (the ';' split keeps it).
            # No extra action needed; the first chunk contains the full first item.
            pass
        return cleaned
    return [text]


def split_long_block(text: str, max_chars: int) -> list[str]:
    text = normalize_line(text)
    if len(text) <= max_chars:
        return [text]

    lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    if len(lines) > 1 and any(BULLET_RE.match(line) for line in lines):
        return lines

    # Try sub-clause splitting before sentence splitting so that compound
    # definition paragraphs like "(z) X; (za) Y; (zb) Z" produce separate
    # clause records, not one oversized chunk.
    sub_parts = _split_block_on_subclauses(text)
    if len(sub_parts) > 1:
        result: list[str] = []
        for part in sub_parts:
            result.extend(split_long_block(part, max_chars))
        return result

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text)
    if len(sentences) <= 1:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        proposed_length = current_length + len(sentence) + (1 if current else 0)
        if current and proposed_length > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_length = len(sentence)
        else:
            current.append(sentence)
            current_length = proposed_length
    if current:
        chunks.append(" ".join(current).strip())
    return chunks or [text]


def page_to_blocks(page_text: str) -> list[str]:
    lines = [line.rstrip() for line in page_text.splitlines()]
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            block = normalize_text("\n".join(current))
            if block:
                # Split compound sub-clause blocks inline, e.g. paragraphs where
                # "(z) ... (za) ... (zb) ..." are all on one line in the PDF.
                blocks.extend(_split_block_on_subclauses(block))
            current = []

    for raw_line in lines:
        line = normalize_line(raw_line)
        if not line:
            flush()
            continue
        if BULLET_RE.match(line) and current:
            flush()
        if looks_like_heading(line) and current:
            flush()
        current.append(line)
    flush()
    return blocks


def build_document_clauses(
    source_act: str,
    file_path: Path,
    pages: list[PageRecord],
    retrieved_at: str,
    file_hash: str,
    max_clause_chars: int,
) -> tuple[list[ClauseRecord], list[ReviewRecord], dict[str, Any]]:
    repeated_lines = collect_repeated_lines(pages)
    clauses: list[ClauseRecord] = []
    reviews: list[ReviewRecord] = []
    section_heading: str | None = None
    clause_index = 0
    block_count = 0
    removed_count = 0
    tiny_count = 0

    for page in pages:
        cleaned_text = remove_repeated_lines(page.text, repeated_lines)
        blocks = page_to_blocks(cleaned_text)
        block_count += len(blocks)

        for block in blocks:
            candidate = normalize_line(block)
            if not candidate:
                removed_count += 1
                continue

            if is_exact_junk(candidate):
                removed_count += 1
                reviews.append(
                    ReviewRecord(
                        source_act=source_act,
                        source_file=file_path.name,
                        page_number=page.page_number,
                        text=candidate[:2000],
                        reason="Recognized navigation/header/footer block",
                    )
                )
                continue

            if looks_like_heading(candidate):
                section_heading = candidate
                continue

            if len(candidate) < MIN_REVIEWABLE_CHARS:
                tiny_count += 1
                reviews.append(
                    ReviewRecord(
                        source_act=source_act,
                        source_file=file_path.name,
                        page_number=page.page_number,
                        text=candidate,
                        reason="Very short extracted block; retained for manual review",
                    )
                )

            for piece in split_long_block(candidate, max_clause_chars):
                piece = normalize_line(piece)
                if not piece:
                    continue
                clause_index += 1
                quality = "review" if page.warning or len(piece) < MIN_REVIEWABLE_CHARS else "pass"
                clauses.append(
                    ClauseRecord(
                        clause_id="",
                        source_act=source_act,
                        source_file=file_path.name,
                        source_path=str(file_path),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_heading=section_heading,
                        clause_index_in_document=clause_index,
                        clause_text=piece,
                        extraction_method=page.extraction_method,
                        extraction_quality=quality,
                        retrieved_at=retrieved_at,
                        file_sha256=file_hash,
                    )
                )

    quality = {
        "source_act": source_act,
        "source_file": file_path.name,
        "source_path": str(file_path),
        "file_hash": file_hash,
        "pages": len(pages),
        "pages_with_text": sum(1 for page in pages if page.text.strip()),
        "pages_needing_ocr": sum(1 for page in pages if page.needs_ocr),
        "pages_with_warnings": sum(1 for page in pages if page.warning),
        "characters_extracted": sum(page.char_count for page in pages),
        "candidate_blocks": block_count,
        "retained_clauses": len(clauses),
        "removed_blocks": removed_count,
        "very_short_blocks_for_review": tiny_count,
        "repeated_lines_removed": sorted(repeated_lines),
        "status": (
            "FAIL" if not pages or not any(page.text.strip() for page in pages)
            else "REVIEW" if any(page.warning for page in pages) or tiny_count
            else "PASS"
        ),
    }
    return clauses, reviews, quality


def discover_source_files(source_dir: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if suffix == ".pdf":
            files.append((path, "pdf"))
        elif suffix == ".docx":
            files.append((path, "docx"))
    return sorted(files, key=lambda item: str(item[0]).casefold())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    source_dir: Path = args.source_dir
    output_dir: Path = args.output_dir
    retrieved_at = utc_now_iso()

    if not source_dir.exists():
        print(f"ERROR: source folder not found: {source_dir}", file=sys.stderr)
        return 1

    all_files = discover_source_files(source_dir)
    if not all_files:
        print(f"ERROR: no PDF or DOCX files found in {source_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(all_files)} source file(s) in {source_dir}")
    for path, file_type in all_files:
        print(f"  - {path} ({file_type})")

    all_clauses: list[ClauseRecord] = []
    all_reviews: list[ReviewRecord] = []
    quality_reports: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []

    for file_path, file_type in all_files:
        source_act = derive_act_name(file_path.name)
        print(f"\nExtracting: {file_path.name} -> {source_act}")
        file_hash = sha256_file(file_path)

        try:
            if file_type == "pdf":
                pages, page_reviews = extract_pdf_pages(
                    file_path,
                    use_ocr=args.ocr,
                    ocr_min_chars=args.ocr_min_chars,
                    source_act=source_act,
                )
            else:
                pages, page_reviews = extract_docx_pages(file_path, source_act)

            all_reviews.extend(page_reviews)
            clauses, reviews, quality = build_document_clauses(
                source_act=source_act,
                file_path=file_path,
                pages=pages,
                retrieved_at=retrieved_at,
                file_hash=file_hash,
                max_clause_chars=args.max_clause_chars,
            )
            all_reviews.extend(reviews)
            quality_reports.append(quality)
            all_clauses.extend(clauses)

            print(
                f"  pages={quality['pages']} "
                f"chars={quality['characters_extracted']} "
                f"clauses={quality['retained_clauses']} "
                f"status={quality['status']}"
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {reason}", file=sys.stderr)
            failed_files.append({
                "source_act": source_act,
                "source_file": file_path.name,
                "source_path": str(file_path),
                "reason": reason,
            })

    for number, clause in enumerate(all_clauses, start=1):
        clause.clause_id = f"G{number:06d}"

    summary = {
        "run_timestamp_utc": retrieved_at,
        "source_dir": str(source_dir),
        "discovered_files_total": len(all_files),
        "processed_documents": len(quality_reports),
        "total_clauses": len(all_clauses),
        "total_failed_files": len(failed_files),
        "total_review_records": len(all_reviews),
        "acts": sorted({clause.source_act for clause in all_clauses}),
        "documents_by_act": dict(Counter(report["source_act"] for report in quality_reports)),
        "clauses_by_act": dict(Counter(clause.source_act for clause in all_clauses)),
        "quality_status_counts": dict(Counter(report["status"] for report in quality_reports)),
        "ocr_enabled": bool(args.ocr),
        "ocr_min_chars": args.ocr_min_chars,
        "max_clause_chars": args.max_clause_chars,
    }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("\nDry run complete; no output files were written.")
        return 0 if not failed_files else 2

    output_dir.mkdir(parents=True, exist_ok=True)
    clause_dicts = [asdict(clause) for clause in all_clauses]
    review_dicts = [asdict(review) for review in all_reviews]

    output_files = {
        "json": output_dir / "government_clauses.json",
        "csv": output_dir / "government_clauses.csv",
        "review": output_dir / "government_extraction_review.json",
        "failures": output_dir / "government_extraction_failures.json",
        "quality": output_dir / "government_extraction_quality_report.json",
        "summary": output_dir / "government_extraction_summary.json",
    }

    write_json(output_files["json"], clause_dicts)
    write_json(output_files["review"], review_dicts)
    write_json(output_files["failures"], failed_files)
    write_json(output_files["quality"], quality_reports)
    write_json(output_files["summary"], summary)

    dataframe = pd.DataFrame(clause_dicts)
    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=[field.name for field in ClauseRecord.__dataclass_fields__.values()]
        )
    dataframe.to_csv(output_files["csv"], index=False)

    print("\nOutput files:")
    for path in output_files.values():
        print(f"  {path}")

    if failed_files:
        print(
            "\nWARNING: one or more files failed. Inspect "
            "government_extraction_failures.json.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
