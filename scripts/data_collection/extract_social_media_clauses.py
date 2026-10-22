"""
Robust clause extraction for social-media policy documents.

This script:
- Recursively discovers PDF and optional DOCX files.
- Extracts PDF text page by page.
- Uses OCR for pages with little or no embedded text when --ocr is enabled.
- Removes repeated headers/footers conservatively.
- Does not delete valid clauses merely because they are short.
- Preserves document, page, section, extraction-method, and hash metadata.
- Splits text into reviewable clause-like records without arbitrary 300-character cuts.
- Writes clauses, failed files, removed blocks, and quality reports.

Expected input structure:

    datasets/SocialMediaPolicies/WhatsApp/source/*.pdf
    datasets/SocialMediaPolicies/Meta/source/*.pdf
    datasets/SocialMediaPolicies/X/source/*.pdf

The script also supports files directly inside the platform folder.

Install the minimum dependencies:

    pip install pymupdf pandas

Optional OCR dependencies:

    pip install pillow pytesseract
    sudo apt-get install tesseract-ocr

Run from the project root:

    python src/extraction/extract_social_media_clauses.py --ocr

For a dry run that does not write output files:

    python src/extraction/extract_social_media_clauses.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import pymupdf  # PyMuPDF newer import name
except ImportError:  # pragma: no cover - compatibility with older installations
    import fitz as pymupdf  # type: ignore

try:
    from docx import Document  # Optional DOCX support
except ImportError:  # pragma: no cover
    Document = None  # type: ignore

try:
    from PIL import Image  # Optional OCR dependency
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    import pytesseract  # Optional OCR dependency
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore


DEFAULT_SOURCE_ROOT = Path("datasets/SocialMediaPolicies")
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_ROOT

# These patterns match complete short navigation/header blocks only.
# They are intentionally anchored so that a valid legal paragraph containing
# words such as "Key Updates" is not deleted.
EXACT_JUNK_PATTERNS = [
    r"^back\s+to\s+top$",
    r"^legal\s+info$",
    r"^table\s+of\s+contents$",
    r"^contents$",
    r"^page\s+\d+(?:\s+of\s+\d+)?$",
    r"^privacy\s+policy$",
    r"^terms\s+of\s+service$",
    r"^terms\s+and\s+conditions$",
    r"^key\s+updates$",
    r"^archived\s+versions?$",
    r"^last\s+updated(?:\s*[:\-].*)?$",
    r"^effective\s+\w+\s+\d{1,2},?\s+\d{4}$",
]
EXACT_JUNK_REGEX = [re.compile(pattern, re.IGNORECASE) for pattern in EXACT_JUNK_PATTERNS]

# A block this short can still be a valid clause. This threshold is only used
# to identify exceptionally tiny extraction fragments for review, not to delete
# them automatically.
MIN_REVIEWABLE_CHARS = 10

# Repeated short lines are likely headers/footers. They are removed only when
# they occur on several pages and across a substantial fraction of the file.
REPEATED_LINE_MIN_OCCURRENCES = 3
REPEATED_LINE_MAX_CHARS = 140
REPEATED_LINE_PAGE_RATIO = 0.60

# Maximum clause size for model input. Splitting occurs only at sentence or list
# boundaries. If no safe boundary is found, the original text is retained.
MAX_CLAUSE_CHARS = 1400

BULLET_RE = re.compile(r"^(?:[-*•▪◦‣]|\(?\d+[.)]|\(?[a-zA-Z][.)])\s+")
HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+)?[A-Z][A-Za-z0-9 &'()/,:—–-]{2,100}$"
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])" )


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
    app_name: str
    source_document: str
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
    app_name: str
    source_document: str
    source_file: str
    page_number: int | None
    text: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract auditable clauses from social-media policy files.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing one folder per platform.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated JSON, CSV, and reports.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use OCR on pages with little or no embedded text.",
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
    """Normalize extracted text while preserving meaningful punctuation and words."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words broken by a line-ending hyphen, but keep ordinary hyphenated
    # words such as "data-processing" intact.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    # Normalize horizontal whitespace while retaining line boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_exact_junk(text: str) -> bool:
    candidate = normalize_line(text)
    if not candidate:
        return True
    return any(pattern.fullmatch(candidate) for pattern in EXACT_JUNK_REGEX)


def looks_like_heading(text: str) -> bool:
    candidate = normalize_line(text)
    if not candidate or len(candidate) > 140:
        return False
    if candidate.endswith((".", ";", ":", "?", "!")):
        return False
    return bool(HEADING_RE.fullmatch(candidate))


def extract_pdf_pages(
    pdf_path: Path,
    use_ocr: bool,
    ocr_min_chars: int,
) -> tuple[list[PageRecord], list[ReviewRecord]]:
    pages: list[PageRecord] = []
    review_records: list[ReviewRecord] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            embedded_text = page.get_text("text") or ""
            embedded_text = normalize_text(embedded_text)
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
                    except Exception as exc:  # OCR failure should not stop all files
                        warning = f"OCR failed: {type(exc).__name__}: {exc}"

            if not text:
                warning = warning or "No text extracted; page may be image-only or corrupt"

            if needs_ocr or warning:
                review_records.append(
                    ReviewRecord(
                        app_name="",
                        source_document=pdf_path.stem,
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

    return pages, review_records


def extract_docx_pages(docx_path: Path) -> tuple[list[PageRecord], list[ReviewRecord]]:
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
    return [PageRecord(1, text, "python-docx", len(text), False, None)], []


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

    threshold = max(
        REPEATED_LINE_MIN_OCCURRENCES,
        int(len(page_list) * REPEATED_LINE_PAGE_RATIO),
    )
    return {
        line
        for line, page_count in line_page_counts.items()
        if page_count >= threshold
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


def block_is_removable(block: str) -> tuple[bool, str | None]:
    candidate = normalize_line(block)
    if not candidate:
        return True, "empty block"
    if is_exact_junk(candidate):
        return True, "recognized navigation/header/footer block"
    return False, None


def split_long_block(text: str, max_chars: int) -> list[str]:
    """Split at sentence/list boundaries, never at an arbitrary character index."""
    text = normalize_line(text)
    if len(text) <= max_chars:
        return [text]

    # Keep list items as separate units where possible.
    list_lines = [normalize_line(line) for line in text.splitlines() if normalize_line(line)]
    if len(list_lines) > 1 and any(BULLET_RE.match(line) for line in list_lines):
        return list_lines

    sentences = SENTENCE_BOUNDARY_RE.split(text)
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
    """Create conservative text blocks while preserving list items and headings."""
    lines = [line.rstrip() for line in page_text.splitlines()]
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            block = normalize_text("\n".join(current))
            if block:
                blocks.append(block)
            current = []

    for raw_line in lines:
        line = normalize_line(raw_line)
        if not line:
            flush()
            continue

        # A new list item starts a new block.
        if BULLET_RE.match(line) and current:
            flush()

        # A likely heading starts a new block. It is retained as context later.
        if looks_like_heading(line) and current:
            flush()

        current.append(line)

    flush()
    return blocks


def build_clauses_for_document(
    app_name: str,
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
    removed_count = 0
    block_count = 0
    tiny_count = 0

    for page in pages:
        cleaned_page_text = remove_repeated_lines(page.text, repeated_lines)
        blocks = page_to_blocks(cleaned_page_text)
        block_count += len(blocks)

        for block in blocks:
            removable, reason = block_is_removable(block)
            if removable:
                removed_count += 1
                reviews.append(
                    ReviewRecord(
                        app_name=app_name,
                        source_document=file_path.stem,
                        source_file=file_path.name,
                        page_number=page.page_number,
                        text=block[:2000],
                        reason=reason or "removed block",
                    )
                )
                continue

            candidate = normalize_line(block)
            if looks_like_heading(candidate):
                section_heading = candidate
                # Keep headings as context but do not treat a standalone heading
                # as a legal clause.
                continue

            if len(candidate) < MIN_REVIEWABLE_CHARS:
                tiny_count += 1
                reviews.append(
                    ReviewRecord(
                        app_name=app_name,
                        source_document=file_path.stem,
                        source_file=file_path.name,
                        page_number=page.page_number,
                        text=candidate,
                        reason="Very short extracted block; retained only for manual review",
                    )
                )

            for piece in split_long_block(candidate, max_clause_chars):
                piece = normalize_line(piece)
                if not piece:
                    continue
                clause_index += 1
                extraction_quality = "review" if page.warning or len(piece) < MIN_REVIEWABLE_CHARS else "pass"
                clauses.append(
                    ClauseRecord(
                        clause_id="",  # Assigned globally after all files are processed.
                        app_name=app_name,
                        source_document=file_path.stem,
                        source_file=file_path.name,
                        source_path=str(file_path),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_heading=section_heading,
                        clause_index_in_document=clause_index,
                        clause_text=piece,
                        extraction_method=page.extraction_method,
                        extraction_quality=extraction_quality,
                        retrieved_at=retrieved_at,
                        file_sha256=file_hash,
                    )
                )

    quality = {
        "source_file": file_path.name,
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


def discover_platform_folders(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    return sorted(path for path in source_root.iterdir() if path.is_dir())


def discover_files(platform_folder: Path) -> list[tuple[Path, str]]:
    source_dir = platform_folder / "source"
    if not source_dir.exists():
        source_dir = platform_folder

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
    source_root: Path = args.source_root
    output_dir: Path = args.output_dir
    retrieved_at = utc_now_iso()

    if not source_root.exists():
        print(f"ERROR: source folder not found: {source_root}", file=sys.stderr)
        return 1

    platform_folders = discover_platform_folders(source_root)
    if not platform_folders:
        print(f"ERROR: no platform folders found under {source_root}", file=sys.stderr)
        return 1

    all_clauses: list[ClauseRecord] = []
    all_reviews: list[ReviewRecord] = []
    failed_files: list[dict[str, str]] = []
    quality_reports: list[dict[str, Any]] = []
    discovered_files_total = 0

    print(f"Found {len(platform_folders)} platform folder(s): {[p.name for p in platform_folders]}")

    for platform_folder in platform_folders:
        app_name = platform_folder.name
        files = discover_files(platform_folder)
        discovered_files_total += len(files)
        print(f"\n=== {app_name}: {len(files)} document(s) ===")

        if not files:
            failed_files.append({
                "app_name": app_name,
                "source_file": "",
                "reason": "No PDF or DOCX files found",
            })
            continue

        for file_path, file_type in files:
            print(f"  extracting: {file_path}")
            file_hash = sha256_file(file_path)

            try:
                if file_type == "pdf":
                    pages, page_reviews = extract_pdf_pages(
                        file_path,
                        use_ocr=args.ocr,
                        ocr_min_chars=args.ocr_min_chars,
                    )
                else:
                    pages, page_reviews = extract_docx_pages(file_path)

                for review in page_reviews:
                    review.app_name = app_name
                all_reviews.extend(page_reviews)

                clauses, reviews, quality = build_clauses_for_document(
                    app_name=app_name,
                    file_path=file_path,
                    pages=pages,
                    retrieved_at=retrieved_at,
                    file_hash=file_hash,
                    max_clause_chars=args.max_clause_chars,
                )
                all_reviews.extend(reviews)
                quality["app_name"] = app_name
                quality["file_hash"] = file_hash
                quality_reports.append(quality)
                all_clauses.extend(clauses)

                print(
                    f"    pages={quality['pages']} "
                    f"chars={quality['characters_extracted']} "
                    f"clauses={quality['retained_clauses']} "
                    f"status={quality['status']}"
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED: {reason}", file=sys.stderr)
                failed_files.append({
                    "app_name": app_name,
                    "source_file": file_path.name,
                    "reason": reason,
                })

    # Assign stable global IDs after processing all files.
    for number, clause in enumerate(all_clauses, start=1):
        clause.clause_id = f"S{number:06d}"

    summary = {
        "run_timestamp_utc": retrieved_at,
        "source_root": str(source_root),
        "discovered_files_total": discovered_files_total,
        "platform_count": len(platform_folders),
        "total_clauses": len(all_clauses),
        "total_failed_files": len(failed_files),
        "total_review_records": len(all_reviews),
        "documents_by_platform": dict(
            Counter(report.get("app_name", "") for report in quality_reports)
        ),
        "clauses_by_platform": dict(
            Counter(clause.app_name for clause in all_clauses)
        ),
        "quality_status_counts": dict(
            Counter(report.get("status", "UNKNOWN") for report in quality_reports)
        ),
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

    clauses_json = output_dir / "social_media_clauses.json"
    clauses_csv = output_dir / "social_media_clauses.csv"
    reviews_json = output_dir / "extraction_review.json"
    failures_json = output_dir / "extraction_failures.json"
    quality_json = output_dir / "extraction_quality_report.json"
    summary_json = output_dir / "extraction_summary.json"

    write_json(clauses_json, clause_dicts)
    write_json(reviews_json, review_dicts)
    write_json(failures_json, failed_files)
    write_json(quality_json, quality_reports)
    write_json(summary_json, summary)

    dataframe = pd.DataFrame(clause_dicts)
    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=[field.name for field in ClauseRecord.__dataclass_fields__.values()]
        )
    dataframe.to_csv(clauses_csv, index=False)

    print("\nOutput files:")
    for path in [clauses_json, clauses_csv, reviews_json, failures_json, quality_json, summary_json]:
        print(f"  {path}")

    if failed_files:
        print("\nWARNING: one or more files failed. Inspect extraction_failures.json.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
