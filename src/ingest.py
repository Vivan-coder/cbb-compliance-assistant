"""Step 2 - Ingest: PDF -> clean per-page text with structural metadata.

Every rule page in the CBB module PDFs starts with a boilerplate header
(bank name, volume, module/date line, 'Section X: Page n of m', MODULE and
CHAPTER lines). We use that header to learn *where we are* in the Rulebook
(module, chapter, section, last-changed date) and then strip it so it never
pollutes the embeddings.  Cover pages and the table of contents have no
'Section ...: Page' line and are dropped.
"""
import json
import re

from pypdf import PdfReader

from config import BASE_URL, PAGES_FILE, PROCESSED_DIR, RAW_DIR, RETRIEVAL_DATE, SOURCES

SECTION_RE = re.compile(r"Section\s+([A-Z]+-[A-Z0-9.]+)\s*:\s*Page\s+(\d+)\s+of\s+(\d+)")
CHAPTER_RE = re.compile(r"CHAPTER\s+([A-Z]+-[A-Z0-9]+)\s*:\s*(.+)")
DATE_RE = re.compile(
    r"^[A-Z]{2}:\s.*?\s{2,}((?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4})\s*$"
)


def normalise(text: str) -> str:
    """Fix PDF extraction artefacts without changing the wording."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # 'CA -1.1.3' / 'FC -C.2' -> 'CA-1.1.3'
    text = re.sub(r"\b(FC|HC|CA|UG|OM|CM|BC|AU|PD|PB)\s+-\s*(?=[A-Z0-9])", r"\1-", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def parse_page(raw: str):
    """Return (meta, body) for a rule page, or None for cover/TOC pages."""
    lines = [l.rstrip() for l in raw.splitlines()]
    sec = None
    date = None
    chapter = chapter_title = None
    body_start = None
    for i, line in enumerate(lines[:20]):
        if sec is None:
            m = SECTION_RE.search(line)
            if m:
                sec = (m.group(1), int(m.group(2)), int(m.group(3)))
        m = DATE_RE.match(line.strip()) if line.strip() else None
        if m and date is None:
            date = m.group(1)
        m = CHAPTER_RE.search(line)
        if m and chapter is None:
            chapter, chapter_title = m.group(1), m.group(2).strip()
            body_start = i + 1
            break
    if sec is None or chapter is None:
        return None
    body = normalise("\n".join(lines[body_start:])).strip()
    meta = {
        "section": sec[0], "page_in_section": sec[1], "pages_in_section": sec[2],
        "chapter": chapter, "chapter_title": chapter_title,
        "last_changed": date,
    }
    return meta, body


def ingest():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    n_pages = 0
    with open(PAGES_FILE, "w", encoding="utf-8") as out:
        for fname, src in SOURCES.items():
            path = RAW_DIR / fname
            if not path.exists():
                print(f"  MISSING {path} (see README: data sourcing)")
                continue
            reader = PdfReader(str(path))
            kept = 0
            for pno, page in enumerate(reader.pages, start=1):
                parsed = parse_page(page.extract_text() or "")
                if not parsed:
                    continue
                meta, body = parsed
                rec = {
                    **meta, "body": body, "pdf_page": pno,
                    "module": src["module"], "module_name": src["module_name"],
                    "edition": src["edition"], "source_file": fname,
                    "source_url": BASE_URL + fname, "retrieval_date": RETRIEVAL_DATE,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            n_pages += kept
            print(f"  {fname}: {kept}/{len(reader.pages)} rule pages kept")
    print(f"Wrote {n_pages} pages -> {PAGES_FILE}")


if __name__ == "__main__":
    ingest()
