"""Step 3 - Chunk by rule number (not by character count).

Each chunk = one numbered rule/guidance paragraph (e.g. FC-C.2.1) including its
(a)/(b)/(c) sub-items.  Pages of the same section are stitched together first so
rules that run across a page break stay whole.

A rule ID is only accepted at the start of a line AND if it belongs to the
current section (rule 'FC-C.2.7' must be inside Section 'FC-C.2').  That guards
against cross-references such as 'FC-1.1.3 to FC-1.1.5 apply...' starting a line.
Very long rules are split on paragraph boundaries into parts that share a rule_id.
"""
import json
import re
from collections import OrderedDict

from config import CHUNKS_FILE, PAGES_FILE

MAX_CHARS = 1800
RULE_START = re.compile(r"^([A-Z]{2}-[A-Z0-9]+(?:\.\d+)+[A-Za-z]{0,2})\s+(?=\S)", re.M)
CONTINUED = re.compile(r"\s*\(\s*continued\s*\)", re.I)


def load_sections():
    sections = OrderedDict()
    with open(PAGES_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            key = (p["source_file"], p["section"])
            sections.setdefault(key, []).append(p)
    return sections


def section_title(first_body: str, section_id: str) -> str:
    m = re.search(rf"^{re.escape(section_id)}\s+(.+)$", first_body, re.M)
    return CONTINUED.sub("", m.group(1)).strip() if m else ""


def split_long(text: str):
    """Split on paragraph/sub-item boundaries so each part <= MAX_CHARS."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for para in re.split(r"\n(?=\s*(?:\([a-z]+\)|\([ivx]+\)|\d+\.)\s)", text):
        if cur and len(cur) + len(para) > MAX_CHARS:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_section(pages):
    first = pages[0]
    sec = first["section"]
    # Drop repeated 'FC-C.2 Risk Assessment (continued)' lines on continuation pages.
    bodies = []
    for p in pages:
        b = p["body"]
        b = re.sub(rf"^{re.escape(sec)}\s+.*\(\s*continued\s*\).*$", "", b, flags=re.M | re.I)
        bodies.append(b)
    text = "\n".join(bodies)
    title = section_title(first["body"], sec)

    if re.search(r"evolution|change log|history", title, re.I):
        return []  # module change-history tables are not rules

    starts, seen_ids = [], set()
    for m in RULE_START.finditer(text):
        # must belong to this section; repeats of an ID stay inside the previous rule
        if m.group(1).startswith(sec + ".") and m.group(1) not in seen_ids:
            seen_ids.add(m.group(1))
            starts.append(m)
    chunks = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        rule_id = m.group(1)
        body = re.sub(r"\n{2,}", "\n", text[m.start():end]).strip()
        body = re.sub(rf"^{re.escape(rule_id)}\s+", "", body, count=1)
        if len(body) < 25 or re.search(r"\[?\s*This (?:Paragraph|Section|Chapter|Rule)[^\]]*?(?:deleted|removed)", body, re.I) and len(body) < 160:
            continue  # empty or 'deleted' stub
        for j, part in enumerate(split_long(body)):
            context = (f"Module {first['module']} ({first['module_name']}) | "
                       f"Chapter {first['chapter']} {first['chapter_title']} | "
                       f"Section {sec} {title} | Rule {rule_id}")
            chunks.append({
                "chunk_id": f"{rule_id}#{j}" if j else rule_id,
                "rule_id": rule_id,
                "part": j,
                "module": first["module"],
                "module_name": first["module_name"],
                "chapter": first["chapter"],
                "chapter_title": first["chapter_title"],
                "section": sec,
                "heading": title,
                "last_changed": first["last_changed"],
                "edition": first["edition"],
                "source_url": first["source_url"],
                "retrieval_date": first["retrieval_date"],
                "text": part,
                "embed_text": f"{context}\n{part}",
            })
    return chunks


def chunk_all():
    all_chunks, seen = [], {}
    for pages in load_sections().values():
        for c in chunk_section(pages):
            if c["chunk_id"] in seen:  # same ID twice in different files -> disambiguate
                n = seen[c["chunk_id"]] = seen[c["chunk_id"]] + 1
                c["chunk_id"] = f"{c['chunk_id']}@{n}"
            else:
                seen[c["chunk_id"]] = 0
            all_chunks.append(c)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    by_mod = {}
    for c in all_chunks:
        by_mod[c["module"]] = by_mod.get(c["module"], 0) + 1
    print(f"Wrote {len(all_chunks)} chunks -> {CHUNKS_FILE}  {by_mod}")
    lens = sorted(len(c["text"]) for c in all_chunks)
    print(f"chunk chars: min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")
    return all_chunks


if __name__ == "__main__":
    chunk_all()
