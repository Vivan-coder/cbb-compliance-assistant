"""Shared paths and source metadata for the CBB compliance assistant."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "data" / "chroma"
EVAL_DIR = ROOT / "eval"

PAGES_FILE = PROCESSED_DIR / "pages.jsonl"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

RETRIEVAL_DATE = "2026-11-05"
BASE_URL = "https://cbben.thomsonreuters.com/sites/default/files/net_file_store/"

# 4 PDFs covering 3 modules (CA is published in two parts).
SOURCES = {
    "Vol_1_FC_July_2022.pdf": {
        "module": "FC", "module_name": "Financial Crime",
        "edition": "July 2022",
    },
    "Vol_1_HC_April_2023.pdf": {
        "module": "HC", "module_name": "High-level Controls (Corporate Governance)",
        "edition": "April 2023",
    },
    "Vol_1_CA_Part_1__July_2019.pdf": {
        "module": "CA", "module_name": "Capital Adequacy (Part 1: Definition of Capital)",
        "edition": "July 2019",
    },
    "Vol_1_CA_Part_2__July_2019.pdf": {
        "module": "CA", "module_name": "Capital Adequacy (Part 2: Credit Risk)",
        "edition": "July 2019",
    },
}

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with this instruction prefix on the *query* side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 8
