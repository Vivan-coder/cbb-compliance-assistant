"""Step 8 - Streamlit demo: CBB Rulebook Compliance Assistant."""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from answer import REFUSAL, answer_question  # noqa: E402
from config import RETRIEVAL_DATE, SOURCES, TOP_K  # noqa: E402

st.set_page_config(page_title="CBB Rulebook Assistant", layout="wide")

CACHE_FILE = Path(__file__).parent / "cached_examples.json"
EXAMPLES = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else []


@st.cache_resource(show_spinner="Loading embedding model and building the index (first run only)...")
def load_retriever():
    from index import build_index
    from retrieve import Retriever
    build_index()
    return Retriever()


def show_result(question, status, text, chunks, cached=False):
    if status == "answered":
        st.success("Answer (every citation validated against the retrieved rules)")
    elif status.startswith("refused"):
        st.warning("The assistant declined to answer - this is intended behaviour when the Rulebook text it has doesn't support an answer.")
    else:
        st.error("Answer withheld: it failed citation validation.")
    st.markdown(text)
    if cached:
        st.caption("Pre-generated example answer (no API call was made).")
    with st.expander(f"Retrieved Rulebook text ({len(chunks)} excerpts)", expanded=False):
        for c in chunks:
            cited = f"[{c['rule_id']}]" in text
            st.markdown(f"**{c['rule_id']}** - {c['heading']} ({c['module']}, last changed {c['last_changed']})"
                        + ("  [cited]" if cited else ""))
            st.text(c["text"])
            st.caption(f"Source: {c['source_url']}")
            st.divider()


# ------------------------------------------------------------------ layout
st.title("CBB Rulebook Compliance Assistant")
st.error("**Not legal or compliance advice.** This is a student portfolio project. Answers come only from a small "
         "extract of the CBB Rulebook (Volume 1: modules FC, HC and CA) and may be incomplete or out of date. "
         "Always check the official Rulebook.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Anthropic API key (optional)", type="password",
                            help="Used only in your browser session to call Claude. Not stored. Without a key you can still use the 5 example questions.")
    mode = st.selectbox("Retrieval", ["hybrid", "vector", "bm25"], index=0,
                        help="hybrid = vector search + BM25 keyword search, merged by reciprocal rank fusion")
    k = st.slider("Rules retrieved (top-k)", 3, 12, TOP_K)
    st.markdown("---")
    st.markdown("**Coverage**\n" + "\n".join(f"- {v['module']} - {v['module_name']} ({v['edition']})" for v in SOURCES.values()))
    st.caption(f"Source: CBB Rulebook PDFs, retrieved {RETRIEVAL_DATE}.")

st.subheader("Try an example (works without a key)")
cols = st.columns(len(EXAMPLES) or 1)
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(ex["question"], key=f"ex{i}", use_container_width=True):
        st.session_state["example"] = i
        st.session_state.pop("live", None)

st.subheader("Ask a question")
q = st.text_input("Your question about FC (AML/CFT), HC (governance) or CA (capital adequacy)",
                  placeholder="e.g. How long must a bank keep suspicious transaction report records?")
if st.button("Ask", type="primary") and q.strip():
    st.session_state.pop("example", None)
    retriever = load_retriever()
    chunks = retriever.search(q, mode, k)
    try:
        key = api_key or None
        a = answer_question(q, chunks, api_key=key) if (key or __import__("os").getenv("ANTHROPIC_API_KEY")) \
            else None
        if a is None:
            best = max((c.get("vector_score", 0) for c in chunks), default=0)
            st.info("No API key entered, so no answer was generated. Showing the Rulebook text that retrieval found "
                    f"(best similarity {best:.2f}). Add a key in the sidebar to get a cited answer, or try an example above.")
            st.session_state["live"] = (q, "no_key", "", chunks)
        else:
            st.session_state["live"] = (q, a.status, a.text, chunks)
    except Exception as e:  # bad key, network, etc.
        st.error(f"Could not generate an answer: {e}")

if "example" in st.session_state:
    ex = EXAMPLES[st.session_state["example"]]
    st.markdown(f"**Q: {ex['question']}**")
    show_result(ex["question"], ex["status"], ex["answer"], ex["chunks"], cached=True)
elif "live" in st.session_state:
    lq, status, text, chunks = st.session_state["live"]
    st.markdown(f"**Q: {lq}**")
    if status == "no_key":
        with st.expander("Retrieved Rulebook text", expanded=True):
            for c in chunks:
                st.markdown(f"**{c['rule_id']}** - {c['heading']} ({c['module']})")
                st.text(c["text"])
    else:
        show_result(lq, status, text, chunks)
