"""Step 6 - Answer with Claude: citation-forced prompt, refusal logic, code-level validation.

Safety layers (a wrong compliance answer is worse than no answer):
  1. Pre-LLM gate: if the best retrieved chunk is only weakly similar, refuse without calling the model.
  2. Prompt: answer only from the excerpts, cite [RULE-ID] inline, refuse with a fixed sentence.
  3. Post-LLM validation in code: every cited ID must be one of the retrieved rule IDs, and a
     non-refusal answer must cite at least one rule. Otherwise the answer is rejected.
"""
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

REFUSAL = "The Rulebook excerpts I have don't cover this."
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
# Minimum cosine similarity (bge-small, query prefix) of the best chunk to even try answering.
# Calibrated on the evaluation set; see eval/questions.csv and README.
MIN_TOP_SIMILARITY = float(os.getenv("MIN_TOP_SIMILARITY", "0.50"))

CITE_RE = re.compile(r"\[([A-Z]{2}-[A-Z0-9]+(?:\.\d+)+[A-Za-z]{0,2})\]")

SYSTEM_PROMPT = f"""You are a compliance research assistant for the Central Bank of Bahrain (CBB) Rulebook, Volume 1 (Conventional Bank licensees).

You will receive numbered RULE EXCERPTS and a QUESTION. Follow these requirements strictly:
1. Answer ONLY from the excerpts provided. Never use outside knowledge, other jurisdictions' rules, or assumptions.
2. Cite the rule ID in square brackets inline after every claim, exactly as it appears in the excerpt header, e.g. [FC-C.2.1]. Cite only IDs that appear in the excerpts.
3. Never invent, guess or "complete" a rule number. Do not cite a rule you were not given.
4. If the excerpts do not contain enough to answer the question, or only partly answer it, reply with exactly this sentence and nothing else if nothing relevant is covered: "{REFUSAL}" If the excerpts answer only part of the question, answer that part with citations and state clearly which part is not covered.
5. Be concise and precise. Preserve the obligation level of the source ('must', 'should', 'may'). Do not give legal advice or recommendations beyond what the rules say."""


@dataclass
class Answer:
    text: str
    status: str  # 'answered' | 'refused' | 'refused_weak_retrieval' | 'rejected_invalid_citation'
    citations: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)
    raw: str = ""
    model: str = ""


def format_context(chunks):
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['rule_id']}] (Module {c['module']}, {c['heading']}; last changed {c['last_changed']})\n{c['text']}")
    return "\n\n".join(blocks)


def validate(text: str, chunks):
    """Return (citations, invalid) where invalid are cited IDs not among the retrieved rule IDs."""
    allowed = {c["rule_id"] for c in chunks}
    cited = list(dict.fromkeys(CITE_RE.findall(text)))
    return cited, [c for c in cited if c not in allowed]


def is_refusal(text: str) -> bool:
    return text.strip().startswith(REFUSAL[:40])


def answer_question(question: str, chunks, api_key: str | None = None, client=None) -> Answer:
    # Layer 1: weak retrieval -> refuse without spending an API call.
    best = max((c.get("vector_score", 0) for c in chunks), default=0)
    if not chunks or best < MIN_TOP_SIMILARITY:
        return Answer(REFUSAL, "refused_weak_retrieval")

    if client is None:
        import anthropic
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("No API key. Set ANTHROPIC_API_KEY in .env or enter one in the app sidebar.")
        client = anthropic.Anthropic(api_key=key)

    user = f"RULE EXCERPTS:\n\n{format_context(chunks)}\n\nQUESTION: {question}"
    messages = [{"role": "user", "content": user}]
    text = ""
    for attempt in range(2):
        resp = client.messages.create(model=MODEL, max_tokens=700,
                                      system=SYSTEM_PROMPT, messages=messages)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if is_refusal(text):
            return Answer(REFUSAL if text.strip() == REFUSAL else text, "refused", raw=text, model=MODEL)
        cited, invalid = validate(text, chunks)
        if cited and not invalid:
            return Answer(text, "answered", citations=cited, raw=text, model=MODEL)
        # Layer 3: one corrective retry, then reject.
        problem = (f"These cited IDs are not in the excerpts: {invalid}." if invalid
                   else "Your answer had no [RULE-ID] citations.")
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"{problem} Rewrite using only IDs from the excerpts, "
                                                           f"or reply with the exact refusal sentence."}]
    cited, invalid = validate(text, chunks)
    return Answer("Answer rejected: it did not pass citation validation, so it is withheld rather than risk a "
                  "misleading citation.", "rejected_invalid_citation", cited, invalid, raw=text, model=MODEL)
