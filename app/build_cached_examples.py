"""Builds app/cached_examples.json: 5 pre-saved example answers so the demo works with no API key.

IMPORTANT (transparency): these answers were NOT produced by a live API call. They were written by hand
from the retrieved rule text, following the same rules as the system prompt (answer only from the excerpts,
cite [RULE-ID] inline, refuse when not covered). Every citation is still checked by answer.validate().
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from answer import REFUSAL, validate  # noqa: E402
from retrieve import Retriever  # noqa: E402

EXAMPLES = [
    ("Within how many days must a bank appoint a permanent replacement if the MLRO position falls vacant?",
     "A conventional bank licensee must appoint a permanent replacement MLRO, after obtaining CBB approval, within "
     "120 calendar days of the vacancy occurring [FC-4.1.9]. In the meantime it must make immediate interim "
     "arrangements, including appointing an acting MLRO, and those interim arrangements must be approved by the "
     "CBB [FC-4.1.9]."),
    ("How must suspicious transaction reports be submitted, and how long must the details be kept?",
     "Reports must be sent to the Financial Intelligence Directorate at the Ministry of Interior and to the CBB's "
     "Compliance Directorate through the Suspicious Transaction Report Online System; paper STRs are not accepted "
     "[FC-5.2.3]. Licensees must retain all relevant details of STRs submitted for at least five years [FC-5.2.5]."),
    ("What are the requirements for the size, independence and chair of the audit committee?",
     "The audit committee must have at least three directors, the majority of whom must be independent and free of "
     "conflicts of interest [HC-3.4.1]. Its chairperson must be independent, must not chair the Board (unless "
     "considered independent) and must not chair any other Board committee [HC-3.4.2]."),
    ("What total capital ratio must a bank hold once the capital conservation buffer is included?",
     "The minimum consolidated Total Capital ratio is 10% of risk-weighted assets, in addition to a minimum Capital "
     "Conservation Buffer of 2.5% [CA-2.2.1]. A bank is therefore required to maintain a consolidated Total Capital "
     "Ratio of 12.5% (and a solo ratio of 8%), and must retain 100% of annual net profit unless its consolidated "
     "ratio is above 12.5% and its solo ratio is above 8% [CA-2A.2.2]."),
    ("What is the minimum liquidity coverage ratio a bank must maintain?",
     REFUSAL),
]


def main():
    r = Retriever()
    out = []
    for q, text in EXAMPLES:
        chunks = r.search(q, "hybrid", 8)
        if text == REFUSAL:
            status = "refused"
            cited, invalid = [], []
        else:
            status = "answered"
            cited, invalid = validate(text, chunks)
            assert cited and not invalid, (q, cited, invalid)
        out.append({"question": q, "answer": text, "status": status, "citations": cited,
                    "chunks": [{k: c[k] for k in ("rule_id", "heading", "module", "last_changed", "source_url", "text")}
                               for c in chunks]})
        print(status, cited, "|", q[:60])
    Path(__file__).with_name("cached_examples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
