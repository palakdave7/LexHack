"""GoodLaw end-to-end analyzer.

Runs all four layers on a document and produces one unified report:
  - Layer 1: existence + name match (verify.py)
  - Layer 2: quote fidelity (quotes.py)
  - Layer 3: proposition support via NLI (claims.py)
  - Layer 4: negative treatment signal (treatment.py)

Output: out/report.json + human-readable summary to stdout.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from dotenv import load_dotenv

from goodlaw.claims import check_claims
from goodlaw.fetch import fetch_all_for_document
from goodlaw.quotes import attribute_quotes, check_quotes, extract_quotes
from goodlaw.treatment import check_treatment
from goodlaw.verify import verify


# Overall document verdict: worst-case across all layers.
_SEVERITY = {"red": 3, "yellow": 2, "gray": 1, "green": 0}


@dataclass
class Report:
    document_path: str
    document_char_count: int
    overall_verdict: str
    summary_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    citations: list[dict[str, Any]] = field(default_factory=list)
    quotes: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    treatment: list[dict[str, Any]] = field(default_factory=list)
    runtime_seconds: float = 0.0


def _worst(verdicts: list[str]) -> str:
    if not verdicts:
        return "gray"
    return max(verdicts, key=lambda v: _SEVERITY.get(v, 0))


def _counts(items: list[Any], key: str = "verdict") -> dict[str, int]:
    c = {"green": 0, "yellow": 0, "red": 0, "gray": 0}
    for it in items:
        v = it.get(key) if isinstance(it, dict) else getattr(it, key, None)
        if v in c:
            c[v] += 1
    return c


def analyze(raw: str, token: str, doc_path: str = "input") -> Report:
    t0 = time.time()

    # Layer 1
    print("Layer 1: verifying citations...")
    vcs = verify(raw, token)

    # Fetch opinions (shared cache between layers 2 and 3)
    print("Fetching cited opinions...")
    opinions, cluster_map = fetch_all_for_document(raw, token)

    # Layer 2
    print("Layer 2: checking quote fidelity...")
    quotes = extract_quotes(raw)
    attribute_quotes(quotes, vcs)
    check_quotes(quotes, opinions, cluster_map)

    # Layer 3
    print("Layer 3: checking proposition support (NLI)...")
    claims = check_claims(raw, vcs, opinions, cluster_map)

    # Layer 4
    print("Layer 4: scanning for negative treatment...")
    treatment = check_treatment(vcs, token)

    # Aggregate
    citations_out = [asdict(vc) for vc in vcs]
    quotes_out = [asdict(q) for q in quotes]
    claims_out = [asdict(c) for c in claims]
    treatment_out = [asdict(t) for t in treatment]

    overall = _worst(
        [
            _worst([c["verdict"] for c in citations_out]),
            _worst([q["verdict"] for q in quotes_out]),
            _worst([c["verdict"] for c in claims_out]),
            _worst([t["verdict"] for t in treatment_out]),
        ]
    )

    return Report(
        document_path=doc_path,
        document_char_count=len(raw),
        overall_verdict=overall,
        summary_counts={
            "citations": _counts(citations_out),
            "quotes": _counts(quotes_out),
            "claims": _counts(claims_out),
            "treatment": _counts(treatment_out),
        },
        citations=citations_out,
        quotes=quotes_out,
        claims=claims_out,
        treatment=treatment_out,
        runtime_seconds=round(time.time() - t0, 1),
    )


def _emoji(v: str) -> str:
    return {"green": "[OK]", "yellow": "[??]", "red": "[XX]", "gray": "[--]"}.get(
        v, "[  ]"
    )


def _print_summary(report: Report) -> None:
    print("\n" + "=" * 60)
    print(
        f"OVERALL VERDICT: {_emoji(report.overall_verdict).strip('[]')} "
        f"{report.overall_verdict.upper()}"
    )
    print("=" * 60)
    print(f"\nDocument: {report.document_path}")
    print(f"Length:   {report.document_char_count:,} chars")
    print(f"Runtime:  {report.runtime_seconds}s\n")

    for layer_name, counts in report.summary_counts.items():
        line = f"  {layer_name:<10}"
        for v in ("green", "yellow", "red", "gray"):
            line += f"  {_emoji(v)}{counts[v]:>3}"
        print(line)

    # Red-flag detail
    print("\nRED FLAGS:")
    any_red = False
    for c in report.citations:
        if c["verdict"] == "red":
            any_red = True
            print(f"  citation {c['text']!r}: {c['status']}")
    for q in report.quotes:
        if q["verdict"] == "red":
            any_red = True
            preview = q["quote"][:70]
            print(f"  quote \"{preview}...\": {q['reason']}")
    for c in report.claims:
        if c["verdict"] == "red":
            any_red = True
            print(f"  claim on {c['citation_text']!r}: {c['reason']}")
    for t in report.treatment:
        if t["verdict"] == "red":
            any_red = True
            print(f"  treatment on {t['citation_text']!r}: {t['reason']}")
    if not any_red:
        print("  (none)")


def main() -> None:
    load_dotenv()
    token = os.getenv("COURTLISTENER_TOKEN")
    if not token:
        sys.exit("COURTLISTENER_TOKEN missing in .env")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brief.txt"
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    report = analyze(raw, token, doc_path=path)
    _print_summary(report)

    os.makedirs("out", exist_ok=True)
    with open("out/report.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print("\nWrote out/report.json")


if __name__ == "__main__":
    main()
