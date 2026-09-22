"""Layer 2 - quote fidelity check.

For every quoted span in the document, decide which citation it belongs to
(by proximity), then look for that exact-ish text in the cited opinion.
Uses rapidfuzz.partial_ratio: robust to whitespace, ellipses, and small
editorial differences between the quote as written and the source text.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from rapidfuzz import fuzz

from goodlaw.fetch import fetch_all_for_document
from goodlaw.verify import VerifiedCitation, verify

# Curly and straight double quotes, plus a few common variants. We accept
# any of them as opening or closing marks, and normalize when comparing.
_QUOTE_OPEN = '"\u201c\u201f\u2033'
_QUOTE_CLOSE = '"\u201d\u201f\u2033'
_QUOTE_PAT = re.compile(
    f"[{_QUOTE_OPEN}]([^{_QUOTE_OPEN}{_QUOTE_CLOSE}]{{15,}}?)[{_QUOTE_CLOSE}]",
    re.DOTALL,
)

# How far after a quote we'll look for a citation to attribute it to.
# Legal writing puts the cite in the same sentence, so ~250 chars is plenty.
ATTRIBUTION_WINDOW = 250

# Two independent tools produce offsets on the same text (our quote regex
# and eyecite). Where a citation sits immediately after a closing quote,
# they often overlap by 1-3 chars depending on how each grabs adjacent
# punctuation. Treat citations that start within this many chars *inside*
# the quote's end as "after" the quote.
ATTRIBUTION_OVERLAP_TOLERANCE = 5

# Score bands. partial_ratio returns 0-100.
GREEN_THRESHOLD = 90  # essentially verbatim
YELLOW_THRESHOLD = 75  # paraphrased or heavy edits - flag for review


@dataclass
class QuoteCheck:
    quote: str  # the quoted text as it appears in the doc
    doc_start: int  # char offset in the doc
    doc_end: int
    attributed_to: str | None = None  # citation text this quote belongs to
    cluster_id: int | None = None  # cluster of the cited case
    best_score: float = 0.0  # 0-100
    best_opinion_id: int | None = None  # which opinion in the cluster matched
    verdict: str = "unchecked"  # green | yellow | red | gray
    reason: str = ""  # human-readable one-liner


def _normalize(text: str) -> str:
    """Fold quote marks and collapse whitespace so tiny formatting differences
    don't tank the match score."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_quotes(raw: str) -> list[QuoteCheck]:
    """Find every quoted span of >=15 characters in the document."""
    out: list[QuoteCheck] = []
    for m in _QUOTE_PAT.finditer(raw):
        out.append(
            QuoteCheck(
                quote=m.group(1),
                doc_start=m.start(),
                doc_end=m.end(),
            )
        )
    return out


def attribute_quotes(
    quotes: list[QuoteCheck],
    verified_citations: list[VerifiedCitation],
) -> None:
    """Attach each quote to the nearest following citation within
    ATTRIBUTION_WINDOW chars. Mutates the QuoteCheck objects in place."""
    for q in quotes:
        best: tuple[int, VerifiedCitation] | None = None  # (distance, citation)
        for vc in verified_citations:
            # Skip citations that are genuinely before or inside the quote.
            # We allow a small overlap because our quote regex and eyecite
            # occasionally disagree by 1-3 chars on where a span ends/starts.
            if vc.start < q.doc_end - ATTRIBUTION_OVERLAP_TOLERANCE:
                continue
            # Distance is measured from the closing quote to the cite start.
            # Clamp negative distances (from the tolerance window) to 0 so
            # a cite that overlaps by 2 chars is treated as immediately after.
            distance = max(0, vc.start - q.doc_end)
            if distance > ATTRIBUTION_WINDOW:
                continue
            if best is None or distance < best[0]:
                best = (distance, vc)

        if best is None:
            q.verdict = "gray"
            q.reason = "no nearby citation to attribute quote to"
            continue

        _, cite = best
        q.attributed_to = cite.text
        q.cluster_id = cite.cluster_id

        # A quote pointing at a red citation is red by definition -
        # the source it names doesn't say what the doc claims because the
        # source doesn't exist as claimed.
        if cite.verdict == "red":
            q.verdict = "red"
            q.reason = f"attributed citation ({cite.text}) is {cite.status}"
        elif cite.cluster_id is None:
            q.verdict = "gray"
            q.reason = "attributed citation not resolvable to a case"


def check_quotes(
    quotes: list[QuoteCheck],
    opinions: dict[int, str],
    cluster_map: dict[int, list[int]],
) -> None:
    """For each quote with a resolved cluster, fuzzy-match against every
    opinion in that cluster and record the best score."""
    for q in quotes:
        if q.verdict in ("red", "gray"):
            continue  # already decided by attribution
        if q.cluster_id is None:
            q.verdict = "gray"
            q.reason = "no cluster to search"
            continue

        opinion_ids = cluster_map.get(q.cluster_id, [])
        if not opinion_ids:
            q.verdict = "gray"
            q.reason = "cluster has no fetched opinions"
            continue

        needle = _normalize(q.quote)
        best_score = 0.0
        best_oid: int | None = None
        for oid in opinion_ids:
            haystack = opinions.get(oid)
            if not haystack:
                continue
            haystack_norm = _normalize(haystack)
            # partial_ratio: best-matching substring of haystack vs needle.
            # Fast (Levenshtein on windowed substrings, C-accelerated).
            score = fuzz.partial_ratio(needle, haystack_norm)
            if score > best_score:
                best_score = score
                best_oid = oid

        q.best_score = best_score
        q.best_opinion_id = best_oid

        if best_score >= GREEN_THRESHOLD:
            q.verdict = "green"
            q.reason = f"verbatim match in opinion {best_oid} (score {best_score:.0f})"
        elif best_score >= YELLOW_THRESHOLD:
            q.verdict = "yellow"
            q.reason = (
                f"partial match in opinion {best_oid} (score {best_score:.0f}) "
                f"- paraphrase or misquote"
            )
        else:
            q.verdict = "red"
            q.reason = (
                f"no match in any opinion of cluster {q.cluster_id} "
                f"(best score {best_score:.0f}) - quote appears fabricated"
            )


def check_document(
    raw: str, token: str
) -> tuple[list[VerifiedCitation], list[QuoteCheck]]:
    """End-to-end: verify citations, fetch opinions, check quotes."""
    verified_citations = verify(raw, token)
    opinions, cluster_map = fetch_all_for_document(raw, token)

    quotes = extract_quotes(raw)
    attribute_quotes(quotes, verified_citations)
    check_quotes(quotes, opinions, cluster_map)

    return verified_citations, quotes


def _emoji(verdict: str) -> str:
    return {"green": "[OK]", "yellow": "[??]", "red": "[XX]", "gray": "[--]"}.get(
        verdict, "[  ]"
    )


def main() -> None:
    load_dotenv()
    token = os.getenv("COURTLISTENER_TOKEN")
    if not token:
        sys.exit("COURTLISTENER_TOKEN missing in .env")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brief.txt"
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    print(f"\nAnalyzing {path}\n")
    verified_citations, quotes = check_document(raw, token)

    print(f"\n--- QUOTE CHECK: {len(quotes)} quotes found ---\n")
    counts = {"green": 0, "yellow": 0, "red": 0, "gray": 0}
    for q in quotes:
        counts[q.verdict] = counts.get(q.verdict, 0) + 1
        preview = q.quote[:80] + ("..." if len(q.quote) > 80 else "")
        print(f'{_emoji(q.verdict)} "{preview}"')
        print(f"       attributed to: {q.attributed_to}")
        print(f"       {q.reason}")
        print()

    print(
        f"Quotes: green={counts['green']}  yellow={counts['yellow']}  "
        f"red={counts['red']}  gray={counts['gray']}"
    )

    os.makedirs("out", exist_ok=True)
    with open("out/quotes.json", "w", encoding="utf-8") as f:
        json.dump([asdict(q) for q in quotes], f, indent=2, default=str)
    print("\nWrote out/quotes.json")


if __name__ == "__main__":
    main()
