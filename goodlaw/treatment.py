"""Layer 4 - negative treatment signal.

For each verified case, use CourtListener's search endpoint to find later
opinions citing it, then scan the returned snippets for negative-treatment
language (overruled, abrogated, criticized, distinguished, etc.).

Uses snippets, not full opinion bodies: one HTTP call per cited case,
which stays well inside CourtListener's search rate limit.

NOT Shepardization - keyword-based signal, reported as such. Flags cases
worth checking against a paid citator before filing.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from goodlaw.verify import VerifiedCitation, verify

SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"

NEGATIVE_TERMS = {
    "red": [
        r"\boverrul(?:ed|ing|es)\b",
        r"\babrogat(?:ed|ing|es)\b",
        r"\bsupersed(?:ed|ing|es)\b",
        r"\breversed\b",
        r"\bvacated\b",
        r"\bno longer good law\b",
    ],
    "yellow": [
        r"\bcriticiz(?:ed|ing|es)\b",
        r"\bquestion(?:ed|ing|s)\b",
        r"\bdistinguish(?:ed|ing|es)\b",
        r"\bdeclin(?:ed|ing|es) to follow\b",
        r"\bcalled into (?:doubt|question)\b",
    ],
}

# Boilerplate phrases that contain treatment terms but are not treatment.
# Publisher slip-opinion headers, standard disposition language, etc.
BOILERPLATE_BLOCKLIST = [
    r"subject to formal revision",
    r"superseded by the advance sheets",
    r"superseded by the (?:final|official) (?:report|reports)",
    r"NOTICE:\s*All slip opinions",
]

# Proximity: the negative term must appear within this many chars of an
# actual citation mention in the same snippet.
PROXIMITY_WINDOW = 250
MAX_CITING_OPINIONS = 20


@dataclass
class TreatmentSignal:
    citation_text: str
    cluster_id: int | None
    citing_opinions_returned: int = 0
    citing_opinions_scanned: int = 0
    negative_hits: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "green"
    reason: str = ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def _search(client: httpx.Client, citation: str) -> list[dict[str, Any]]:
    r = client.get(
        SEARCH_URL,
        params={
            "type": "o",
            "q": f'"{citation}"',
            "order_by": "dateFiled desc",
        },
        timeout=30.0,
    )
    if r.status_code == 429:
        raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
    r.raise_for_status()
    return r.json().get("results", [])


def _result_cluster_id(result: dict[str, Any]) -> int | None:
    for key in ("cluster_id", "cluster", "id"):
        val = result.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
    return None


def _snippet_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    if result.get("snippet"):
        parts.append(str(result["snippet"]))
    for op in result.get("opinions", []) or []:
        for key in ("snippet", "text", "plain_text"):
            v = op.get(key)
            if v:
                parts.append(str(v))
    for key in ("caseName", "case_name", "syllabus"):
        v = result.get(key)
        if v:
            parts.append(str(v))
    return " ".join(parts)


def _is_boilerplate(context: str) -> bool:
    for pat in BOILERPLATE_BLOCKLIST:
        if re.search(pat, context, re.IGNORECASE):
            return True
    return False


def _scan_for_negative_treatment(
    snippet: str, citation_str: str
) -> list[dict[str, Any]]:
    """A negative term counts only when it sits within PROXIMITY_WINDOW
    chars of an actual mention of the cited citation string, and only
    when its surrounding context isn't publisher boilerplate."""
    hits: list[dict[str, Any]] = []

    # Locate citation mentions
    cite_pattern = re.escape(citation_str).replace(r"\ ", r"\s+")
    cite_matches = [m.span() for m in re.finditer(cite_pattern, snippet, re.IGNORECASE)]
    if not cite_matches:
        return hits  # snippet matched via full-text but doesn't repeat the cite

    for severity, patterns in NEGATIVE_TERMS.items():
        for pat in patterns:
            for tm in re.finditer(pat, snippet, re.IGNORECASE):
                # Nearest citation mention distance
                term_mid = (tm.start() + tm.end()) // 2
                min_dist = min(
                    abs(term_mid - ((cs + ce) // 2)) for cs, ce in cite_matches
                )
                if min_dist > PROXIMITY_WINDOW:
                    continue

                ctx_start = max(0, tm.start() - 120)
                ctx_end = min(len(snippet), tm.end() + 120)
                context = snippet[ctx_start:ctx_end].strip()

                if _is_boilerplate(context):
                    continue

                hits.append(
                    {
                        "severity": severity,
                        "term": tm.group(),
                        "distance_to_cite": min_dist,
                        "context": context,
                    }
                )
    return hits


def check_treatment(
    verified_citations: list[VerifiedCitation], token: str
) -> list[TreatmentSignal]:
    results: list[TreatmentSignal] = []
    headers = {"Authorization": f"Token {token}", "User-Agent": "GoodLaw/0.1"}

    seen_clusters: set[int] = set()
    unique_targets: list[VerifiedCitation] = []
    for vc in verified_citations:
        if vc.verdict != "green" or vc.cluster_id is None:
            continue
        if vc.cluster_id in seen_clusters:
            continue
        seen_clusters.add(vc.cluster_id)
        unique_targets.append(vc)

    with httpx.Client(headers=headers) as client:
        for vc in unique_targets:
            sig = TreatmentSignal(
                citation_text=vc.text,
                cluster_id=vc.cluster_id,
            )

            try:
                results_list = _search(client, vc.text)
            except Exception as e:
                sig.reason = f"search failed: {type(e).__name__}: {e}"
                results.append(sig)
                continue

            sig.citing_opinions_returned = len(results_list)

            citing = [
                r for r in results_list if _result_cluster_id(r) != vc.cluster_id
            ][:MAX_CITING_OPINIONS]

            all_hits: list[dict[str, Any]] = []
            for r in citing:
                snippet = _snippet_text(r)
                if not snippet:
                    continue
                sig.citing_opinions_scanned += 1
                hits = _scan_for_negative_treatment(snippet, vc.text)
                for h in hits:
                    h["citing_case"] = r.get("caseName") or r.get("case_name") or ""
                    h["citing_date"] = r.get("dateFiled") or r.get("date_filed") or ""
                    h["citing_cluster_id"] = _result_cluster_id(r)
                all_hits.extend(hits)

            sig.negative_hits = all_hits

            red_hits = [h for h in all_hits if h["severity"] == "red"]
            yellow_hits = [h for h in all_hits if h["severity"] == "yellow"]

            if red_hits:
                sig.verdict = "red"
                sig.reason = (
                    f"{len(red_hits)} strong negative-treatment signal(s) "
                    f"across {sig.citing_opinions_scanned} citing opinions "
                    f"(term: '{red_hits[0]['term']}' near the citation) - "
                    f"verify with a paid citator"
                )
            elif yellow_hits:
                sig.verdict = "yellow"
                sig.reason = (
                    f"{len(yellow_hits)} weak negative-treatment signal(s) "
                    f"across {sig.citing_opinions_scanned} citing opinions "
                    f"(term: '{yellow_hits[0]['term']}' near the citation) - "
                    f"possibly limited"
                )
            else:
                sig.verdict = "green"
                sig.reason = (
                    f"no negative-treatment signals in "
                    f"{sig.citing_opinions_scanned} citing opinions scanned "
                    f"(search returned {sig.citing_opinions_returned})"
                )

            results.append(sig)

    return results


def _emoji(v: str) -> str:
    return {"green": "[OK]", "yellow": "[??]", "red": "[XX]"}.get(v, "[  ]")


def main() -> None:
    load_dotenv()
    token = os.getenv("COURTLISTENER_TOKEN")
    if not token:
        sys.exit("COURTLISTENER_TOKEN missing in .env")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brief.txt"
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    print(f"\nAnalyzing {path}\n")
    vcs = verify(raw, token)
    signals = check_treatment(vcs, token)

    print(f"\n--- TREATMENT CHECK: {len(signals)} cases scanned ---\n")
    counts = {"green": 0, "yellow": 0, "red": 0}
    for s in signals:
        counts[s.verdict] = counts.get(s.verdict, 0) + 1
        print(f"{_emoji(s.verdict)} {s.citation_text}")
        print(f"       {s.reason}")
        for h in s.negative_hits[:2]:
            print(
                f"       hit: '{h['term']}' in {h['citing_case']} "
                f"({h['citing_date']}) at distance {h['distance_to_cite']}"
            )
            print(f"         context: ...{h['context'][:180]}...")
        print()

    print(
        f"Treatment: green={counts['green']}  "
        f"yellow={counts['yellow']}  red={counts['red']}"
    )

    os.makedirs("out", exist_ok=True)
    with open("out/treatment.json", "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in signals], f, indent=2, default=str)
    print("\nWrote out/treatment.json")


if __name__ == "__main__":
    main()
