"""Layer 4 - negative treatment signal.

For each verified case, search later opinions that cite it and look for
language signaling the case has been overruled, abrogated, criticized, or
superseded. This is NOT Shepardization - it is a keyword-based signal that
flags cases likely worth checking against a paid citator before relying on
them in a filing. Reported as such.

Uses CourtListener's /search/?type=o&q=<citation> endpoint, then scans
each citing opinion for negative-treatment terms within a window of the
citation itself.
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

from goodlaw.fetch import _clean  # reuse HTML cleaner from layer 2
from goodlaw.verify import VerifiedCitation, verify

SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
OPINION_URL = "https://www.courtlistener.com/api/rest/v4/opinions/{id}/"

# Terms we scan for near a citation. Ordered from strongest to weakest signal.
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
        r"\blimit(?:ed|ing|s)\b",
        r"\bdeclin(?:ed|ing|es) to follow\b",
        r"\bcalled into (?:doubt|question)\b",
    ],
}

# How much text around a citation counts as "discussing" that citation.
CONTEXT_WINDOW = 300

# Limit search results per citation - each opinion is another fetch.
MAX_CITING_OPINIONS = 10


@dataclass
class TreatmentSignal:
    citation_text: str
    cluster_id: int | None
    citing_opinions_scanned: int = 0
    negative_hits: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = "green"  # green | yellow | red
    reason: str = ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _search(client: httpx.Client, query: str) -> list[dict[str, Any]]:
    r = client.get(
        SEARCH_URL,
        params={"type": "o", "q": query, "order_by": "dateFiled desc"},
        timeout=30.0,
    )
    if r.status_code == 429:
        raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
    r.raise_for_status()
    return r.json().get("results", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _get_opinion_body(client: httpx.Client, opinion_id: int) -> str | None:
    r = client.get(OPINION_URL.format(id=opinion_id), timeout=30.0)
    if r.status_code == 429:
        raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
    if r.status_code != 200:
        return None
    data = r.json()
    for field_name in [
        "plain_text",
        "html_with_citations",
        "html",
        "html_lawbox",
        "xml_harvard",
    ]:
        raw = data.get(field_name)
        if raw and str(raw).strip():
            is_markup = field_name.startswith(("html", "xml"))
            return _clean(str(raw), is_markup)
    return None


def _scan_for_negative_treatment(
    citing_text: str, citation_str: str
) -> list[dict[str, Any]]:
    """Look for negative-treatment terms within CONTEXT_WINDOW chars of
    where the citation appears in the citing opinion."""
    hits: list[dict[str, Any]] = []

    # Find all mentions of this citation in the citing opinion.
    # Loose match on volume+reporter+page (strip commas and normalize spaces).
    cite_pattern = re.escape(citation_str).replace(r"\ ", r"\s+")
    cite_matches = list(re.finditer(cite_pattern, citing_text, re.IGNORECASE))
    if not cite_matches:
        return hits

    for cm in cite_matches:
        start = max(0, cm.start() - CONTEXT_WINDOW)
        end = min(len(citing_text), cm.end() + CONTEXT_WINDOW)
        context = citing_text[start:end]

        for severity, patterns in NEGATIVE_TERMS.items():
            for pat in patterns:
                for tm in re.finditer(pat, context, re.IGNORECASE):
                    hits.append(
                        {
                            "severity": severity,
                            "term": tm.group(),
                            "context": context[
                                max(0, tm.start() - 80) : tm.end() + 80
                            ].strip(),
                        }
                    )
    return hits


def check_treatment(
    verified_citations: list[VerifiedCitation], token: str
) -> list[TreatmentSignal]:
    """Run layer 4 on every green citation. Red cites are already flagged;
    running treatment on them would only muddy the report."""
    results: list[TreatmentSignal] = []
    headers = {"Authorization": f"Token {token}", "User-Agent": "GoodLaw/0.1"}

    # Dedupe by cluster so we only scan each real case once (Id., supra
    # inherit from parent).
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
                sig.reason = f"search failed: {type(e).__name__}"
                results.append(sig)
                continue

            # Filter out the case citing itself (own cluster's opinions)
            citing = [r for r in results_list if r.get("cluster_id") != vc.cluster_id][
                :MAX_CITING_OPINIONS
            ]

            all_hits: list[dict[str, Any]] = []
            for r in citing:
                oid = r.get("id")
                if not oid:
                    continue
                body = _get_opinion_body(client, oid)
                if not body:
                    continue
                sig.citing_opinions_scanned += 1
                hits = _scan_for_negative_treatment(body, vc.text)
                for h in hits:
                    h["citing_opinion_id"] = oid
                    h["citing_case"] = r.get("caseName", "")
                    h["citing_date"] = r.get("dateFiled", "")
                all_hits.extend(hits)

            sig.negative_hits = all_hits

            red_hits = [h for h in all_hits if h["severity"] == "red"]
            yellow_hits = [h for h in all_hits if h["severity"] == "yellow"]

            if red_hits:
                sig.verdict = "red"
                sig.reason = (
                    f"{len(red_hits)} strong negative-treatment signal(s) found "
                    f"across {sig.citing_opinions_scanned} citing opinions "
                    f"(term: '{red_hits[0]['term']}') - verify with a paid citator"
                )
            elif yellow_hits:
                sig.verdict = "yellow"
                sig.reason = (
                    f"{len(yellow_hits)} weak negative-treatment signal(s) "
                    f"found across {sig.citing_opinions_scanned} citing opinions "
                    f"(term: '{yellow_hits[0]['term']}') - possibly limited"
                )
            else:
                sig.verdict = "green"
                sig.reason = (
                    f"no negative-treatment signals found in "
                    f"{sig.citing_opinions_scanned} citing opinions scanned"
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
                f"       hit: '{h['term']}' in {h['citing_case']} ({h['citing_date']})"
            )
            print(f"         context: ...{h['context'][:150]}...")
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
