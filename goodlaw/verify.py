"""Layer 1 - existence check via CourtListener's citation-lookup endpoint."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from goodlaw.extract import extract

CL_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"

# Per-citation status codes that CourtListener returns.
STATUS_MEANING = {
    200: "resolved",      # matched a real case in the corpus
    404: "not_found",     # valid format, no case exists at this cite
    300: "ambiguous",     # multiple possible reporters (e.g. "1 H. 150")
    400: "bad_reporter",  # reporter not recognized at all
    429: "rate_limited",
}


@dataclass
class VerifiedCitation:
    text: str
    kind: str
    start: int
    end: int
    parsed_case_name: str | None      # from eyecite, what the document *says*
    parsed_year: str | None
    parsed_pin_cite: str | None

    status_code: int | None = None    # from CourtListener
    status: str = "unchecked"         # human-readable label
    verdict: str = "unknown"          # green | yellow | red | gray

    # populated when status_code == 200
    canonical_name: str | None = None
    canonical_court: str | None = None
    canonical_date: str | None = None
    cluster_id: int | None = None
    opinion_ids: list[int] = field(default_factory=list)
    name_mismatch: bool = False       # doc says "Foo v. Bar" but corpus says "Baz v. Qux"


def _cluster_summary(cluster: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields we care about out of a CourtListener cluster payload."""
    return {
        "canonical_name": cluster.get("case_name") or cluster.get("case_name_short"),
        "canonical_date": cluster.get("date_filed"),
        # sub_opinions is a list of full URLs -> extract the numeric id at the end
        "opinion_ids": [
            int(u.rstrip("/").rsplit("/", 1)[-1])
            for u in cluster.get("sub_opinions", [])
            if u
        ],
        "cluster_id": cluster.get("id"),
    }


def _names_match(parsed: str | None, canonical: str | None) -> bool:
    """Loose party-name check: does the first party surname from the doc appear
    anywhere in the canonical name? Legal citations abbreviate wildly
    ('Brown v. Board' vs 'Brown v. Bd. of Educ. of Topeka') so we stay lenient
    and only flag hard mismatches."""
    if not parsed or not canonical:
        return True  # can't compare -> don't flag
    first_word = parsed.split(" v.")[0].strip().split()[-1].lower()
    return first_word in canonical.lower()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _post(client: httpx.Client, text: str) -> list[dict[str, Any]]:
    """One POST to CourtListener. Retries on transient errors with backoff."""
    r = client.post(CL_URL, data={"text": text}, timeout=30.0)
    if r.status_code == 429:
        # rate-limited -> tenacity will retry after backoff
        raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
    r.raise_for_status()
    return r.json()


def verify(raw_text: str, token: str) -> list[VerifiedCitation]:
    parsed = extract(raw_text)

    # Only send the doc through CourtListener once, not per-citation.
    # /citation-lookup/ extracts and validates every cite in one call
    # (up to 250 per request).
    headers = {"Authorization": f"Token {token}", "User-Agent": "GoodLaw/0.1"}
    with httpx.Client(headers=headers) as client:
        cl_results = _post(client, raw_text)

    # CourtListener returns results keyed by start_index in the *cleaned* text.
    # Our eyecite call also uses cleaned text, so offsets should align. Build a
    # lookup by rounded start position with a small window for slippage.
    cl_by_start: dict[int, dict[str, Any]] = {r["start_index"]: r for r in cl_results}

    out: list[VerifiedCitation] = []
    for p in parsed:
        vc = VerifiedCitation(
            text=p.text,
            kind=p.kind,
            start=p.start,
            end=p.end,
            parsed_case_name=p.case_name,
            parsed_year=p.year,
            parsed_pin_cite=p.pin_cite,
        )

        # Only case citations get sent to CL. Id/supra/law citations are gray.
        if not p.lookupable:
            vc.status = "not_applicable"
            vc.verdict = "gray"
            out.append(vc)
            continue

        # Try exact start match first, then a small window (CL and eyecite
        # occasionally disagree by 1-2 chars on where a cite begins).
        hit = cl_by_start.get(p.start)
        if hit is None:
            for delta in (-2, -1, 1, 2):
                if (p.start + delta) in cl_by_start:
                    hit = cl_by_start[p.start + delta]
                    break

        if hit is None:
            vc.status = "not_returned"
            vc.verdict = "yellow"
            out.append(vc)
            continue

        code = hit.get("status")
        vc.status_code = code
        vc.status = STATUS_MEANING.get(code, f"unknown_{code}")

        clusters = hit.get("clusters", []) or []
        if code == 200 and clusters:
            summary = _cluster_summary(clusters[0])
            vc.canonical_name = summary["canonical_name"]
            vc.canonical_date = summary["canonical_date"]
            vc.cluster_id = summary["cluster_id"]
            vc.opinion_ids = summary["opinion_ids"]
            vc.name_mismatch = not _names_match(vc.parsed_case_name, vc.canonical_name)
            if vc.name_mismatch:
                vc.status = "name_mismatch"
                vc.verdict = "red"
            else:
                vc.verdict = "green"
        elif code == 404:
            vc.verdict = "red"       # format-valid but no such case
        elif code == 300:
            vc.verdict = "yellow"    # ambiguous reporter
        elif code == 400:
            vc.verdict = "red"       # reporter unrecognized
        else:
            vc.verdict = "yellow"

        out.append(vc)

    return out


def _emoji(verdict: str) -> str:
    return {"green": "[OK]", "yellow": "[??]", "red": "[XX]", "gray": "[--]"}.get(
        verdict, "[  ]"
    )


def main() -> None:
    load_dotenv()
    token = os.getenv("COURTLISTENER_TOKEN")
    if not token:
        sys.exit("COURTLISTENER_TOKEN missing. Put it in .env at repo root.")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brief.txt"
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    results = verify(raw, token)

    counts = {"green": 0, "yellow": 0, "red": 0, "gray": 0}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    print(
        f"\n{len(results)} citations checked  |  "
        f"OK green={counts['green']}  ?? yellow={counts['yellow']}  "
        f"XX red={counts['red']}  -- gray={counts['gray']}\n"
    )

    for r in results:
        print(f"{_emoji(r.verdict)} {r.text!r:<40} [{r.status}]")
        if r.canonical_name:
            print(f"       -> resolved: {r.canonical_name} ({r.canonical_date})")
            if r.name_mismatch:
                print(
                    f"       !! NAME MISMATCH: document says "
                    f"{r.parsed_case_name!r}, corpus says {r.canonical_name!r}"
                )
        elif r.verdict == "red":
            print(f"       !! document says {r.parsed_case_name!r} - not in corpus")

    os.makedirs("out", exist_ok=True)
    with open("out/verified.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print("\nWrote out/verified.json")


if __name__ == "__main__":
    main()