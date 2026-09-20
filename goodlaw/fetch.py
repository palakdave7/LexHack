"""Layer 2 plumbing - fetch opinion bodies from CourtListener into the cache."""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from goodlaw import cache
from goodlaw.verify import verify

OPINION_URL = "https://www.courtlistener.com/api/rest/v4/opinions/{id}/"

# CourtListener stores opinion bodies in several fields, populated
# inconsistently across cases. Try in order of preference: plain text first
# (cheapest to work with), then HTML variants, then raw XML.
TEXT_FIELDS_IN_PRIORITY_ORDER = [
    "plain_text",
    "html_with_citations",
    "html",
    "html_lawbox",
    "html_columbia",
    "html_anon_2020",
    "xml_harvard",
]


def _clean(raw: str, is_html: bool) -> str:
    """Strip markup and normalize whitespace. What comes out is what we'll
    search when checking quotes."""
    if is_html:
        parser = "lxml-xml" if raw.lstrip().startswith("<?xml") else "lxml"
        soup = BeautifulSoup(raw, parser)
        # Kill script/style content; keep everything else including footnotes.
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    else:
        text = raw
    # Collapse repeated whitespace and stray control chars.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _get_opinion(client: httpx.Client, opinion_id: int) -> dict[str, Any]:
    r = client.get(OPINION_URL.format(id=opinion_id), timeout=30.0)
    if r.status_code == 429:
        raise httpx.HTTPStatusError("rate limited", request=r.request, response=r)
    r.raise_for_status()
    return r.json()


def fetch_opinion(client: httpx.Client, opinion_id: int, cluster_id: int) -> str | None:
    """Fetch one opinion, cache it, return the cleaned text.
    Returns None if fetch fails or opinion has no usable text field."""
    cached = cache.get_opinion(opinion_id)
    if cached is not None:
        return cached

    prior_failure = cache.had_failure(opinion_id)
    if prior_failure:
        # Don't retry the same failure over and over. Bench can re-run
        # by manually clearing the failures table.
        return None

    try:
        payload = _get_opinion(client, opinion_id)
    except Exception as e:
        cache.record_failure(opinion_id, f"http:{type(e).__name__}:{e}")
        return None

    for field in TEXT_FIELDS_IN_PRIORITY_ORDER:
        raw = payload.get(field)
        if raw and str(raw).strip():
            is_html = field.startswith("html") or field.startswith("xml")
            text = _clean(str(raw), is_html)
            if len(text) < 200:
                # Something odd - stub opinion, error page, whatever. Skip.
                continue
            cache.store_opinion(opinion_id, cluster_id, text, field)
            return text

    cache.record_failure(opinion_id, "no_text_field_populated")
    return None


def fetch_all_for_document(
    raw_text: str, token: str
) -> tuple[dict[int, str], dict[int, list[int]]]:
    """Verify a document, then fetch every opinion in every resolved cluster.

    Returns:
      opinions:            {opinion_id: cleaned_text} for successful fetches
      cluster_to_opinions: {cluster_id: [opinion_id, ...]} - lets layer 2 try
                           every opinion in a cluster when checking a quote,
                           so a quote in the majority is found even when we
                           happened to also fetch the dissent.
    """
    verified = verify(raw_text, token)

    to_fetch: list[tuple[int, int]] = []
    seen: set[int] = set()
    cluster_to_opinions: dict[int, list[int]] = {}
    for vc in verified:
        if not vc.cluster_id or not vc.opinion_ids:
            continue
        cluster_to_opinions.setdefault(vc.cluster_id, [])
        for oid in vc.opinion_ids:
            cluster_to_opinions[vc.cluster_id].append(oid)
            if oid not in seen:
                seen.add(oid)
                to_fetch.append((vc.cluster_id, oid))

    opinions: dict[int, str] = {}
    headers = {"Authorization": f"Token {token}", "User-Agent": "GoodLaw/0.1"}
    with httpx.Client(headers=headers) as client:
        for cluster_id, opinion_id in to_fetch:
            text = fetch_opinion(client, opinion_id, cluster_id)
            status = f"{len(text):,} chars" if text else "FAILED"
            print(f"  cluster {cluster_id} / opinion {opinion_id}: {status}")
            if text:
                opinions[opinion_id] = text

    return opinions, cluster_to_opinions


def main() -> None:
    load_dotenv()
    token = os.getenv("COURTLISTENER_TOKEN")
    if not token:
        sys.exit("COURTLISTENER_TOKEN missing in .env")

    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brief.txt"
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    print(f"\nFetching opinions for citations in {path}\n")
    fetched, cluster_map = fetch_all_for_document(raw, token)

    print(
        f"\n{len(fetched)} opinions in memory this run "
        f"across {len(cluster_map)} clusters\n"
    )

    stats = cache.cache_stats()
    print("Cache state:")
    print(f"  opinions cached: {stats['opinions_cached']}")
    print(f"  total chars:     {stats['total_chars']:,}")
    print(f"  fetch failures:  {stats['failures']}")

    # Preview so we can eyeball that we got the right text
    for oid, text in fetched.items():
        preview = text[:250].replace("\n", " ")
        parent_cluster = next(
            (cid for cid, oids in cluster_map.items() if oid in oids), "?"
        )
        print(f"\n--- opinion {oid} (cluster {parent_cluster}, first 250 chars) ---")
        print(preview)


if __name__ == "__main__":
    main()
