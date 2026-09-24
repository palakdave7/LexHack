"""Layer 3 - proposition support via NLI.

For each citation with a nearby claim sentence, retrieve the passage from
the cited opinion (pin-cite guided if available, else semantic search),
then run NLI to decide whether the passage supports/contradicts/neither.

Model: tasksource/deberta-small-long-nli
  - 142M params, 1680-token context (so retrieved passages aren't silently
    truncated), Apache 2.0, trained on fact-verification data (nli_fever,
    doc-nli). Runs on CPU, no API keys, offline after first download.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from goodlaw.fetch import fetch_all_for_document
from goodlaw.verify import VerifiedCitation, verify

NLI_MODEL = "tasksource/deberta-small-long-nli"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# How many chars around a citation count as "the sentence being cited for".
CLAIM_WINDOW_BEFORE = 400  # sentences leading up to the cite
CLAIM_WINDOW_AFTER = 50  # sometimes the claim continues past the cite

# Passage chunk sizing when we search inside an opinion.
CHUNK_SIZE_CHARS = 1200  # ~250 words, safely under NLI's 1680-token cap
CHUNK_OVERLAP_CHARS = 200
TOP_K_PASSAGES = 3  # try the 3 most similar chunks; keep the highest entailment

# NLI thresholds. The model returns probabilities over {entailment, neutral, contradiction}.
ENTAIL_THRESHOLD = 0.60
CONTRADICT_THRESHOLD = 0.60


@dataclass
class ClaimCheck:
    citation_text: str
    citation_start: int
    cluster_id: int | None
    claim_sentence: str  # the sentence the cite is supporting
    retrieved_passage: str  # what we searched from the opinion
    retrieved_from_opinion: int | None
    entailment_prob: float = 0.0
    neutral_prob: float = 0.0
    contradiction_prob: float = 0.0
    verdict: str = "unchecked"  # green | yellow | red | gray
    reason: str = ""


# --- Model loading (cached across calls in one Python process) --------------


@lru_cache(maxsize=1)
def _load_nli():
    """Load NLI model + tokenizer once per process."""
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    mdl.eval()
    return tok, mdl


@lru_cache(maxsize=1)
def _load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def _label_map(model) -> dict[str, int]:
    """The tasksource model uses id2label; normalize to lowercase keys."""
    return {v.lower(): int(k) for k, v in model.config.id2label.items()}


# --- Text handling ----------------------------------------------------------
# Legal abbreviations that end in a period but do NOT end a sentence.
# Anything with an internal period is fine (U.S., F.3d) - the regex handles
# those below - but abbreviations like "Inc." at the end of a case name
# need this list.
_LEGAL_ABBREV = (
    "U.S",
    "F.2d",
    "F.3d",
    "F.4th",
    "F.Supp",
    "F.Supp.2d",
    "F.Supp.3d",
    "S.Ct",
    "L.Ed",
    "L.Ed.2d",
    "Cal",
    "N.Y",
    "Ill",
    "Mass",
    "Tex",
    "v",
    "vs",
    "Inc",
    "Corp",
    "Co",
    "LLC",
    "Ltd",
    "Ass'n",
    "Bros",
    "Cir",
    "No",
    "Nos",
    "art",
    "sec",
    "id",
    "Id",
)


def _protect(text: str) -> str:
    """Replace periods inside known legal abbreviations with a sentinel so
    a naive sentence splitter doesn't break on them. Also protects internal
    periods like the two in 'U.S.' by matching the abbreviation as a whole."""
    out = text
    for ab in _LEGAL_ABBREV:
        # Match the abbreviation followed by an optional final period,
        # bounded so we don't touch words that merely contain it.
        pattern = re.compile(rf"\b{re.escape(ab)}\.?", re.IGNORECASE)
        out = pattern.sub(lambda m: m.group().replace(".", "\x00"), out)
    # Also protect single letters followed by a period in mid-sentence
    # ("Justice A. Scalia") - runs of these are almost never sentence ends.
    out = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", lambda m: m.group(1) + "\x00", out)
    return out


def _unprotect(text: str) -> str:
    return text.replace("\x00", ".")


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, sentence_text), ...] for the sentences in text.
    Offsets are into the ORIGINAL (pre-protection) text."""
    protected = _protect(text)
    # Split on ., !, or ? followed by whitespace and a capital letter or quote.
    boundaries = [0]
    for m in re.finditer(r"[.!?]\s+(?=[A-Z\"\u201c])", protected):
        boundaries.append(m.end())
    boundaries.append(len(protected))

    out = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        sent = _unprotect(protected[start:end]).strip()
        sent = re.sub(r"\s+", " ", sent)
        if sent:
            out.append((start, end, sent))
    return out


def _sentence_around(raw: str, cite_start: int) -> str:
    """Return the sentence that *contains* the citation at cite_start.

    The claim being cited-for is almost always the sentence ending at the
    citation, or occasionally the sentence just before if the cite stands
    alone. We pick the containing sentence and, if it's mostly citation
    text (short, low letter ratio), fall back to the previous one."""
    sentences = _split_sentences(raw)
    if not sentences:
        return ""

    containing_idx = None
    for i, (start, end, _) in enumerate(sentences):
        if start <= cite_start < end:
            containing_idx = i
            break

    if containing_idx is None:
        # cite is past all sentence boundaries - take the last sentence
        containing_idx = len(sentences) - 1

    candidate = sentences[containing_idx][2]

    # If the "sentence" is mostly citation garbage (few words, dominated by
    # digits and periods), fall back to the previous sentence.
    words = re.findall(r"[A-Za-z]{3,}", candidate)
    if len(words) < 5 and containing_idx > 0:
        candidate = sentences[containing_idx - 1][2] + " " + candidate

    return candidate.strip()


# --- Retrieval --------------------------------------------------------------


def _chunk(text: str) -> list[str]:
    """Break an opinion into overlapping char-window chunks. Simple and
    good enough - we don't need semantic chunking here because retrieval
    is broad (top-1) and NLI does the fine-grained matching."""
    step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
    return [text[i : i + CHUNK_SIZE_CHARS] for i in range(0, len(text), step)]


def _find_pin_cite_passage(opinion_text: str, pin_cite: str | None) -> str | None:
    """If the citation includes a pin cite (e.g. 'at 448'), try to find
    that page marker in the opinion and return the surrounding passage.
    CourtListener's reporter texts include page markers like '*448'."""
    if not pin_cite:
        return None
    m = re.search(r"\d+", pin_cite)
    if not m:
        return None
    page = m.group()
    # Try common page-marker patterns Harvard/CL use
    for pattern in [rf"\*{page}\b", rf"page {page}\b", rf"\[{page}\]"]:
        match = re.search(pattern, opinion_text)
        if match:
            start = max(0, match.start() - 100)
            end = min(len(opinion_text), match.start() + CHUNK_SIZE_CHARS)
            return opinion_text[start:end]
    return None


def _retrieve_passages(
    claim: str,
    opinions: dict[int, str],
    opinion_ids: list[int],
    pin_cite: str | None,
) -> tuple[str, int | None]:
    """Return the single best passage across all opinions in the cluster,
    plus which opinion it came from."""
    # First try pin-cite in each opinion. If found, that's the strongest signal.
    for oid in opinion_ids:
        text = opinions.get(oid)
        if not text:
            continue
        passage = _find_pin_cite_passage(text, pin_cite)
        if passage:
            return passage, oid

    # Fall back to semantic search over chunked opinions
    embedder = _load_embedder()
    claim_vec = embedder.encode(claim, normalize_embeddings=True)

    best_passage = ""
    best_score = -1.0
    best_oid: int | None = None

    for oid in opinion_ids:
        text = opinions.get(oid)
        if not text:
            continue
        chunks = _chunk(text)
        if not chunks:
            continue
        chunk_vecs = embedder.encode(chunks, normalize_embeddings=True, batch_size=16)
        # cosine sim = dot product because both are L2-normalized
        sims = np.asarray(chunk_vecs) @ np.asarray(claim_vec)
        top_idx = int(np.argmax(sims))
        if sims[top_idx] > best_score:
            best_score = float(sims[top_idx])
            best_passage = chunks[top_idx]
            best_oid = oid

    return best_passage, best_oid


# --- NLI --------------------------------------------------------------------


def _nli(premise: str, hypothesis: str) -> dict[str, float]:
    """Return {'entailment': p, 'neutral': p, 'contradiction': p}."""
    tok, mdl = _load_nli()
    inputs = tok(
        premise,
        hypothesis,
        truncation=True,
        max_length=1680,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = mdl(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).tolist()

    label_ix = _label_map(mdl)
    return {
        "entailment": probs[label_ix["entailment"]],
        "neutral": probs[label_ix["neutral"]],
        "contradiction": probs[label_ix["contradiction"]],
    }


# --- Orchestration ----------------------------------------------------------


def check_claims(
    raw: str,
    verified_citations: list[VerifiedCitation],
    opinions: dict[int, str],
    cluster_map: dict[int, list[int]],
) -> list[ClaimCheck]:
    """Run layer 3 on every citation whose earlier layers didn't already
    decide the verdict. Skips red (already fabricated) and gray (statute etc.)."""
    results: list[ClaimCheck] = []

    for vc in verified_citations:
        # Only check citations that layer 1 could resolve to a real case.
        # Red cites are already flagged; gray cites have nothing to check against.
        if vc.verdict != "green" or vc.cluster_id is None:
            continue

        claim = _sentence_around(raw, vc.start)
        if len(claim) < 20:
            continue  # sentence too short to be a real proposition

        opinion_ids = cluster_map.get(vc.cluster_id, [])
        if not opinion_ids:
            continue

        # Prefer resolved pin cite (from short-form) over parsed pin cite
        pin = vc.parsed_pin_cite

        passage, oid = _retrieve_passages(claim, opinions, opinion_ids, pin)
        if not passage:
            results.append(
                ClaimCheck(
                    citation_text=vc.text,
                    citation_start=vc.start,
                    cluster_id=vc.cluster_id,
                    claim_sentence=claim,
                    retrieved_passage="",
                    retrieved_from_opinion=None,
                    verdict="gray",
                    reason="no passage retrievable",
                )
            )
            continue

        probs = _nli(premise=passage, hypothesis=claim)

        cc = ClaimCheck(
            citation_text=vc.text,
            citation_start=vc.start,
            cluster_id=vc.cluster_id,
            claim_sentence=claim,
            retrieved_passage=passage[:400] + ("..." if len(passage) > 400 else ""),
            retrieved_from_opinion=oid,
            entailment_prob=probs["entailment"],
            neutral_prob=probs["neutral"],
            contradiction_prob=probs["contradiction"],
        )

        if probs["contradiction"] >= CONTRADICT_THRESHOLD:
            cc.verdict = "red"
            cc.reason = (
                f"CONTRADICTED by cited passage (p={probs['contradiction']:.2f})"
            )
        elif probs["entailment"] >= ENTAIL_THRESHOLD:
            cc.verdict = "green"
            cc.reason = f"SUPPORTED by cited passage (p={probs['entailment']:.2f})"
        else:
            cc.verdict = "yellow"
            cc.reason = (
                f"UNVERIFIED - passage did not clearly support the claim "
                f"(entail={probs['entailment']:.2f}) - human review required"
            )

        results.append(cc)

    return results


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
    print("(first run downloads NLI + embedder models, ~900MB total)")

    vcs = verify(raw, token)
    opinions, cluster_map = fetch_all_for_document(raw, token)
    claims = check_claims(raw, vcs, opinions, cluster_map)

    print(f"\n--- CLAIM CHECK: {len(claims)} claims analyzed ---\n")
    counts = {"green": 0, "yellow": 0, "red": 0, "gray": 0}
    for c in claims:
        counts[c.verdict] = counts.get(c.verdict, 0) + 1
        preview = c.claim_sentence[:100] + (
            "..." if len(c.claim_sentence) > 100 else ""
        )
        print(f'{_emoji(c.verdict)} "{preview}"')
        print(f"       cite: {c.citation_text}  |  opinion {c.retrieved_from_opinion}")
        print(f"       {c.reason}")
        print()

    print(
        f"Claims: green={counts['green']}  yellow={counts['yellow']}  "
        f"red={counts['red']}  gray={counts['gray']}"
    )

    os.makedirs("out", exist_ok=True)
    with open("out/claims.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in claims], f, indent=2, default=str)
    print("\nWrote out/claims.json")


if __name__ == "__main__":
    main()
