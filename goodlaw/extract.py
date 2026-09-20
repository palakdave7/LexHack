"""Layer 0 - extract every legal citation from a block of text. Fully local."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

from eyecite import clean_text, get_citations
from eyecite.models import FullCaseCitation
from eyecite import resolve_citations
from eyecite.models import (
    FullCaseCitation,
    IdCitation,
    ShortCaseCitation,
    SupraCitation,
)
@dataclass
class ExtractedCitation:
    text: str
    kind: str
    start: int
    end: int
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    case_name: str | None = None
    year: str | None = None
    pin_cite: str | None = None
    lookupable: bool = False
    # NEW: for short-form cites, the text of the full cite they resolve to
    resolved_to: str | None = None
    resolved_pin_cite: str | None = None

def extract(raw: str) -> list[ExtractedCitation]:
    text = clean_text(raw, ["all_whitespace"])
    raw_cites = get_citations(text)

    # Resolve short-form (Id, supra, short case) to their full-form parents.
    # Returns dict[Resource, list[CitationBase]] - each resource is one
    # underlying case, and every mention of it in the doc groups under it.
    resolutions = resolve_citations(raw_cites)

    # Invert: for each cite object, remember which full cite it resolves to.
    cite_to_full: dict[int, FullCaseCitation] = {}
    for resource, cite_list in resolutions.items():
        # The resource's citation attribute is the canonical full cite for the group
        anchor = getattr(resource, "citation", None)
        if not isinstance(anchor, FullCaseCitation):
            continue
        for c in cite_list:
            cite_to_full[id(c)] = anchor

    results: list[ExtractedCitation] = []
    for c in raw_cites:
        start, end = c.span()
        groups = getattr(c, "groups", {}) or {}
        md = getattr(c, "metadata", None)

        plaintiff = getattr(md, "plaintiff", None)
        defendant = getattr(md, "defendant", None)
        case_name = f"{plaintiff} v. {defendant}" if plaintiff and defendant else None
        year = getattr(c, "year", None) or getattr(md, "year", None)

        # Pull pin cite from wherever eyecite put it (metadata.pin_cite for
        # full cites, metadata.pin_cite for short/id/supra too).
        pin_cite = getattr(md, "pin_cite", None)

        # For a short-form cite, look up the full cite it resolves to.
        resolved_to = None
        resolved_pin = None
        anchor = cite_to_full.get(id(c))
        if anchor is not None and not isinstance(c, FullCaseCitation):
            resolved_to = anchor.matched_text()
            resolved_pin = pin_cite  # the pin cite on the short form is what matters

        # A citation is "lookupable" if we can send something concrete to
        # CourtListener - a full cite, or a short cite that resolves to one.
        lookupable = isinstance(c, FullCaseCitation) or resolved_to is not None

        results.append(
            ExtractedCitation(
                text=c.matched_text(),
                kind=type(c).__name__,
                start=start,
                end=end,
                volume=groups.get("volume"),
                reporter=groups.get("reporter"),
                page=groups.get("page"),
                case_name=case_name,
                year=str(year) if year else None,
                pin_cite=pin_cite,
                lookupable=lookupable,
                resolved_to=resolved_to,
                resolved_pin_cite=resolved_pin,
            )
        )

    return results


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_brief.txt"
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    cites = extract(raw)
    lookupable = sum(1 for c in cites if c.lookupable)

    print(f"\n{len(cites)} citations found in {path} "
          f"({lookupable} resolvable against a case database)\n")

    for c in cites:
        mark = "*" if c.lookupable else " "
        print(f"{mark} [{c.kind:<20}] chars {c.start:>4}-{c.end:<4} {c.text!r}")
        if isinstance_short := c.resolved_to:
            print(f"      -> resolves to: {isinstance_short!r} pin={c.resolved_pin_cite}")
        elif c.lookupable:
            print(f"      volume={c.volume} reporter={c.reporter} page={c.page} "
                  f"name={c.case_name} year={c.year} pin={c.pin_cite}")
    os.makedirs("out", exist_ok=True)
    with open("out/extracted.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cites], f, indent=2)
    print("\nWrote out/extracted.json")


if __name__ == "__main__":
    main()