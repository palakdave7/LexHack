"""Layer 0 - extract every legal citation from a block of text. Fully local."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

from eyecite import clean_text, get_citations
from eyecite.models import FullCaseCitation


@dataclass
class ExtractedCitation:
    text: str          # exactly as it appears in the document
    kind: str          # FullCaseCitation, ShortCaseCitation, IdCitation, ...
    start: int         # character offset where the citation begins
    end: int           # character offset where it ends
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    case_name: str | None = None
    year: str | None = None
    pin_cite: str | None = None
    lookupable: bool = False   # can CourtListener resolve this one?


def extract(raw: str) -> list[ExtractedCitation]:
    text = clean_text(raw, ["all_whitespace"])
    results: list[ExtractedCitation] = []

    for c in get_citations(text):
        start, end = c.span()
        groups = getattr(c, "groups", {}) or {}
        md = getattr(c, "metadata", None)

        plaintiff = getattr(md, "plaintiff", None)
        defendant = getattr(md, "defendant", None)
        case_name = f"{plaintiff} v. {defendant}" if plaintiff and defendant else None
        year = getattr(c, "year", None) or getattr(md, "year", None)

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
                pin_cite=getattr(md, "pin_cite", None),
                lookupable=isinstance(c, FullCaseCitation),
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
        if c.lookupable:
            print(f"      volume={c.volume} reporter={c.reporter} page={c.page} "
                  f"name={c.case_name} year={c.year} pin={c.pin_cite}")

    os.makedirs("out", exist_ok=True)
    with open("out/extracted.json", "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cites], f, indent=2)
    print("\nWrote out/extracted.json")


if __name__ == "__main__":
    main()