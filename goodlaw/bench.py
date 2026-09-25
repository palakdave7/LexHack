"""Benchmark layer 1 (existence + name-mismatch) on labeled citations.

Runs each labeled citation through verify() individually with pacing to
respect CourtListener's rate limit. Reports precision, recall, F1,
confusion matrix, and per-failure-mode recall.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from goodlaw.verify import verify

BENCH_PATH = Path("data/bench/labeled_citations.jsonl")
INTER_REQUEST_DELAY = 6.0  # seconds between CourtListener calls


def _load() -> list[dict]:
    with BENCH_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _predict_one(text: str, token: str) -> str:
    """Wrap the citation as a minimal sentence and classify."""
    doc = f"See {text}"
    results = verify(doc, token)
    case_results = [r for r in results if r.status != "not_applicable"]
    if not case_results:
        # eyecite couldn't parse it as a case cite - counts as fake
        # (bad_reporter fails here by design)
        return "fake"
    r = case_results[0]
    if r.verdict == "green":
        return "real"
    if r.verdict == "red":
        return "fake"
    return "unknown"


def main() -> None:
    load_dotenv()
    token = os.getenv("COURTLISTENER_TOKEN")
    if not token:
        sys.exit("COURTLISTENER_TOKEN missing in .env")

    items = _load()
    print(f"Loaded {len(items)} labeled citations from {BENCH_PATH}")
    print(
        f"Running with {INTER_REQUEST_DELAY}s delay between calls "
        f"(~{len(items) * INTER_REQUEST_DELAY:.0f}s total)\n"
    )

    predictions = []
    for i, item in enumerate(items, 1):
        try:
            pred = _predict_one(item["text"], token)
        except Exception as e:
            pred = "error"
            print(f"  [!!] {item['id']} threw {type(e).__name__}: {e}")
        correct = pred == item["label"]
        predictions.append(
            {
                "id": item["id"],
                "text": item["text"],
                "label": item["label"],
                "source": item["source"],
                "pred": pred,
                "correct": correct,
            }
        )
        mark = "OK" if correct else "XX"
        print(
            f"  [{mark}] {item['id']:<8} label={item['label']:<4} "
            f"pred={pred:<7} {item['text'][:60]}"
        )
        time.sleep(INTER_REQUEST_DELAY)

    tp = sum(1 for p in predictions if p["label"] == "fake" and p["pred"] == "fake")
    fn = sum(1 for p in predictions if p["label"] == "fake" and p["pred"] == "real")
    fp = sum(1 for p in predictions if p["label"] == "real" and p["pred"] == "fake")
    tn = sum(1 for p in predictions if p["label"] == "real" and p["pred"] == "real")
    abst = sum(1 for p in predictions if p["pred"] in ("unknown", "error"))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (positive class = 'fake')")
    print("=" * 60)
    print("                pred=fake   pred=real")
    print(f"  label=fake     TP={tp:<3}      FN={fn}")
    print(f"  label=real     FP={fp:<3}      TN={tn}")
    print(f"  abstained: {abst}")
    print()
    print(f"  Accuracy:  {accuracy:.3f}   (over classified cites)")
    print(f"  Precision: {precision:.3f}   (of cites flagged fake, how many were)")
    print(f"  Recall:    {recall:.3f}   (of true fakes, how many we caught)")
    print(f"  F1:        {f1:.3f}")

    print("\nPer-failure-mode recall (fake bucket):")
    by_source: dict[str, list[dict]] = {}
    for p in predictions:
        if p["label"] == "fake":
            by_source.setdefault(p["source"], []).append(p)
    for src, group in sorted(by_source.items()):
        caught = sum(1 for p in group if p["pred"] == "fake")
        print(f"  {src:<22} {caught}/{len(group)}")

    os.makedirs("out", exist_ok=True)
    with open("out/bench.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_total": len(items),
                "confusion": {
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    "abstained": abst,
                },
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "per_failure_mode": {
                    src: {
                        "n": len(g),
                        "caught": sum(1 for p in g if p["pred"] == "fake"),
                    }
                    for src, g in by_source.items()
                },
                "predictions": predictions,
            },
            f,
            indent=2,
        )
    print("\nWrote out/bench.json")


if __name__ == "__main__":
    main()
