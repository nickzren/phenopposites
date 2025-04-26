#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Set, Tuple

from dotenv import load_dotenv
from pronto import Ontology


log = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


def build_hpo_graph(ontology: Ontology) -> Dict[str, Set[str]]:
    """Return {parent_id -> {child_id, …}} including only HP:* terms."""
    graph: Dict[str, Set[str]] = {}
    for term in ontology.terms():
        if not term.id.startswith("HP:"):
            continue
        for parent in term.superclasses(distance=1):
            if parent.id.startswith("HP:"):
                graph.setdefault(parent.id, set()).add(term.id)
    return graph


def make_descendant_function(graph: Dict[str, Set[str]]):
    """Return a cached descendant resolver tied to *graph*."""

    @lru_cache(maxsize=None)
    def descendants(start: str) -> Set[str]:
        stack = [start]
        seen: Set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(graph.get(cur, ()))
        return seen

    return descendants


def propagate_opposites(
    hpo_path: Path, pairs_path: Path, out_path: Path
) -> None:
    ont = Ontology(str(hpo_path))
    terms = {t.id: t for t in ont.terms() if t.id.startswith("HP:")}

    graph = build_hpo_graph(ont)
    descendants = make_descendant_function(graph)

    existing: Set[Tuple[str, str]] = set()
    original_rows, inherited_rows = [], []

    with pairs_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["inherit"] = "f"
            original_rows.append(row)
            pair = (row["id1"].strip(), row["id2"].strip())
            existing.update({pair, pair[::-1]})

    for row in original_rows:
        s1 = descendants(row["id1"]) if row["id1"] in terms else set()
        s2 = descendants(row["id2"]) if row["id2"] in terms else set()
        for a in s1:
            for b in s2:
                if a == b or (a, b) in existing:
                    continue
                existing.update({(a, b), (b, a)})
                inherited_rows.append(
                    {
                        "id1": a,
                        "id2": b,
                        "logical": row["logical"],
                        "text": row["text"],
                        "inherit": "t",
                        "term1": terms[a].name.lower() if a in terms else "",
                        "term2": terms[b].name.lower() if b in terms else "",
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id1", "id2", "logical", "text", "inherit", "term1", "term2"],
        )
        writer.writeheader()
        writer.writerows(original_rows + inherited_rows)

    log.info(
        "Original: %d | Inherited: %d | Total: %d",
        len(original_rows),
        len(inherited_rows),
        len(original_rows) + len(inherited_rows),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Propagate HPO opposite_of pairs.")
    p.add_argument("--hpo", type=Path, help="Path to hp.obo")
    p.add_argument("--pairs", type=Path, help="CSV of root opposite pairs")
    p.add_argument("--out", type=Path, help="Output CSV path")
    return p.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()
    env_input_dir = os.getenv("INPUT_DIR")
    env_output_dir = os.getenv("OUTPUT_DIR")

    if not args.hpo and not env_input_dir:
        raise SystemExit("Provide --hpo or set INPUT_DIR.")
    if not args.out and not env_output_dir:
        raise SystemExit("Provide --out or set OUTPUT_DIR.")

    hpo_path = Path(args.hpo) if args.hpo else Path(env_input_dir) / "hp.obo"
    pairs_path = (
        Path(args.pairs)
        if args.pairs
        else Path(env_output_dir) / "hpo_opposites_unified.csv"
    )
    out_path = (
        Path(args.out)
        if args.out
        else Path(env_output_dir) / "hpo_opposites_inherited.csv"
    )

    propagate_opposites(hpo_path, pairs_path, out_path)


if __name__ == "__main__":
    main()