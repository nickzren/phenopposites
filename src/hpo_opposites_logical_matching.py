#!/usr/bin/env python3
"""
Generate phenotype opposites by matching bearer–quality pairs whose qualities
are logical opposites (from pato_opposites.csv). 
"""
import os
import re
from collections import defaultdict
from typing import Dict, Set, Tuple, List

import pandas as pd
from owlready2 import World
from dotenv import load_dotenv

load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
if not INPUT_DIR or not OUTPUT_DIR:
    raise EnvironmentError("INPUT_DIR and OUTPUT_DIR environment variables must be set.")

HP_OWL_PATH = os.path.join(INPUT_DIR, "hp.owl")
PATO_OPPOSITES_PATH = os.path.join(OUTPUT_DIR, "pato_opposites.csv")
HPO_BEARER_QUALITY_CSV = os.path.join(OUTPUT_DIR, "hpo_bearer_quality.csv")
LOGICAL_OUT_CSV = os.path.join(OUTPUT_DIR, "hpo_opposites_logical.csv")


def _format_obo_id(obo_id: str) -> str:
    return obo_id.replace("_", ":", 1) if "_" in obo_id else obo_id


def _short_hpo_id(full_iri: str) -> str:
    match = re.search(r"(HP_\d+)$", full_iri)
    return _format_obo_id(match.group(1)) if match else full_iri


def _assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def _load_opposite_qualities() -> Dict[str, Set[str]]:
    _assert_exists(PATO_OPPOSITES_PATH)
    df = pd.read_csv(PATO_OPPOSITES_PATH, header=None, names=["q1", "q2"])
    mapping: Dict[str, Set[str]] = defaultdict(set)
    for q1, q2 in df.itertuples(index=False):
        q1, q2 = map(str.strip, (q1, q2))
        q1, q2 = _format_obo_id(q1), _format_obo_id(q2)
        mapping[q1].add(q2)
        mapping[q2].add(q1)
    return dict(mapping)


def _parse_expressions(cls) -> Tuple[Set[str], Set[str]]:
    expressions = getattr(cls, "is_a", []) + getattr(cls, "equivalent_to", [])
    ub_ids, pt_ids = set(), set()
    for expr in expressions:
        expr_str = str(expr)
        ub_ids.update(re.findall(r"UBERON_\d+", expr_str))
        pt_ids.update(re.findall(r"PATO_\d+", expr_str))
    return ub_ids, pt_ids


def _extract_bearer_quality() -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[str, str]]:
    _assert_exists(HP_OWL_PATH)
    world = World()
    hpo_ont = world.get_ontology(HP_OWL_PATH).load()

    bearer_quality: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    hp_labels: Dict[str, str] = {}

    for cls in sorted(hpo_ont.classes(), key=lambda c: str(c.iri)):
        full_iri = str(cls.iri)
        if not full_iri.startswith("http://purl.obolibrary.org/obo/HP_"):
            continue

        hpo_id = _short_hpo_id(full_iri)
        hp_labels[hpo_id] = ";".join(sorted(cls.label)) if cls.label else ""

        ub_ids, pt_ids = _parse_expressions(cls)
        if not (ub_ids and pt_ids):
            continue

        for ub in ub_ids:
            for pt in pt_ids:
                bearer_quality[(_format_obo_id(ub), _format_obo_id(pt))].add(hpo_id)

    rows = [
        {
            "hpo_id": hpo_id,
            "hpo_term": hp_labels[hpo_id],
            "bearer_iri": bearer,
            "quality_iri": quality,
        }
        for (bearer, quality), ids in sorted(bearer_quality.items())
        for hpo_id in sorted(ids)
    ]
    pd.DataFrame(rows).to_csv(HPO_BEARER_QUALITY_CSV, index=False)
    print(f"Wrote {len(rows)} bearer-quality pairs to {HPO_BEARER_QUALITY_CSV}")
    return bearer_quality, hp_labels


def _canonical_pair(
    c1: str, t1: str, c2: str, t2: str
) -> Tuple[str, str, str, str]:
    return (c1, t1, c2, t2) if c1 < c2 else (c2, t2, c1, t1)


def _find_opposites(
    bearer_quality: Dict[Tuple[str, str], Set[str]],
    opposites: Dict[str, Set[str]],
    labels: Dict[str, str],
) -> List[Tuple[str, str, str, str]]:
    bearer_dict: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for (bearer, quality), ids in bearer_quality.items():
        bearer_dict[bearer].append((quality, sorted(ids)))

    pairs: Set[Tuple[str, str, str, str]] = set()
    for bearer, q_list in bearer_dict.items():
        q_list.sort(key=lambda x: x[0])
        for i, (q1, ids1) in enumerate(q_list):
            if q1 not in opposites:
                continue
            opp_set = opposites[q1]
            for q2, ids2 in q_list[i + 1 :]:
                if q2 not in opp_set:
                    continue
                for c1 in ids1:
                    for c2 in ids2:
                        if c1 != c2:
                            pairs.add(
                                _canonical_pair(
                                    c1, labels.get(c1, ""), c2, labels.get(c2, "")
                                )
                            )
    return sorted(pairs, key=lambda p: (p[0], p[2]))


def main() -> None:
    opposite_map = _load_opposite_qualities()
    bearer_quality_map, hp_labels = _extract_bearer_quality()
    if not bearer_quality_map:
        print("No bearer-quality pairs found.")
        return

    opposite_pairs = _find_opposites(bearer_quality_map, opposite_map, hp_labels)
    rows = [
        {
            "hpo_id1": id1,
            "hpo_term1": term1,
            "hpo_id2": id2,
            "hpo_term2": term2,
        }
        for id1, term1, id2, term2 in opposite_pairs
    ]

    pd.DataFrame(rows).to_csv(LOGICAL_OUT_CSV, index=False)
    print(f"Found {len(rows)} unique opposite phenotype pairs. Wrote to {LOGICAL_OUT_CSV}")


if __name__ == "__main__":
    main()