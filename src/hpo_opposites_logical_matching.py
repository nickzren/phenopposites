#!/usr/bin/env python3

import os
import re
import pandas as pd
from collections import defaultdict
from owlready2 import World
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def format_obo_id(obo_id):
    """
    Converts an OBO id from underscore to colon format.
    For example, 'HP_0000098' becomes 'HP:0000098'.
    """
    return obo_id.replace("_", ":", 1) if "_" in obo_id else obo_id

def short_hpo_id(full_iri):
    """
    Extracts HP_####### from a URI like:
      http://purl.obolibrary.org/obo/HP_0000098
    Returns the formatted id 'HP:0000098' if found; otherwise returns the input.
    """
    match = re.search(r"(HP_\d+)$", full_iri)
    return format_obo_id(match.group(1)) if match else full_iri

def load_opposites_from_tsv():
    """
    Reads pato_opposites.tsv, where each line has:
      PATO_0000xxxx [tab] PATO_0000yyyy
    Returns a dict { quality_id : set([opposite_id, ...]) } with formatted ids.
    """
    pato_opposites_path = os.path.join(OUTPUT_DIR, "pato_opposites.csv")
    if not os.path.exists(pato_opposites_path):
        print(f"[ERROR] Missing file: {pato_opposites_path}")
        return {}

    opp_map = defaultdict(set)
    with open(pato_opposites_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            q1, q2 = parts[0].strip(), parts[1].strip()
            # Format the PATO ids
            q1 = format_obo_id(q1)
            q2 = format_obo_id(q2)
            opp_map[q1].add(q2)
            opp_map[q2].add(q1)
    return dict(opp_map)

def extract_hpo_bearer_quality():
    """
    Loads hp.owl using Owlready2, collects .is_a / .equivalent_to lines 
    for classes whose IRI starts with "HP_" to identify references to UBERON_#### 
    (bearer) and PATO_#### (quality). 
    Writes results to hpo_bearer_quality.csv.

    Returns a dict: { (bearer, quality) : set([hpo_id, ...]) },
    plus a dict of { hpo_id : label } for quick reference.
    """
    hp_owl_path = os.path.join(INPUT_DIR, "hp.owl")
    if not os.path.exists(hp_owl_path):
        print(f"[ERROR] Missing HPO file: {hp_owl_path}")
        return {}, {}

    print(f"[INFO] Loading HPO from {hp_owl_path} ...")
    world = World()
    hpo_ont = world.get_ontology(hp_owl_path).load()

    bearer_quality_map = defaultdict(set)
    hp_labels = {}

    # Process classes in sorted order by IRI
    for cls in sorted(list(hpo_ont.classes()), key=lambda cls: str(cls.iri)):
        full_iri = str(cls.iri)
        if not full_iri.startswith("http://purl.obolibrary.org/obo/HP_"):
            continue

        short_id = short_hpo_id(full_iri)

        if cls.label:
            hp_labels[short_id] = ";".join(sorted(cls.label))
        else:
            hp_labels[short_id] = ""

        is_a_exprs = getattr(cls, "is_a", [])
        eq_exprs  = getattr(cls, "equivalent_to", [])
        all_strs = [str(x) for x in is_a_exprs] + [str(x) for x in eq_exprs]

        has_uberon = any("UBERON_" in s for s in all_strs)
        has_pato   = any("PATO_"   in s for s in all_strs)
        if not (has_uberon and has_pato):
            continue

        ub_ids = set()
        pt_ids = set()
        for s in all_strs:
            ub_ids.update(re.findall(r"UBERON_\d+", s))
            pt_ids.update(re.findall(r"PATO_\d+",   s))

        # Format IDs for consistent output and use sorted order
        for ub in sorted(ub_ids):
            formatted_ub = format_obo_id(ub)
            for pt in sorted(pt_ids):
                formatted_pt = format_obo_id(pt)
                bearer_quality_map[(formatted_ub, formatted_pt)].add(short_id)

    # Write results in sorted order with formatted ids
    rows = []
    for (bearer, quality) in sorted(bearer_quality_map.keys()):
        for hpo_id in sorted(bearer_quality_map[(bearer, quality)]):
            rows.append({
                "hpo_id": hpo_id,
                "hpo_term": hp_labels.get(hpo_id, ""),
                "bearer_iri": bearer,
                "quality_iri": quality
            })
    df = pd.DataFrame(rows, columns=["hpo_id", "hpo_term", "bearer_iri", "quality_iri"])
    out_csv = os.path.join(OUTPUT_DIR, "hpo_bearer_quality.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {len(df)} (bearer, quality) pairs to {out_csv}")

    return bearer_quality_map, hp_labels

def find_opposites(bearer_quality_map, opposite_map, hp_labels):
    """
    bearer_quality_map: { (bearer, quality) : set([hpo_id, ...]) }
    opposite_map: { quality_id : set([opposite_quality_id, ...]) }
    hp_labels: { hpo_id : label }

    Returns a list of opposite phenotypes:
      (hpo_id1, hpo_term1, hpo_id2, hpo_term2, bearer, quality1, quality2)
    """
    bearer_dict = defaultdict(list)
    for (bearer, quality), hpo_ids in bearer_quality_map.items():
        bearer_dict[bearer].append((quality, sorted(hpo_ids)))

    opposite_phenos = []
    for bearer in sorted(bearer_dict.keys()):
        qual_list = sorted(bearer_dict[bearer], key=lambda x: x[0])
        n = len(qual_list)
        for i in range(n):
            q1, hp_list1 = qual_list[i]
            if q1 not in opposite_map:
                continue
            opp_q_for_q1 = sorted(opposite_map[q1])
            for j in range(i+1, n):
                q2, hp_list2 = qual_list[j]
                if q2 in opp_q_for_q1:
                    for c1 in hp_list1:
                        for c2 in hp_list2:
                            if c1 != c2:
                                opposite_phenos.append((
                                    c1, hp_labels.get(c1, ""),
                                    c2, hp_labels.get(c2, ""),
                                    bearer, q1, q2
                                ))
    opposite_phenos = sorted(opposite_phenos, key=lambda x: (x[0], x[2], x[4], x[5], x[6]))
    return opposite_phenos

def main():
    # 1) Load PATO opposites (with formatted ids)
    opposite_map = load_opposites_from_tsv()

    # 2) Extract (bearer, quality) from HPO (with formatted ids)
    bearer_quality_map, hp_labels = extract_hpo_bearer_quality()
    if not bearer_quality_map:
        print("[INFO] No bearer-quality pairs found, or data missing.")
        return

    # 3) Find opposite phenotypes
    opposite_phenos = find_opposites(bearer_quality_map, opposite_map, hp_labels)

    # 4) Write results
    out_csv = os.path.join(OUTPUT_DIR, "hpo_opposites_logical.csv")
    rows = []
    for (c1, lbl1, c2, lbl2, bearer, q1, q2) in opposite_phenos:
        rows.append({
            "hpo_id1": c1,
            "hpo_term1": lbl1,
            "hpo_id2": c2,
            "hpo_term2": lbl2
        })
    rows = sorted(rows, key=lambda r: (r["hpo_id1"], r["hpo_id2"]))
    df = pd.DataFrame(rows, columns=["hpo_id1", "hpo_term1", "hpo_id2", "hpo_term2"])
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Found {len(df)} opposite phenotype pairs. Wrote to {out_csv}")

if __name__ == "__main__":
    main()