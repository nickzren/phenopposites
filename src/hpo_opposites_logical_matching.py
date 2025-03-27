#!/usr/bin/env python3

import os
import re
import pandas as pd
from collections import defaultdict
from owlready2 import World
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# Load Bio_ClinicalBERT model once at module level
model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')

if not INPUT_DIR or not OUTPUT_DIR:
    raise EnvironmentError("INPUT_DIR and OUTPUT_DIR environment variables must be set.")

def format_obo_id(obo_id):
    return obo_id.replace("_", ":", 1) if "_" in obo_id else obo_id

def short_hpo_id(full_iri):
    match = re.search(r"(HP_\d+)$", full_iri)
    return format_obo_id(match.group(1)) if match else full_iri

def assert_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")

def load_opposite_qualities():
    """
    Loads quality opposites from pato_opposites.csv.
    Returns a dict {quality_id: set(opposite_ids)}.
    """
    pato_opposites_path = os.path.join(OUTPUT_DIR, "pato_opposites.csv")
    assert_file_exists(pato_opposites_path)

    df = pd.read_csv(pato_opposites_path, header=None, names=["quality1", "quality2"])
    opp_map = defaultdict(set)
    for _, row in df.iterrows():
        q1 = format_obo_id(row["quality1"].strip())
        q2 = format_obo_id(row["quality2"].strip())
        opp_map[q1].add(q2)
        opp_map[q2].add(q1)

    return dict(opp_map)

def parse_ontology_expressions(cls):
    """
    Extracts UBERON and PATO IDs from ontology expressions.
    """
    expressions = getattr(cls, "is_a", []) + getattr(cls, "equivalent_to", [])
    ub_ids, pt_ids = set(), set()

    for expr in expressions:
        expr_str = str(expr)
        ub_ids.update(re.findall(r"UBERON_\d+", expr_str))
        pt_ids.update(re.findall(r"PATO_\d+", expr_str))

    return ub_ids, pt_ids

def extract_hpo_bearer_quality():
    """
    Extracts (bearer, quality) from hp.owl, writes hpo_bearer_quality.csv.
    Returns mappings for bearer_quality_map and hp_labels.
    """
    hp_owl_path = os.path.join(INPUT_DIR, "hp.owl")
    assert_file_exists(hp_owl_path)
    world = World()
    hpo_ont = world.get_ontology(hp_owl_path).load()

    bearer_quality_map, hp_labels = defaultdict(set), {}

    for cls in sorted(hpo_ont.classes(), key=lambda c: str(c.iri)):
        full_iri = str(cls.iri)
        if not full_iri.startswith("http://purl.obolibrary.org/obo/HP_"):
            continue

        short_id = short_hpo_id(full_iri)
        hp_labels[short_id] = ";".join(sorted(cls.label)) if cls.label else ""

        ub_ids, pt_ids = parse_ontology_expressions(cls)
        if not (ub_ids and pt_ids):
            continue

        for ub in sorted(ub_ids):
            formatted_ub = format_obo_id(ub)
            for pt in sorted(pt_ids):
                formatted_pt = format_obo_id(pt)
                bearer_quality_map[(formatted_ub, formatted_pt)].add(short_id)

    rows = [
        {
            "hpo_id": hpo_id,
            "hpo_term": hp_labels[hpo_id],
            "bearer_iri": bearer,
            "quality_iri": quality
        }
        for (bearer, quality), hpo_ids in sorted(bearer_quality_map.items())
        for hpo_id in sorted(hpo_ids)
    ]

    out_csv = os.path.join(OUTPUT_DIR, "hpo_bearer_quality.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} bearer-quality pairs to {out_csv}")

    return bearer_quality_map, hp_labels

def find_opposites(bearer_quality_map, opposite_map, hp_labels):
    """
    Identifies opposite phenotype pairs based on bearer-quality mappings.
    Returns sorted list of opposite phenotype tuples:
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
            opp_q_for_q1 = opposite_map[q1]
            for j in range(i + 1, n):
                q2, hp_list2 = qual_list[j]
                if q2 in opp_q_for_q1:
                    for c1 in hp_list1:
                        for c2 in hp_list2:
                            if c1 != c2:
                                opposite_phenos.append((
                                    c1,
                                    hp_labels.get(c1, ""),
                                    c2,
                                    hp_labels.get(c2, ""),
                                    bearer, q1, q2
                                ))

    opposite_phenos = sorted(opposite_phenos, key=lambda x: (x[0], x[2], x[4], x[5], x[6]))
    return opposite_phenos

def main():
    opposite_map = load_opposite_qualities()
    bearer_quality_map, hp_labels = extract_hpo_bearer_quality()

    if not bearer_quality_map:
        print("No bearer-quality pairs found.")
        return

    opposite_phenos = find_opposites(bearer_quality_map, opposite_map, hp_labels)

    # Build final rows with similarity score
    rows = []
    for (c1, lbl1, c2, lbl2, bearer, q1, q2) in opposite_phenos:
        # Safeguard against empty labels
        text1 = lbl1 if lbl1 else ""
        text2 = lbl2 if lbl2 else ""

        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        score = cosine_similarity(emb1.cpu().numpy().reshape(1, -1),
                                  emb2.cpu().numpy().reshape(1, -1))[0][0]

        rows.append({
            "hpo_id1": c1,
            "hpo_term1": lbl1,
            "hpo_id2": c2,
            "hpo_term2": lbl2,
            "similarity_score": f"{score:.4f}"
        })

    out_csv = os.path.join(OUTPUT_DIR, "hpo_opposites_logical.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Found {len(rows)} opposite phenotype pairs. Wrote to {out_csv}")

if __name__ == "__main__":
    main()