#!/usr/bin/env python3

import os
import csv
from dotenv import load_dotenv
from pronto import Ontology
from collections import defaultdict

# Load environment variables
load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def build_hpo_graph(hpo_ontology):
    """
    Builds a dict of { parent_id -> set(child_ids) } for direct children only.
    """
    hpo_graph = defaultdict(set)
    for term in hpo_ontology.terms():
        # Only track standard HPO terms
        if term.id.startswith("HP:"):
            for parent in term.superclasses(distance=1):  # direct parents only
                if parent.id.startswith("HP:"):
                    hpo_graph[parent.id].add(term.id)
    return hpo_graph

def get_limited_descendants(start_id, known_opposites, hpo_graph):
    """
    DFS to collect descendants of `start_id`. 
    Stops exploring deeper if a node is already in `known_opposites`.
    Returns all visited nodes (including start_id).
    """
    visited = set()
    stack = [start_id]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        # Stop further expansion if current already has an opposite
        if current in known_opposites:
            continue

        # Explore children
        for child in hpo_graph[current]:
            if child not in visited:
                stack.append(child)

    return visited

def propagate_opposites(hpo_obo_path, unified_opposites_path, output_path):
    """
    Propagates opposite_of relationships by:
      1. Building an HPO graph (parent->children).
      2. Reading original opposite pairs from `hpo_opposites_unified.csv`.
      3. For each pair, DFS from both sides with early stopping.
      4. Cross all descendant nodes to create new inherited pairs.
      5. Mark new nodes in `known_opposites` to stop expansions in later pairs.
    """
    # Load HPO ontology
    ontology = Ontology(hpo_obo_path)
    # Extract only terms that start with HP:
    terms_by_id = {t.id: t for t in ontology.terms() if t.id.startswith("HP:")}

    # Build parent->children graph
    hpo_graph = build_hpo_graph(ontology)

    # Read original (non-inherited) opposite pairs
    existing_pairs = set()
    rows_original = []
    with open(unified_opposites_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            id1 = row["id1"].strip()
            id2 = row["id2"].strip()
            existing_pairs.add((id1, id2))
            existing_pairs.add((id2, id1))
            row["inherit"] = "f"
            rows_original.append(row)

    # Keep track of which nodes are already known to have an opposite
    known_opposites = set()
    rows_inherited = []

    # DFS expansions
    for row in rows_original:
        id1, id2 = row["id1"], row["id2"]

        # Collect descendants of each root
        s1 = get_limited_descendants(id1, known_opposites, hpo_graph) if id1 in terms_by_id else set()
        s2 = get_limited_descendants(id2, known_opposites, hpo_graph) if id2 in terms_by_id else set()

        # Cross all combinations
        for x in s1:
            for y in s2:
                if x != y and (x, y) not in existing_pairs:
                    existing_pairs.add((x, y))
                    existing_pairs.add((y, x))
                    known_opposites.add(x)
                    known_opposites.add(y)
                    rows_inherited.append({
                        "id1": x,
                        "id2": y,
                        "logical": row["logical"],
                        "text": row["text"],
                        "inherit": "t",
                        "term1": terms_by_id[x].name.lower() if x in terms_by_id and terms_by_id[x].name else "",
                        "term2": terms_by_id[y].name.lower() if y in terms_by_id and terms_by_id[y].name else ""
                    })

        # Mark the root nodes themselves as known to have opposites
        if id1 in terms_by_id:
            known_opposites.add(id1)
        if id2 in terms_by_id:
            known_opposites.add(id2)

    final_rows = rows_original + rows_inherited
    fieldnames = ["id1", "id2", "logical", "text", "inherit", "term1", "term2"]

    # Write output
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"[INFO] Propagated opposite_of relationships saved to: {output_path}")
    print(f"[INFO] Original pairs: {len(rows_original)} | Inherited pairs: {len(rows_inherited)} | Total: {len(final_rows)}")

if __name__ == "__main__":
    if not INPUT_DIR or not OUTPUT_DIR:
        raise ValueError("[ERROR] INPUT_DIR and OUTPUT_DIR must be set.")

    hpo_obo = os.path.join(INPUT_DIR, "hp.obo")
    unified_opposites = os.path.join(OUTPUT_DIR, "hpo_opposites_unified.csv")
    output_file = os.path.join(OUTPUT_DIR, "hpo_opposites_inherited.csv")

    propagate_opposites(hpo_obo, unified_opposites, output_file)
