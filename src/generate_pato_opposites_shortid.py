#!/usr/bin/env python3

import os
import re
from dotenv import load_dotenv
from owlready2 import World
import csv

load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def short_pato_id(full_iri):
    match = re.search(r"(PATO_\d+)$", full_iri)
    return match.group(1) if match else full_iri

def generate_pato_opposites_no_duplicates():
    pato_owl_path = os.path.join(INPUT_DIR, "pato-full.owl")
    output_csv = os.path.join(OUTPUT_DIR, "pato_opposites.csv")

    if not os.path.exists(pato_owl_path):
        print(f"[ERROR] Cannot find PATO OWL file: {pato_owl_path}")
        return

    print(f"[INFO] Loading PATO from: {pato_owl_path}")
    world = World()
    pato_ont = world.get_ontology(pato_owl_path).load()
    graph = world.as_rdflib_graph()

    query_find_property = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?prop
    WHERE {
      ?prop rdfs:label ?label .
      FILTER (lcase(str(?label)) = "is_opposite_of")
    }
    """
    matching_props = [str(row[0]) for row in graph.query(query_find_property)]

    if not matching_props:
        print("[WARN] No property labeled 'is_opposite_of' found in PATO.")
        print("[INFO] Creating an empty output file.")
        open(output_csv, 'w').close()
        return

    pairs = set()
    for prop_iri in matching_props:
        query_find_pairs = f"""
        SELECT ?s ?o
        WHERE {{
           ?s <{prop_iri}> ?o .
        }}
        """
        results = graph.query(query_find_pairs)
        for row in results:
            subj_id = short_pato_id(str(row[0]))
            obj_id = short_pato_id(str(row[1]))
            pair = tuple(sorted([subj_id, obj_id]))
            pairs.add(pair)

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for s, o in sorted(pairs):
            writer.writerow([s, o])

    print(f"[INFO] Wrote {len(pairs)} unique 'is_opposite_of' pairs (short IDs) to {output_csv}")

if __name__ == "__main__":
    generate_pato_opposites_no_duplicates()
