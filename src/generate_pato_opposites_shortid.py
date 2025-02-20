#!/usr/bin/env python3

import os
import re
from dotenv import load_dotenv
from owlready2 import World

# Load environment variables
load_dotenv()

# Get input and output directories from .env
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def short_pato_id(full_iri):
    """
    Given a PATO IRI like:
      http://purl.obolibrary.org/obo/PATO_0000299
    or
      obo:PATO_0000299
    returns just the local ID: PATO_0000299
    """
    match = re.search(r"(PATO_\d+)$", full_iri)
    return match.group(1) if match else full_iri

def generate_pato_opposites_no_duplicates():
    # Paths for input and output
    pato_owl_path = os.path.join(INPUT_DIR, "pato-full.owl")
    output_tsv = os.path.join(OUTPUT_DIR, "pato_opposites.tsv")

    # Check file existence
    if not os.path.exists(pato_owl_path):
        print(f"[ERROR] Cannot find PATO OWL file: {pato_owl_path}")
        return

    # Load PATO ontology
    print(f"[INFO] Loading PATO from: {pato_owl_path}")
    world = World()
    pato_ont = world.get_ontology(pato_owl_path).load()
    graph = world.as_rdflib_graph()

    # Query to find the property with rdfs:label 'is_opposite_of'
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
        open(output_tsv, 'w').close()
        return

    # Gather all pairs, storing in canonical order
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
            subj_iri = str(row[0])
            obj_iri  = str(row[1])

            # Convert to short IDs (e.g., PATO_0000299)
            subj_id = short_pato_id(subj_iri)
            obj_id  = short_pato_id(obj_iri)

            # Store in canonical (min, max) order to avoid duplicates
            pair = tuple(sorted([subj_id, obj_id]))
            pairs.add(pair)

    # Write to TSV using just the short IDs
    with open(output_tsv, 'w', encoding='utf-8') as f:
        for (s, o) in sorted(pairs):
            f.write(f"{s}\t{o}\n")

    print(f"[INFO] Wrote {len(pairs)} unique 'is_opposite_of' pairs (short IDs) to {output_tsv}")

if __name__ == "__main__":
    generate_pato_opposites_no_duplicates()