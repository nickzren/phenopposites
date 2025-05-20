#!/usr/bin/env python3
"""
list_exact_bearer_ontologies.py
───────────────────────────────
Discover every ontology *prefix* that is actually used as a **bearer** for a
PATO quality anywhere in HPO, *without relying on a hard-coded allow-list*.

Logic
-----
A class is considered a bearer iff it is a **named class that appears at the
top level of the filler** of one of these relations inside a logical
definition:

  • RO:0000052   inheres_in
  • RO:0002502   inheres_in_part_of
  • RO:0002314   inheres_in_part_of   (alt ID)
  • RO:0000058   towards
  • BFO:0000066  occurs_in

Nested classes reached only through `part_of` or other restrictions are
ignored.  All unique prefixes (text before “_” in the CURIE) are printed,
except **HP** (self-references).

Usage
-----
python list_exact_bearer_ontologies.py --hp-owl data/input/hp.owl
"""

import argparse, re
from owlready2 import get_ontology, Restriction, And, Or, Not

ID_RX       = re.compile(r'([A-Za-z]+)_[0-9]+')
LINK_LOCAL  = {"RO_0000052","RO_0002502","RO_0002314",
               "RO_0000058","BFO_0000066"}
Q_MODIFIER  = "RO_0002573"

def curie(iri: str):
    m = ID_RX.search(iri)
    return m.group() if m else None

def prefix_from_iri(iri: str):
    m = ID_RX.search(iri)
    return m.group(1) if m else None

def local_id(iri: str):
    m = re.search(r'([A-Za-z]+_[0-9]+)$', iri)
    return m.group(1) if m else None

def direct_prefixes(filler):
    """Return prefixes of named classes at the top level of the filler."""
    out = set()
    if hasattr(filler, "iri"):
        p = prefix_from_iri(filler.iri)
        if p and p != "HP":
            out.add(p)
            return out
    if isinstance(filler, And):
        for part in filler.Classes:
            if hasattr(part, "iri"):
                p = prefix_from_iri(part.iri)
                if p and p != "HP":
                    out.add(p)
    return out

def process_and(and_expr, seen):
    # detect at least one PATO quality in the group
    has_quality = any(
        (hasattr(x, "iri") and prefix_from_iri(x.iri) == "PATO") or
        (isinstance(x, Restriction) and local_id(x.property.iri) == Q_MODIFIER
         and hasattr(x.value, "iri") and prefix_from_iri(x.value.iri) == "PATO")
        for x in and_expr.Classes
    )
    if not has_quality:
        return

    for sub in and_expr.Classes:
        if isinstance(sub, Restriction) and local_id(sub.property.iri) in LINK_LOCAL:
            seen |= direct_prefixes(sub.value)

def walk(expr, seen):
    if isinstance(expr, And):
        process_and(expr, seen)
        for part in expr.Classes: walk(part, seen)
    elif isinstance(expr, Restriction):
        walk(expr.value, seen)
    elif isinstance(expr, Or):
        for part in expr.Classes: walk(part, seen)
    elif isinstance(expr, Not):
        walk(expr.Class, seen)

def list_bearer_prefixes(hp_owl_path: str):
    onto = get_ontology(hp_owl_path).load()
    prefixes = set()
    for cls in onto.classes():
        for ax in list(cls.is_a) + list(cls.equivalent_to):
            walk(ax, prefixes)
    for p in sorted(prefixes):
        print(p)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp-owl", required=True, help="Path to hp.owl")
    list_bearer_prefixes(ap.parse_args().hp_owl)