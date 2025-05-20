#!/usr/bin/env python3
"""
precise_bearer_quality.py
Return exact bearer–quality pairs for a list of HPO terms.

  • Bearer   = UBERON / CL / GO / CHEBI / PR / NBO
  • Quality  = PATO
  • A pair is recorded only when the quality appears in the same restriction
    whose property is inheres_in / inheres_in_part_of / towards / occurs_in.

Usage
-----
python precise_bearer_quality.py data/input/hp.owl HP:0000093 HP:0003127
"""

import re, sys
from collections import defaultdict
from owlready2 import get_ontology, Restriction, And, Or, Not

# --------------------------------------------------------------------------- #
ID_RX       = re.compile(r'(HP|UBERON|PATO|GO|CL|CHEBI|PR|NBO)_[0-9]+')
BEARERS     = {"UBERON", "CL", "GO", "CHEBI", "PR", "NBO"}
LINK_LOCAL  = {         # local IDs of RO/BFO relations that link bearer↔quality
    "RO_0000052", "RO_0002502", "RO_0002314", "RO_0000058", "BFO_0000066"
}
Q_MODIFIER  = "RO_0002573"   # has_modifier – carries additional PATO term
# --------------------------------------------------------------------------- #


def curie(iri: str):
    """Return (prefix, CURIE) or (None, None) if no match."""
    m = ID_RX.search(iri)
    return (m.group(1), m.group().replace("_", ":")) if m else (None, None)


def local_id(iri: str) -> str | None:
    """Return the last 'PREFIX_1234567' fragment of an IRI, else None."""
    m = re.search(r'([A-Za-z]+_[0-9]+)$', iri)
    return m.group(1) if m else None


def label(cls):
    return cls.label[0] if cls.label else cls.name


# ── direct bearer extraction (no recursion into nested restrictions) ─────────
def direct_bearers(filler):
    bears = set()
    if hasattr(filler, "iri"):
        pref, cur = curie(filler.iri)
        if pref in BEARERS:
            bears.add((label(filler), cur))
            return bears
    if isinstance(filler, And):
        for part in filler.Classes:
            if hasattr(part, "iri"):
                pref, cur = curie(part.iri)
                if pref in BEARERS:
                    bears.add((label(part), cur))
    return bears


# ── process a single And([...]) filler ──────────────────────────────────────
def process_and(and_expr, mapping):
    local_quals = set()

    # pass 1 – collect PATO qualities in this And-group
    for sub in and_expr.Classes:
        # named PATO class
        if hasattr(sub, "iri"):
            pref, cur = curie(sub.iri)
            if pref == "PATO":
                local_quals.add((label(sub), cur))

        # has_modifier some PATO_x
        if isinstance(sub, Restriction):
            if local_id(sub.property.iri) == Q_MODIFIER and hasattr(sub.value, "iri"):
                q_pref, q_cur = curie(sub.value.iri)
                if q_pref == "PATO":
                    local_quals.add((label(sub.value), q_cur))

    # pass 2 – link each bearer to the collected qualities
    for sub in and_expr.Classes:
        if not isinstance(sub, Restriction):
            continue

        prop_local = local_id(sub.property.iri)
        if prop_local not in LINK_LOCAL:
            continue

        for b in direct_bearers(sub.value):
            mapping[b].update(local_quals)


# ── recursive walker ────────────────────────────────────────────────────────
def walk(expr, mapping):
    if isinstance(expr, And):
        process_and(expr, mapping)
        for s in expr.Classes:
            walk(s, mapping)
    elif isinstance(expr, Restriction):
        walk(expr.value, mapping)
    elif isinstance(expr, Or):
        for s in expr.Classes:
            walk(s, mapping)
    elif isinstance(expr, Not):
        walk(expr.Class, mapping)


def extract(term):
    mp = defaultdict(set)          # bearer → {qualities}
    for ax in list(term.is_a) + list(term.equivalent_to):
        walk(ax, mp)
    return mp


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python precise_bearer_quality.py hp.owl HP:0000093 [...]")

    owl_path, hp_ids = sys.argv[1], sys.argv[2:]
    onto = get_ontology(owl_path).load()

    for pid in hp_ids:
        term = onto.search_one(iri=f"*{pid.replace(':', '_')}")
        if not term:
            print(f"{pid} not found"); continue

        print(f"\nPhenotype: {label(term)} ({pid})")
        pq_map = extract(term)
        if not pq_map:
            print("  No bearer–quality pairs found."); continue

        print("  bearer → qualities")
        last = len(pq_map) - 1
        for i, ((b_lbl, b_id), qset) in enumerate(sorted(pq_map.items(), key=lambda x: x[0][1])):
            pre = "└─" if i == last else "├─"
            print(f"  {pre} {b_lbl} ({b_id})")
            for q_lbl, q_id in sorted(qset, key=lambda x: x[1]):
                print(f"      • {q_lbl} ({q_id})")