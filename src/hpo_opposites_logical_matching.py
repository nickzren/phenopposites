#!/usr/bin/env python3
"""
generate_hpo_opposites.py
Builds a CSV of opposite HPO term pairs using precise bearer–quality mapping.

• hpo_opposites_logical.csv          (column “strict” = 't' or 'f')
"""

import os, itertools, re
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from owlready2 import World, Restriction, And, Or, Not

load_dotenv()
INPUT_DIR  = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

HP_OWL_PATH          = os.path.join(INPUT_DIR,  "hp.owl")
PATO_OPPOSITES_PATH  = os.path.join(OUTPUT_DIR, "pato_opposites.csv")
HPO_BEARER_QUALITY   = os.path.join(OUTPUT_DIR, "hpo_bearer_quality.csv")
LOGICAL_OUT_CSV      = os.path.join(OUTPUT_DIR, "hpo_opposites_logical.csv")

ID_RX       = re.compile(r'(HP|UBERON|PATO|GO|CL|CHEBI|PR|NBO|MPATH)_[0-9]+')
BEARERS     = {"UBERON", "CL", "GO", "CHEBI", "PR", "NBO", "MPATH"}
LINK_LOCAL  = {"RO_0000052","RO_0002502","RO_0002314","RO_0000058","BFO_0000066"}
Q_MODIFIER  = "RO_0002573"
HP_ID_RX    = re.compile(r'(HP_\d+)$')
fmt_id      = lambda obo: obo.replace("_", ":", 1)

def curie(iri: str):
    m = ID_RX.search(iri)
    return (m.group(1), fmt_id(m.group())) if m else (None, None)

def local_id(iri: str):
    m = re.search(r'([A-Za-z]+_[0-9]+)$', iri)
    return m.group(1) if m else None

def direct_bearers(filler):
    """
    Return all bearer classes that occur *directly* within the current filler
    (either the filler itself or its immediate intersection members). This
    preserves the 1‑to‑1 mapping between each bearer set and the local PATO
    qualities collected in process_and().
    """
    out = set()

    def maybe_add(obj):
        if hasattr(obj, "iri"):
            pref, cur = curie(obj.iri)
            if pref in BEARERS:
                out.add(cur)

    # Add the filler itself (it may already be a bearer class)
    maybe_add(filler)

    # Add any named classes that appear directly in an intersection
    if isinstance(filler, And):
        for part in filler.Classes:
            maybe_add(part)

    return out

def allowed_qualities(opp: dict[str, set[str]]) -> set[str]:
    """
    Return the full set of PATO IDs that participate in at least one
    opposite‑of pair. This will be used to filter hpo_bearer_quality.csv.
    """
    return set(opp.keys())

def process_and(and_expr, mapping, allowed_qs):
    local_qs = set()
    for sub in and_expr.Classes:
        if hasattr(sub, "iri") and curie(sub.iri)[0] == "PATO":
            q = curie(sub.iri)[1]
            if q in allowed_qs:
                local_qs.add(q)
        if isinstance(sub, Restriction):
            if local_id(sub.property.iri) == Q_MODIFIER and hasattr(sub.value, "iri"):
                if curie(sub.value.iri)[0] == "PATO":
                    q = curie(sub.value.iri)[1]
                    if q in allowed_qs:
                        local_qs.add(q)

    for sub in and_expr.Classes:
        if isinstance(sub, Restriction) and local_id(sub.property.iri) in LINK_LOCAL:
            for b in direct_bearers(sub.value):
                mapping[b].update(local_qs)

def walk(expr, mapping, allowed_qs):
    if isinstance(expr, And):
        process_and(expr, mapping, allowed_qs)
        for part in expr.Classes:
            walk(part, mapping, allowed_qs)
    elif isinstance(expr, Restriction):
        walk(expr.value, mapping, allowed_qs)
    elif isinstance(expr, Or):
        for part in expr.Classes:
            walk(part, mapping, allowed_qs)
    elif isinstance(expr, Not):
        walk(expr.Class, mapping, allowed_qs)

def load_opposite_map():
    df = pd.read_csv(PATO_OPPOSITES_PATH, header=None, names=["q1","q2"])
    opp = defaultdict(set)
    for q1, q2 in df.itertuples(False):
        q1, q2 = map(lambda x: fmt_id(str(x).strip()), (q1, q2))
        opp[q1].add(q2); opp[q2].add(q1)
    return opp

def extract(allowed_qs):
    w = World(); hpo = w.get_ontology(HP_OWL_PATH).load()
    bmap, qmap, labels, rows = defaultdict(set), defaultdict(set), {}, []
    for cls in hpo.classes():
        iri = str(cls.iri); m = HP_ID_RX.search(iri)
        if not m: continue
        hp = fmt_id(m.group(1))
        labels[hp] = ";".join(sorted(cls.label)) if cls.label else ""
        mapping = defaultdict(set)
        for ax in list(cls.is_a) + list(cls.equivalent_to):
            walk(ax, mapping, allowed_qs)
        if not mapping: continue                     # skip terms w/o bearers
        for b, qs in mapping.items():
            bmap[hp].add(b)
            for q in qs:
                if q in allowed_qs:
                    qmap[hp].add(q)
                    rows.append({"hpo_id":hp,"hpo_term":labels[hp],"bearer_iri":b,"quality_iri":q})
    pd.DataFrame(rows).to_csv(HPO_BEARER_QUALITY, index=False)
    unique_prefixes = sorted({bearer.split(":")[0] for bearer in {row["bearer_iri"] for row in rows}})
    print(f"Unique BEARER prefixes used ({len(unique_prefixes)}):", unique_prefixes)
    return bmap, qmap, labels

def qualities_opposed(q1,q2,opp):
    return any(q in opp and opp[q] & q2 for q in q1)

def make_pair(a,b,lbl):
    return (a,lbl[a],b,lbl[b]) if a<b else (b,lbl[b],a,lbl[a])

def find_pairs(bearers, qualities, opp, labels):
    idx = defaultdict(list)
    for hp, bs in bearers.items():
        for b in bs: idx[b].append(hp)
    strict, nonstrict, seen = set(), set(), set()
    for group in idx.values():
        for a,b in itertools.combinations(sorted(set(group)),2):
            key = (a,b) if a<b else (b,a)
            if key in seen: continue; seen.add(key)
            if not qualities_opposed(qualities[a],qualities[b],opp): continue
            ba, bb = bearers[a], bearers[b]
            pair   = make_pair(a,b,labels)
            if ba and ba==bb:     strict.add(pair)       # identical non-empty
            elif ba & bb:         nonstrict.add(pair)    # overlap but diff
    nonstrict -= strict
    return sorted(strict,key=lambda x:(x[0],x[2])), sorted(nonstrict,key=lambda x:(x[0],x[2]))

def main():
    opp = load_opposite_map()
    aqs = allowed_qualities(opp)

    bmap, qmap, lbl = extract(aqs)

    for hp_id in ("HP:0000024", "HP:0012648"):
        print(f"{hp_id}  bearers: {sorted(bmap.get(hp_id, []))}  "
              f"qualities: {sorted(qmap.get(hp_id, []))}")

    strict, nonstrict = find_pairs(bmap,qmap,opp,lbl)
    # merge into one table with a 'strict' flag ('t' for strict, 'f' otherwise)
    rows = [(*p, "t") for p in strict] + [(*p, "f") for p in nonstrict]
    pd.DataFrame(
        rows,
        columns=["hpo_id1", "hpo_term1", "hpo_id2", "hpo_term2", "strict"]
    ).to_csv(LOGICAL_OUT_CSV, index=False)
    print(f"Total logical pairs: {len(rows):>6} → {LOGICAL_OUT_CSV}")

if __name__ == "__main__":
    if not INPUT_DIR or not OUTPUT_DIR:
        raise EnvironmentError("Set INPUT_DIR and OUTPUT_DIR")
    for p in (HP_OWL_PATH, PATO_OPPOSITES_PATH):
        if not os.path.exists(p): raise FileNotFoundError(p)
    main()