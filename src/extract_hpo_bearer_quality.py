from owlready2 import get_ontology, Thing, Or, And, Not, Restriction
import re

# 1) LOAD THE ONTOLOGY
ontology_path = "data/input/hp.owl"
onto = get_ontology(ontology_path).load()

# 2) SPECIFY THE HPO TERMS YOU WANT TO INSPECT
phenotype_ids = [
    "HP:0003268",
    "HP:0005961"
]

# 3) REGEX TO EXTRACT ONTOLOGY IDS
#    Matches things like HP_0000824, UBERON_0000979, PATO_0000574, etc.
id_pattern = re.compile(r'(HP|UBERON|PATO|GO|CL|CHEBI)_[0-9]+')

def get_label_or_name(cls):
    """Helper to get a human-readable label if present, else use the raw class name."""
    if cls.label and len(cls.label) > 0:
        return cls.label[0]
    return cls.name

def classify_named_class(named_class):
    """
    Determine if 'named_class' is from PATO (i.e., a Quality)
    or from one of the typical 'bearer' ontologies (UBERON, CL, GO, CHEBI, etc.).
    Returns (bearer_set, quality_set).
    """
    bearer_set, quality_set = set(), set()
    iri = named_class.iri
    match = id_pattern.search(iri)
    if not match:
        return bearer_set, quality_set
    
    oid = match.group().replace("_", ":")  # e.g. UBERON_0000979 -> UBERON:0000979
    label = get_label_or_name(named_class)
    
    # PATO => Quality
    if "PATO_" in iri:
        quality_set.add((label, oid))
    # UBERON, CL, GO, CHEBI => "Bearers" (structures or processes or chemicals)
    elif any(x in iri for x in ("UBERON_", "CL_", "GO_", "CHEBI_")):
        bearer_set.add((label, oid))
    # You might add more logic for other ontologies if needed
    
    return bearer_set, quality_set

def parse_class_expression(expr):
    """
    Recursively parse any class expression (named class, Restriction, And, Or, etc.)
    and collect 'bearer' and 'quality' references.
    """
    bearers, qualities = set(), set()
    
    # CASE 1: Named class (e.g., PATO:0000574, UBERON:0000979, etc.)
    if hasattr(expr, "iri"):
        bset, qset = classify_named_class(expr)
        bearers.update(bset)
        qualities.update(qset)
    
    # CASE 2: Restriction, e.g., has_part some X, inheres_in some Y
    if isinstance(expr, Restriction):
        # Dive into the 'value' (the filler for that restriction)
        if hasattr(expr, "value"):
            # Recursively parse the filler
            child_bearers, child_qualities = parse_class_expression(expr.value)
            bearers.update(child_bearers)
            qualities.update(child_qualities)
    
    # CASE 3: Logical AND / OR expression
    #         e.g. And([ClassA, Restriction(...), ClassB])
    #         or Or([ ... ])
    if isinstance(expr, And) or isinstance(expr, Or):
        for subexpr in expr.Classes:
            child_bearers, child_qualities = parse_class_expression(subexpr)
            bearers.update(child_bearers)
            qualities.update(child_qualities)
    
    # CASE 4: Negation (Not) - parse whatever is inside
    if isinstance(expr, Not):
        child_bearers, child_qualities = parse_class_expression(expr.Class)
        bearers.update(child_bearers)
        qualities.update(child_qualities)
    
    return bearers, qualities

def extract_bearers_qualities(cls):
    """
    Gather all logical definitions from both 'is_a' (subClassOf) and 'equivalent_to',
    parse them recursively, and return sets of (label, ID) for bearers and qualities.
    """
    all_bearers, all_qualities = set(), set()
    
    # 1) Check subClassOf axioms
    for super_expr in cls.is_a:
        b, q = parse_class_expression(super_expr)
        all_bearers.update(b)
        all_qualities.update(q)
    
    # 2) Check equivalentTo axioms
    for eq_expr in cls.equivalent_to:
        b, q = parse_class_expression(eq_expr)
        all_bearers.update(b)
        all_qualities.update(q)
    
    return all_bearers, all_qualities

# MAIN EXECUTION:
for pid in phenotype_ids:
    # Convert HP:0000824 -> HP_0000824 to match typical IRIs in HPO
    iri_pid = pid.replace(":", "_")
    
    # Attempt to locate the class using its full IRI pattern
    pheno_class = onto.search_one(iri=f"*{iri_pid}")
    if not pheno_class:
        print(f"Phenotype {pid} not found in the ontology.")
        continue
    
    label = get_label_or_name(pheno_class)
    print(f"\nPhenotype: {label} ({pid})")
    
    # Parse the class definitions
    bearers, qualities = extract_bearers_qualities(pheno_class)
    
    # Print results
    if not bearers and not qualities:
        print("  No bearer or quality references found in logical axioms.")
    else:
        if bearers:
            print("  Bearers:")
            for b_label, b_id in sorted(bearers):
                print(f"    - {b_label} ({b_id})")
        if qualities:
            print("  Qualities:")
            for q_label, q_id in sorted(qualities):
                print(f"    - {q_label} ({q_id})")