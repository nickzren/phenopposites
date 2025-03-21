#!/usr/bin/env python3
import os
import csv
import re
from dotenv import load_dotenv
from pronto import Ontology

# Load environment variables (like INPUT_DIR, OUTPUT_DIR) from .env file
load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# Morphological prefix swaps (both directions).
# Each tuple is (pattern, replacement). For every prefix pair, we
# include two rules (A->B, B->A).
MORPH_RULES = [
    # (hemi)?hyper <-> (hemi)?hypo
    (re.compile(r'\b(hemi)?hyper(\w*)\b', re.IGNORECASE), r'\1hypo\2'),
    (re.compile(r'\b(hemi)?hypo(\w*)\b',  re.IGNORECASE), r'\1hyper\2'),

    # macro <-> micro
    (re.compile(r'\bmacro(\w*)\b', re.IGNORECASE), r'micro\1'),
    (re.compile(r'\bmicro(\w*)\b', re.IGNORECASE), r'macro\1'),

    # tachy <-> brady
    (re.compile(r'\btachy(\w*)\b', re.IGNORECASE), r'brady\1'),
    (re.compile(r'\bbrady(\w*)\b', re.IGNORECASE), r'tachy\1'),

    # hyperplastic <-> hypoplastic
    (re.compile(r'\bhyperplastic(\w*)\b', re.IGNORECASE), r'hypoplastic\1'),
    (re.compile(r'\bhypoplastic(\w*)\b', re.IGNORECASE), r'hyperplastic\1'),

    # intra <-> extra
    (re.compile(r'\bintra(\w*)\b', re.IGNORECASE), r'extra\1'),
    (re.compile(r'\bextra(\w*)\b', re.IGNORECASE), r'intra\1'),

    # endo <-> exo
    (re.compile(r'\bendo(\w*)\b', re.IGNORECASE), r'exo\1'),
    (re.compile(r'\bexo(\w*)\b', re.IGNORECASE), r'endo\1'),

    # pre <-> post
    (re.compile(r'\bpre(\w*)\b', re.IGNORECASE), r'post\1'),
    (re.compile(r'\bpost(\w*)\b', re.IGNORECASE), r'pre\1'),

    # poly <-> mono
    (re.compile(r'\bpoly(\w*)\b', re.IGNORECASE), r'mono\1'),
    (re.compile(r'\bmono(\w*)\b', re.IGNORECASE), r'poly\1'),
]

def load_antonyms(csv_path):
    """
    Load antonym pairs from CSV into:
      1) whole_word_map  (single-word antonym pairs)
      2) prefix_map      (short prefix swaps for short words)
      3) multi_word_map  (multi-word or hyphenated antonym patterns)

    CSV columns: left,right
    """
    whole_word_map = {}
    prefix_map = {}
    multi_word_map = {}

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            left = row[0].strip().lower()
            right = row[1].strip().lower()

            # 1) Multi-word or hyphenated
            if ' ' in left or ' ' in right or '-' in left or '-' in right:
                multi_word_map[left] = right
                multi_word_map[right] = left

            # 2) Short single-word => prefix-based mapping
            elif len(left) < 6 or len(right) < 6:
                prefix_map[left] = right
                prefix_map[right] = left

            # 3) Regular single-word antonyms
            else:
                whole_word_map[left] = right
                whole_word_map[right] = left

    return whole_word_map, prefix_map, multi_word_map

def extract_labels_and_synonyms(ontology):
    """
    Returns a dict of HPO IDs -> set of labels + synonyms.
    Excludes synonyms with scope == 'RELATED'.
    """
    term_dict = {}
    for term in ontology.terms():
        labels = set()
        if term.name:
            labels.add(term.name.lower())
        for syn in term.synonyms:
            if syn.scope != 'RELATED':
                labels.add(syn.description.lower())
        term_dict[term.id] = labels
    return term_dict

def build_label_index(term_dict):
    """
    Build a reverse index: label -> set of term_ids with that label.
    """
    label_to_ids = {}
    for term_id, labels in term_dict.items():
        for lbl in labels:
            label_to_ids.setdefault(lbl, set()).add(term_id)
    return label_to_ids

def morphological_swaps(label):
    """
    Applies known morphological prefix swaps to 'label'.
    Returns a set of all distinct transformations or empty set if no rule applies.

    Example:
      'hypertonia' -> {'hypotonia'}
      'intraarticular' -> {'extraarticular'}
      'bradycardia' -> {'tachycardia'}
    """
    results = set()
    original_label = label.lower()

    for pattern, replacement in MORPH_RULES:
        swapped = pattern.sub(replacement, original_label)
        if swapped != original_label:
            # Standardize to lowercase and strip spaces
            results.add(swapped.strip().lower())

    return results

def generate_opposite_labels(label, whole_word_map, prefix_map, multi_word_map):
    """
    1) Check multi-word exact replacements
    2) Perform morphological swaps
    3) Single-word or prefix-based replacements
    Returns a sorted list of candidate opposite strings.
    """
    label_l = label.lower()
    opposite_labels = set()

    # 1) Multi-word exact match
    if label_l in multi_word_map:
        opposite_labels.add(multi_word_map[label_l])

    # 2) Morphological swaps
    morph_swapped_set = morphological_swaps(label_l)
    if morph_swapped_set:
        opposite_labels.update(morph_swapped_set)

    # 3) Single-word or prefix-based
    parts = re.findall(r'\w+|\W+', label_l)
    for i, token in enumerate(parts):
        if token.isalnum():
            # Whole-word mapping
            if token in whole_word_map:
                new_parts = parts.copy()
                new_parts[i] = whole_word_map[token]
                opposite_labels.add(''.join(new_parts).strip())

            # Prefix-based
            for prefix, opp_prefix in prefix_map.items():
                if token.startswith(prefix):
                    new_parts = parts.copy()
                    new_parts[i] = token.replace(prefix, opp_prefix, 1)
                    opposite_labels.add(''.join(new_parts).strip())

    return sorted(opposite_labels)

def find_opposites_text_matching(ontology_path, antonyms_csv, output_file):
    """
    Detect opposite terms in the HPO using text-matching based on:
      - multi-word mappings
      - morphological swaps
      - whole-word or prefix-based replacements
    Writes results to output_file.
    """
    whole_word_map, prefix_map, multi_word_map = load_antonyms(antonyms_csv)
    ontology = Ontology(ontology_path)
    term_dict = extract_labels_and_synonyms(ontology)
    id_to_canonical_label = {t.id: t.name for t in ontology.terms()}
    label_to_ids = build_label_index(term_dict)

    opposites = []
    seen_pairs = set()
    term_ids = sorted(term_dict.keys())

    # Check each term's labels for potential antonyms
    for term_id in term_ids:
        for label in term_dict[term_id]:
            candidates = generate_opposite_labels(label, whole_word_map, prefix_map, multi_word_map)
            for opp_label in candidates:
                # See if the opposite label appears in other terms
                if opp_label in label_to_ids:
                    for other_id in label_to_ids[opp_label]:
                        if other_id != term_id:
                            pair = tuple(sorted((term_id, other_id)))
                            if pair not in seen_pairs:
                                seen_pairs.add(pair)
                                opposites.append((term_id, label, other_id, opp_label))

    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("hpo_id1\thpo_term1\thpo_id2\thpo_term2\n")
        for t1, lbl1, t2, lbl2 in sorted(opposites):
            name1 = id_to_canonical_label.get(t1, "N/A")
            name2 = id_to_canonical_label.get(t2, "N/A")
            f.write(f"{t1}\t{name1}\t{t2}\t{name2}\n")

if __name__ == "__main__":
    hp_obo = os.path.join(INPUT_DIR, "hp.obo")
    antonyms_csv = os.path.join("antonyms", "text-patterns.txt")
    output_file = os.path.join(OUTPUT_DIR, "hpo_opposites_text.tsv")

    find_opposites_text_matching(hp_obo, antonyms_csv, output_file)