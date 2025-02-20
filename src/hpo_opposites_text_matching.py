#!/usr/bin/env python3
import os
import csv
import re
from dotenv import load_dotenv
from pronto import Ontology

load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def load_antonyms(csv_path):
    """
    Load antonym pairs from CSV into:
      1) whole_word_map      (for single-word pairs)
      2) prefix_map          (for short prefix swaps)
      3) multi_word_map      (for multi-word or hyphenated patterns)
      4) domain_map          (for domain-specific pairs that do not match via prefix or morphological rules)
    """
    whole_word_map = {}
    prefix_map = {}
    multi_word_map = {}
    domain_map = {}

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in sorted(reader):
            if len(row) == 2:
                left = row[0].strip().lower()
                right = row[1].strip().lower()

                # Decide if this is a single word, multi-word, or special domain phrase
                if ' ' in left or ' ' in right:
                    # If it's multi-word or domain-specific, we can store it in domain_map
                    # But if you want all multi-word pairs in multi_word_map, do that instead.
                    domain_map[left] = right
                    domain_map[right] = left
                elif len(left) < 6 or len(right) < 6:
                    prefix_map[left] = right
                    prefix_map[right] = left
                else:
                    whole_word_map[left] = right
                    whole_word_map[right] = left

    return whole_word_map, prefix_map, multi_word_map, domain_map

def extract_labels_and_synonyms(ontology):
    """
    Returns a dict of HPO IDs -> set of labels + synonyms (excluding 'RELATED').
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

def morphological_swaps(label):
    """
    Applies known morphological regex swaps for hyper/hypo, etc.
    Example: hemihypertrophy -> hemihypotrophy,
             hyperintensities -> hypointensities, etc.
    Return the transformed label or None if no swap applies.
    """
    new_label = label

    # Example: "hemihypertrophy" -> "hemihypotrophy"
    # This says: if we have (hemi)? hyper  then some letters, we replace hyper->hypo
    # Adjust or add more if you have other morphological patterns
    new_label = re.sub(r'\b(hemi)?hyper(\w*)\b', r'\1hypo\2', new_label)
    
    # If no difference, return None to indicate no morphological change
    if new_label == label:
        return None
    else:
        return new_label

def generate_opposite_labels(label, whole_word_map, prefix_map, multi_word_map, domain_map):
    """
    1) Check domain_map (exact phrase replacements)
    2) Morphological regex swaps
    3) Traditional single-word or prefix-based replacements
    """
    label_l = label.lower()
    opposite_labels = set()

    # 1) Domain-level exact phrase check
    #    If the entire label is in domain_map, do a direct swap
    if label_l in domain_map:
        opposite_labels.add(domain_map[label_l])

    # Alternatively, for multi-word domain patterns that might appear as substrings,
    # you'd scan through domain_map keys and see if they appear in the label.
    # For demonstration, we keep it simple and only handle an exact match.

    # 2) Morphological Swaps
    morph_swapped = morphological_swaps(label_l)
    if morph_swapped and morph_swapped != label_l:
        opposite_labels.add(morph_swapped)

    # 3) Token-based for single words / prefixes
    parts = re.findall(r'\w+|\W+', label_l)
    modified = False
    for i, token in enumerate(parts):
        if token.isalnum():
            if token in whole_word_map:
                new_parts = parts.copy()
                new_parts[i] = whole_word_map[token]
                opposite_labels.add(''.join(new_parts).strip())

            for prefix, opp_prefix in prefix_map.items():
                if token.startswith(prefix):
                    new_parts = parts.copy()
                    new_parts[i] = token.replace(prefix, opp_prefix, 1)
                    opposite_labels.add(''.join(new_parts).strip())

    return sorted(opposite_labels)

def find_opposites_text_matching(ontology_path, antonyms_csv, output_file):
    """
    Main function to detect opposite terms in HPO via text matching.
    """
    # Load antonym maps
    whole_word_map, prefix_map, multi_word_map, domain_map = load_antonyms(antonyms_csv)

    # Load ontology
    ontology = Ontology(ontology_path)
    term_dict = extract_labels_and_synonyms(ontology)
    id_to_canonical_label = {term.id: term.name for term in ontology.terms()}

    opposites = []
    # For stable iteration order
    term_ids = sorted(term_dict.keys())
    for term_id in term_ids:
        labels = sorted(term_dict[term_id])
        for label in labels:
            candidates = generate_opposite_labels(label, whole_word_map, prefix_map, multi_word_map, domain_map)
            if not candidates:
                continue
            # See if any candidate appears in other terms
            for opp_label in candidates:
                for other_id in term_ids:
                    if other_id == term_id:
                        continue
                    if opp_label in term_dict[other_id]:
                        opposites.append((term_id, label, other_id, opp_label))

    # Write output
    seen_pairs = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("hpo_id1\thpo_term1\thpo_id2\thpo_term2\n")
        for t1, lbl1, t2, lbl2 in sorted(opposites):
            pair = tuple(sorted([t1, t2]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                name1 = id_to_canonical_label.get(t1, "N/A")
                name2 = id_to_canonical_label.get(t2, "N/A")
                f.write(f"{t1}\t{name1}\t{t2}\t{name2}\n")

if __name__ == "__main__":
    hp_obo = os.path.join(INPUT_DIR, "hp.obo")
    antonyms_csv = os.path.join("antonyms", "text-patterns.txt")
    output_file = os.path.join(OUTPUT_DIR, "hpo_opposites_text.tsv")
    find_opposites_text_matching(hp_obo, antonyms_csv, output_file)