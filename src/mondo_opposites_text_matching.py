#!/usr/bin/env python3
import os
import csv
import re
from dotenv import load_dotenv
from pronto import Ontology

load_dotenv()
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

MORPH_RULES = [
    (re.compile(r'\b(hemi)?hyper(\w*)\b', re.IGNORECASE), r'\1hypo\2'),
    (re.compile(r'\b(hemi)?hypo(\w*)\b', re.IGNORECASE), r'\1hyper\2'),
    (re.compile(r'\bmacro(\w*)\b', re.IGNORECASE), r'micro\1'),
    (re.compile(r'\bmicro(\w*)\b', re.IGNORECASE), r'macro\1'),
    (re.compile(r'\btachy(\w*)\b', re.IGNORECASE), r'brady\1'),
    (re.compile(r'\bbrady(\w*)\b', re.IGNORECASE), r'tachy\1'),
    (re.compile(r'\bhyperplastic(\w*)\b', re.IGNORECASE), r'hypoplastic\1'),
    (re.compile(r'\bhypoplastic(\w*)\b', re.IGNORECASE), r'hyperplastic\1'),
]

def load_antonyms(csv_path):
    whole_word_map, prefix_map = {}, {}
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 2:
                continue
            left, right = row[0].strip().lower(), row[1].strip().lower()
            if len(left) < 6 or len(right) < 6:
                prefix_map[left] = right
                prefix_map[right] = left
            else:
                whole_word_map[left] = right
                whole_word_map[right] = left
    return whole_word_map, prefix_map

def extract_labels_and_synonyms(ontology):
    term_dict = {}
    for term in ontology.terms():
        if not term.id.startswith("MONDO:"):
            continue
        labels = set()
        if term.name:
            labels.add(term.name.lower())
        for syn in term.synonyms:
            if syn.scope != 'RELATED':
                labels.add(syn.description.lower())
        term_dict[term.id] = labels
    return term_dict

def build_label_index(term_dict):
    label_to_ids = {}
    for term_id, labels in term_dict.items():
        for lbl in labels:
            label_to_ids.setdefault(lbl, set()).add(term_id)
    return label_to_ids

def morphological_swaps(label):
    results = set()
    original = label.lower()
    for pattern, replacement in MORPH_RULES:
        swapped = pattern.sub(replacement, original)
        if swapped != original:
            results.add(swapped.strip())
    return results

def generate_opposite_labels(label, whole_word_map, prefix_map):
    label_l = label.lower()
    opposites = {}
    for morph_label in morphological_swaps(label_l):
        opposites[morph_label] = "morphological"
    parts = re.findall(r'\w+|\W+', label_l)
    for i, token in enumerate(parts):
        if token.isalnum():
            if token in whole_word_map:
                new_parts = parts.copy()
                new_parts[i] = whole_word_map[token]
                opposites[''.join(new_parts).strip()] = "whole-word"
            for prefix, opp_prefix in prefix_map.items():
                if token.startswith(prefix):
                    new_parts = parts.copy()
                    new_parts[i] = token.replace(prefix, opp_prefix, 1)
                    opposites[''.join(new_parts).strip()] = "prefix-based"
    return opposites

def find_opposites_text_matching(ontology_path, antonyms_csv, output_file):
    whole_word_map, prefix_map = load_antonyms(antonyms_csv)
    ontology = Ontology(ontology_path)
    term_dict = extract_labels_and_synonyms(ontology)
    id_to_canonical_label = {t.id: t.name for t in ontology.terms()}
    label_to_ids = build_label_index(term_dict)
    opposites = []
    seen_pairs = set()
    term_ids = sorted(term_dict.keys())

    for term_id in term_ids:
        for label in term_dict[term_id]:
            candidates = generate_opposite_labels(label, whole_word_map, prefix_map)
            for opp_label, logic_source in candidates.items():
                if opp_label in label_to_ids:
                    for other_id in label_to_ids[opp_label]:
                        if other_id != term_id:
                            pair = tuple(sorted((term_id, other_id)))
                            if pair not in seen_pairs:
                                seen_pairs.add(pair)
                                name1 = id_to_canonical_label.get(term_id, "N/A")
                                name2 = id_to_canonical_label.get(other_id, "N/A")
                                opposites.append((term_id, name1, other_id, name2))

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["mondo_id1", "mondo_term1", "mondo_id2", "mondo_term2"])
        for t1, name1, t2, name2 in sorted(opposites):
            writer.writerow([t1, name1, t2, name2])

if __name__ == "__main__":
    mondo_obo = os.path.join(INPUT_DIR, "mondo.obo")
    antonyms_csv = os.path.join("antonyms", "text-patterns.txt")
    output_file = os.path.join(OUTPUT_DIR, "mondo_opposites_text.csv")
    find_opposites_text_matching(mondo_obo, antonyms_csv, output_file)