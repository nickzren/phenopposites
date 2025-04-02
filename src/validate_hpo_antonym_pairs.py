import os
import pandas as pd
from dotenv import load_dotenv

def canonicalize_pair(id1, id2):
    return tuple(sorted([id1.strip(), id2.strip()]))

def main():
    load_dotenv()
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    unified_file = os.path.join(OUTPUT_DIR, "hpo_opposites_unified.csv")
    antonyms_file = os.path.join("opposites", "antonyms_HP.csv")
    
    # Read the unified file (TSV)
    df_unified = pd.read_csv(unified_file, dtype=str)
    # Read the phenopposite antonyms CSV file
    df_antonyms = pd.read_csv(antonyms_file, dtype=str)
    
    # Generate canonical pairs from unified file
    unified_pairs = set(df_unified.apply(lambda row: canonicalize_pair(row['id1'], row['id2']), axis=1))
    
    # Generate canonical pairs from antonyms_HP.csv
    antonym_pairs = set(df_antonyms.apply(lambda row: canonicalize_pair(row['id1'], row['id2']), axis=1))
    
    # Find pairs in antonym file that are missing from unified file
    missing_pairs = antonym_pairs - unified_pairs
    
    # Split missing pairs into categories
    only_logical = []
    only_text = []
    both_logical_text = []
    
    for _, row in df_antonyms.iterrows():
        pair = canonicalize_pair(row['id1'], row['id2'])
        if pair in missing_pairs:
            logical = row['logical'].strip().lower() == 't'
            text = row['text'].strip().lower() == 't'
            term1 = row['term1'].strip()
            term2 = row['term2'].strip()
            
            if logical and not text:
                only_logical.append((pair, term1, term2))
            elif text and not logical:
                only_text.append((pair, term1, term2))
            elif logical and text:
                both_logical_text.append((pair, term1, term2))
    
    # Print categorized results
    print("The following HPO id pairs from phenopposite_antonyms.csv do not exist in hpo_opposites_unified.tsv:")
    
    print("\n--- Only Logical (logical=True, text=False) ---")
    for pair, term1, term2 in only_logical:
        print(f"{pair}: {term1} ↔ {term2}")
    
    print("\n--- Only Text (logical=False, text=True) ---")
    for pair, term1, term2 in only_text:
        print(f"{pair}: {term1} ↔ {term2}")
    
    print("\n--- Both Logical and Text (logical=True, text=True) ---")
    for pair, term1, term2 in both_logical_text:
        print(f"{pair}: {term1} ↔ {term2}")
    
if __name__ == '__main__':
    main()
