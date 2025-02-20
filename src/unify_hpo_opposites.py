#!/usr/bin/env python3
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def canonicalize_row(row):
    id1, term1, id2, term2 = row['id1'], row['term1'], row['id2'], row['term2']
    if id1 > id2:
        row['id1'], row['term1'] = id2, term2
        row['id2'], row['term2'] = id1, term1
    return row

def combine_flags(group):
    text_flag = 't' if (group['text'] == 't').any() else 'f'
    logical_flag = 't' if (group['logical'] == 't').any() else 'f'
    row = group.iloc[0].copy()
    row['text'] = text_flag
    row['logical'] = logical_flag
    return row

def main():
    text_file = os.path.join(OUTPUT_DIR, "hpo_opposites_text.tsv")
    logical_file = os.path.join(OUTPUT_DIR, "hpo_opposites_logical.tsv")
    output_file = os.path.join(OUTPUT_DIR, "hpo_opposites_unified.tsv")

    # Read text-based file and assign flags: text = 't', logical = 'f'
    df_text = pd.read_csv(text_file, sep='\t')
    df_text['text'] = 't'
    df_text['logical'] = 'f'
    
    # Read logical file and assign flags: text = 'f', logical = 't'
    df_logical = pd.read_csv(logical_file, sep='\t')
    df_logical['text'] = 'f'
    df_logical['logical'] = 't'

    # Rename columns
    column_rename_map = {
        'hpo_id1': 'id1',
        'hpo_term1': 'term1',
        'hpo_id2': 'id2',
        'hpo_term2': 'term2'
    }
    df_text.rename(columns=column_rename_map, inplace=True)
    df_logical.rename(columns=column_rename_map, inplace=True)

    # Combine both dataframes
    df_combined = pd.concat([df_text, df_logical], ignore_index=True)
    
    # Canonicalize rows so that each pair is consistently ordered
    df_combined = df_combined.apply(canonicalize_row, axis=1)

    # Group by the full canonical tuple to keep distinct term strings
    group_cols = ['id1', 'term1', 'id2', 'term2']
    df_unified = df_combined.groupby(group_cols, as_index=False).apply(combine_flags)
    df_unified.reset_index(drop=True, inplace=True)
    
    # Reorder columns to new format
    column_order = ['id1', 'id2', 'logical', 'text', 'term1', 'term2']
    df_unified = df_unified[column_order]
    
    # Write out the unified output
    df_unified.to_csv(output_file, sep='\t', index=False)
    print(f"[INFO] Wrote {len(df_unified)} unified pairs to {output_file}")

if __name__ == '__main__':
    main()
