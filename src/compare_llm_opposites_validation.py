#!/usr/bin/env python3
"""
compare_llm_opposites_validation.py

Reads *_validated.csv files in data/output/llm_validated/,
merges them, writes a *_disagreements_all_llms.csv for every
(term_type, category) found, and reports each provider’s
agreement rate against a user-specified base provider
(default: openai).
"""

import argparse
import glob
import os
import sys
import pandas as pd

PROVIDERS = ["openai", "aws", "deepseek", "together", "google"]
FILE_SUFFIX = {
    "openai":   "_openai_validated.csv",
    "deepseek": "_deepseek_validated.csv",
    "aws":      "_claude_validated.csv",
    "together": "_llama_validated.csv",
    "google":   "_gemini_validated.csv",
}


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare yes/no validation results across LLM providers."
    )
    p.add_argument(
        "--input_dir",
        default=os.path.join("data", "output", "llm_validated"),
        help="Dir containing *_validated.csv files (default: %(default)s)",
    )
    p.add_argument(
        "-b",
        "--base_provider",
        choices=PROVIDERS,
        default="openai",
        help="Provider to use as the agreement baseline (default: %(default)s)",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def provider_file(input_dir: str, term_type: str, category: str, provider: str) -> str:
    return os.path.join(
        input_dir, f"{term_type}_opposites_{category}{FILE_SUFFIX[provider]}"
    )


def standardize_columns(df: pd.DataFrame, term_type: str) -> pd.DataFrame:
    rename_map = {
        "id1": f"{term_type}_id1",
        "id2": f"{term_type}_id2",
        "term1": f"{term_type}_term1",
        "term2": f"{term_type}_term2",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def load_df(path: str, term_type: str, provider: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_columns(df, term_type)
    return df.rename(columns={"inverse_verified_by_llm": f"inverse_verified_by_llm_{provider}"})


def merge_providers(
    input_dir: str, term_type: str, category: str
) -> tuple[pd.DataFrame, list[str]]:
    join_cols = [
        f"{term_type}_id1",
        f"{term_type}_term1",
        f"{term_type}_id2",
        f"{term_type}_term2",
    ]

    dfs, used = [], []
    for provider in PROVIDERS:
        path = provider_file(input_dir, term_type, category, provider)
        if os.path.exists(path):
            df = load_df(path, term_type, provider)
            dfs.append(df[join_cols + [f"inverse_verified_by_llm_{provider}"]])
            used.append(provider)

    if not dfs:
        raise FileNotFoundError(f"No provider files found for {term_type}-{category}")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=join_cols)

    return merged, used


def agreement_stats(
    df: pd.DataFrame, providers_used: list[str], base: str, term_type: str, category: str
) -> None:
    if base not in providers_used:
        print(
            f"[!] Skipping agreement stats for {term_type}-{category}: base provider '{base}' not present",
            file=sys.stderr,
        )
        return
    total = len(df)
    base_col = f"inverse_verified_by_llm_{base}"
    print(f"\n{term_type.upper()}-{category.upper()} – agreement with {base.upper()} ({total:,} rows)")
    for p in providers_used:
        if p == base:
            continue
        col = f"inverse_verified_by_llm_{p}"
        matches = (df[base_col] == df[col]).sum()
        pct = matches / total * 100
        print(f"  {base.capitalize()} vs {p:<9}: {pct:6.2f}%  ({matches}/{total})")


def process_category(
    input_dir: str, term_type: str, category: str, base: str
) -> None:
    try:
        merged, providers_used = merge_providers(input_dir, term_type, category)
    except FileNotFoundError as e:
        print(f"[!] Skipping {term_type}-{category}: {e}", file=sys.stderr)
        return

    verdict_cols = [f"inverse_verified_by_llm_{p}" for p in providers_used]
    disagreements = merged[merged[verdict_cols].nunique(axis=1) > 1]

    out_path = os.path.join(
        input_dir, f"{term_type}_{category}_disagreements_all_llms.csv"
    )
    disagreements.to_csv(out_path, index=False)
    print(f"[✓] {term_type.upper()}-{category.upper()}: {len(disagreements)} rows → {out_path}")

    agreement_stats(merged, providers_used, base, term_type, category)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    base = args.base_provider

    term_types = ("hpo", "mondo")
    categories_for = lambda tt: {
        os.path.basename(f).split("_opposites_")[1].split(FILE_SUFFIX[base])[0]
        for f in glob.glob(
            os.path.join(args.input_dir, f"{tt}_opposites_*{FILE_SUFFIX[base]}")
        )
    }

    for t in term_types:
        for c in sorted(categories_for(t)):
            process_category(args.input_dir, t, c, base)


if __name__ == "__main__":
    main()