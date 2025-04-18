#!/usr/bin/env python3
"""
compare_llm_opposites_validation.py

Reads the *_validated.csv files in **data/output/llm_validated/**,
merges them, writes a *_disagreements_all_llms.csv* for each term type,
and reports the agreement rate of every provider against OpenAI.
"""

import argparse
import os
import pandas as pd

PROVIDERS = ["openai", "aws", "deepseek", "together", "google"]
FILE_SUFFIX = {
    "openai": "_openai_validated.csv",
    "deepseek": "_deepseek_validated.csv",
    "aws": "_claude_validated.csv",
    "together": "_llama_validated.csv",
    "google": "_gemini_validated.csv",
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
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def load_df(input_dir: str, term_type: str, provider: str) -> pd.DataFrame:
    path = os.path.join(input_dir, f"{term_type}_opposites_text{FILE_SUFFIX[provider]}")
    df = pd.read_csv(path)
    return df.rename(
        columns={"inverse_verified_by_llm": f"inverse_verified_by_llm_{provider}"}
    )


def merge_providers(input_dir: str, term_type: str) -> pd.DataFrame:
    dfs = [load_df(input_dir, term_type, p) for p in PROVIDERS]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df,
            on=[
                f"{term_type}_id1",
                f"{term_type}_term1",
                f"{term_type}_id2",
                f"{term_type}_term2",
            ],
        )
    return merged


def agreement_stats(df: pd.DataFrame, term_type: str) -> None:
    total = len(df)
    base_col = "inverse_verified_by_llm_openai"
    print(f"\n{term_type.upper()} – agreement with OpenAI ({total:,} rows)")
    for p in PROVIDERS[1:]:
        col = f"inverse_verified_by_llm_{p}"
        matches = (df[base_col] == df[col]).sum()
        pct = matches / total * 100
        print(f"  OpenAI vs {p:<9}: {pct:6.2f}%  ({matches}/{total})")


# --------------------------------------------------------------------------- #
#  Per‑type processing                                                        #
# --------------------------------------------------------------------------- #
def process_type(input_dir: str, term_type: str) -> None:
    merged = merge_providers(input_dir, term_type)

    # write disagreement file
    verdict_cols = [f"inverse_verified_by_llm_{p}" for p in PROVIDERS]
    disagreements = merged[merged[verdict_cols].nunique(axis=1) > 1]
    out_path = os.path.join(input_dir, f"{term_type}_disagreements_all_llms.csv")
    disagreements.to_csv(out_path, index=False)
    print(f"[✓] {term_type.upper()}: {len(disagreements)} disagreement rows → {out_path}")

    # print agreement percentages
    agreement_stats(merged, term_type)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    for t in ("hpo", "mondo"):
        process_type(args.input_dir, t)


if __name__ == "__main__":
    main()