#!/usr/bin/env python3
"""
compare_llm_opposites_validation.py
Report pair-wise agreement between LLM validations
"""

import argparse, glob, os, sys
import pandas as pd

LLM = ["openai", "deepseek", "claude", "llama", "gemini", "grok", "qwen"]

SEP = "  "

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/output/llm_validated")
    p.add_argument("-b", "--base_llm", choices=LLM, default="openai")
    return p.parse_args()

def provider_file(indir: str, ttype: str, cat: str, prov: str) -> str:
    # Not used anymore, but left for compatibility
    suffix = f"_{prov}_validated.csv"
    return os.path.join(indir, f"{ttype}_opposites_{cat}{suffix}")

def standardize(df: pd.DataFrame, ttype: str) -> pd.DataFrame:
    return df.rename(columns={
        "id1":   f"{ttype}_id1",
        "id2":   f"{ttype}_id2",
        "term1": f"{ttype}_term1",
        "term2": f"{ttype}_term2",
    })

def load_df(path: str, ttype: str, prov: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize(df, ttype)
    return df.rename(columns={"inverse_verified_by_llm": f"inverse_verified_by_llm_{prov}"})

def merge(indir: str, ttype: str, cat: str) -> tuple[pd.DataFrame, list[str]]:
    join = [f"{ttype}_id1", f"{ttype}_term1", f"{ttype}_id2", f"{ttype}_term2"]
    dfs, used = [], []
    for p in LLM:
        suffix = f"_{p}_validated.csv"
        fp = os.path.join(indir, f"{ttype}_opposites_{cat}{suffix}")
        if os.path.exists(fp):
            dfs.append(load_df(fp, ttype, p)[join + [f"inverse_verified_by_llm_{p}"]])
            used.append(p)
    if not dfs:
        raise FileNotFoundError
    m = dfs[0]
    for d in dfs[1:]:
        m = m.merge(d, on=join)
    return m, used

def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d * 100.0

def norm(s: pd.Series) -> pd.Series:
    return s.fillna("N/A").astype(str).str.strip().str.lower()

def cell(match: int, denom: int) -> str:
    return f"{pct(match, denom):6.2f}% ({match}/{denom})" if denom else "N/A"

def agreement(df: pd.DataFrame, used: list[str], base: str, ttype: str, cat: str):
    base_col = f"inverse_verified_by_llm_{base}"
    df[base_col] = norm(df[base_col])
    total = len(df)
    yes_mask, no_mask = df[base_col] == "yes", df[base_col] == "no"
    yes_tot, no_tot = yes_mask.sum(), no_mask.sum()

    rows = []
    for p in used:
        if p == base:
            continue
        col = f"inverse_verified_by_llm_{p}"
        df[col] = norm(df[col])
        rows.append((
            p,
            cell((df[col] == df[base_col]).sum(), total),
            cell(((df[col] == "yes") & yes_mask).sum(), yes_tot),
            cell(((df[col] == "no") & no_mask).sum(), no_tot),
        ))
    if not rows:
        return

    headers = ("LLM", "Overall", "Yes", "No")
    widths = [max(len(h), max(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)]
    header_line = SEP.join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    separator = "-" * (sum(widths) + len(SEP) * 3)

    print(f"\n{ttype.upper()}-{cat.upper()} – agreement with {base.upper()} ({total:,} rows)")
    print(header_line)
    print(separator)
    for r in rows:
        print(SEP.join(f"{r[i]:<{widths[i]}}" for i in range(4)))

def process(indir: str, ttype: str, cat: str, base: str):
    try:
        df, used = merge(indir, ttype, cat)
    except FileNotFoundError:
        print(f"[!] missing {ttype}-{cat}")
        return
    disagreements = df[df[[f"inverse_verified_by_llm_{p}" for p in used]].nunique(axis=1) > 1]
    disagreements.to_csv(os.path.join(indir, f"{ttype}_{cat}_disagreements_all_llms.csv"), index=False)
    agreement(df, used, base, ttype, cat)

def main():
    args = parse_args()
    for ttype in ("hpo", "mondo"):
        suffix = f"_{args.base_llm}_validated.csv"
        cats = {
            os.path.basename(fp).split("_opposites_")[1].split(suffix)[0]
            for fp in glob.glob(os.path.join(args.input_dir, f"{ttype}_opposites_*{suffix}"))
        }
        for cat in sorted(cats):
            process(args.input_dir, ttype, cat, args.base_llm)

if __name__ == "__main__":
    main()