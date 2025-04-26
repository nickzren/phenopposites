#!/usr/bin/env python3
"""
extract_hpo_inherited_opposites.py

Aggregate every `hpo_opposites_inherited_*_validated.csv` under
<OUTPUT_DIR>/llm_validated/ and write two files:

1. hpo_opposites_inherited_verified_unified.csv
   • ALL rows from every provider
   • original columns: id1, term1, id2, term2, logical, text, inherit
   • + one column per *present* provider
   • + pct_yes  (float 0‒1)  + majority (t|f) – majority = pct_yes ≥ cutoff

2. hpo_opposites_verified.csv
   • rows where majority == t
   • only the original columns (no provider / pct columns)

Summary counts for logical=text=inherit='t' are printed for the
filtered file.

Example:
    python extract_hpo_inherited_opposites.py --cutoff 0.6
"""

import argparse, os, re, csv
from collections import defaultdict
from typing import Dict, Tuple, List

import pandas as pd
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()
OUT_DIR  = os.getenv("OUTPUT_DIR")
if not OUT_DIR:
    raise RuntimeError("OUTPUT_DIR env var not set")

VAL_DIR  = os.path.join(OUT_DIR, "llm_validated")
BASE_COLS = ["id1", "term1", "id2", "term2", "logical", "text", "inherit"]

PROV_SUFFIX = {
    "openai":   "_openai_validated.csv",
    "deepseek": "_deepseek_validated.csv",
    "claude":   "_claude_validated.csv",
    "llama":    "_llama_validated.csv",
    "gemini":   "_gemini_validated.csv",
    "grok":     "_grok_validated.csv",
}

INHERIT_RE = re.compile(r"hpo_opposites_inherited_.*?_validated\.csv$")
Pair = Tuple[str, str, str, str]   # canonicalised key


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def canonical(id1: str, term1: str, id2: str, term2: str) -> Pair:
    return (id2, term2, id1, term1) if id1 > id2 else (id1, term1, id2, term2)


def add_votes(csv_path: str, provider: str,
              vote_map: Dict[Pair, Dict[str, str]],
              base_rows: Dict[Pair, Dict[str, str]]):
    df = pd.read_csv(csv_path, dtype=str)
    for _, row in df.iterrows():
        pair = canonical(
            row.get("hpo_id1")   or row["id1"],
            row.get("hpo_term1") or row["term1"],
            row.get("hpo_id2")   or row["id2"],
            row.get("hpo_term2") or row["term2"],
        )
        verdict = str(row["inverse_verified_by_llm"]).strip().lower()
        vote_map[pair][provider] = "yes" if verdict == "yes" else "no"

        if pair not in base_rows:                 # preserve first-found source row
            base_rows[pair] = {c: row.get(c) or row.get(f"hpo_{c}") for c in BASE_COLS}


def build_tables(cutoff: float):
    vote_map:  Dict[Pair, Dict[str, str]] = defaultdict(dict)
    base_rows: Dict[Pair, Dict[str, str]] = {}
    present_prov: List[str] = []

    # ingest
    for prov, suff in PROV_SUFFIX.items():
        for fn in os.listdir(VAL_DIR):
            if INHERIT_RE.match(fn) and fn.endswith(suff):
                present_prov.append(prov)
                add_votes(os.path.join(VAL_DIR, fn), prov, vote_map, base_rows)
                break

    present_prov.sort()

    full_rows, pass_rows = [], []
    for pair, base in base_rows.items():
        answers = vote_map[pair]
        answered = len(answers)
        yes_cnt  = sum(v == "yes" for v in answers.values())
        pct_yes  = 0.0 if answered == 0 else yes_cnt / answered
        majority = pct_yes >= cutoff

        # unified row (original + provider answers + stats)
        full = base.copy()
        for prov in present_prov:
            full[prov] = answers.get(prov, "N/A")
        full["pct_yes"]  = f"{pct_yes:.2f}"
        full["majority"] = "t" if majority else "f"
        full_rows.append(full)

        if majority:
            pass_rows.append(base)     # keep original columns only

    return present_prov, full_rows, pass_rows


def write_csv(rows: List[Dict[str, str]], columns: List[str], fname: str):
    path = os.path.join(OUT_DIR, fname)
    pd.DataFrame(rows)[columns].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[INFO] wrote {len(rows)} rows → {path}")
    return path


# ────────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.5, help="majority threshold (default 0.5)")
    args = ap.parse_args()

    provs, full_rows, pass_rows = build_tables(args.cutoff)

    # unified file (all rows + extras)
    cols_full = BASE_COLS + provs + ["pct_yes", "majority"]
    write_csv(full_rows, cols_full, "hpo_opposites_inherited_verified_unified.csv")

    # verified (filtered, original columns only)
    path_verified = write_csv(pass_rows, BASE_COLS, "hpo_opposites_verified.csv")

    # summary counts on verified
    df_pass = pd.read_csv(path_verified, dtype=str)
    log_cnt  = (df_pass.logical == "t").sum() if "logical" in df_pass else 0
    text_cnt = (df_pass.text    == "t").sum() if "text"    in df_pass else 0
    inh_cnt  = (df_pass.inherit == "t").sum() if "inherit" in df_pass else 0
    print(f"[INFO] logical=t pairs: {log_cnt}")
    print(f"[INFO] text=t pairs   : {text_cnt}")
    print(f"[INFO] inherit=t pairs: {inh_cnt}")


if __name__ == "__main__":
    main()