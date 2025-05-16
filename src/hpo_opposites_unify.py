#!/usr/bin/env python3
"""
unify_hpo_votes.py

• hpo_opposites_text_unified.csv
• hpo_opposites_logical_unified.csv
  – every row, one column per *present* provider + pct_yes + majority

• hpo_opposites_unified.csv
  – keep a pair when majority(text) **or** majority(logical) is true

Default majority cutoff is ≥ 50 %, override with --cutoff 0.6, etc.
"""

import os, re, csv, argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OUT_DIR = os.getenv("OUTPUT_DIR", "data/output")
VAL_DIR = os.path.join(OUT_DIR, "llm_validated")

LLM = ["openai", "deepseek", "claude", "llama", "gemini", "grok", "qwen"]

TEXT_RE  = re.compile(r"hpo_opposites_text_.*?_validated\.csv$")
LOGIC_RE = re.compile(r"hpo_opposites_logical_.*?_validated\.csv$")

Pair = Tuple[str, str, str, str]


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cutoff", type=float, default=0.5,
                   help="majority threshold (default: 0.5)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def canonical(id1: str, term1: str, id2: str, term2: str) -> Pair:
    return (id2, term2, id1, term1) if id1 > id2 else (id1, term1, id2, term2)


def add_votes(
    path: str,
    prov: str,
    bucket: Dict[Pair, Dict[str, str]],
    extra_data: Dict[Pair, Dict[str, Any]],
):
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        pair = canonical(
            r.get("hpo_id1") or r.get("id1"),
            r.get("hpo_term1") or r.get("term1"),
            r.get("hpo_id2") or r.get("id2"),
            r.get("hpo_term2") or r.get("term2"),
        )
        ans = str(r["inverse_verified_by_llm"]).strip().lower()
        bucket[pair][prov] = "yes" if ans == "yes" else "no"
        # store the full original row so we can propagate extra columns (e.g. "strict")
        extra_data.setdefault(pair, {}).update(r.to_dict())


def votes_table(pattern: re.Pattern, outfile: str, cutoff: float, include_strict: bool = False) -> Dict[Pair, bool]:
    bucket: Dict[Pair, Dict[str, str]] = defaultdict(dict)
    present: List[str] = []
    extra_data: Dict[Pair, Dict[str, Any]] = {}

    for prov in LLM:
        suff = f"_{prov}_validated.csv"
        for fn in os.listdir(VAL_DIR):
            if pattern.match(fn) and fn.endswith(suff):
                present.append(prov)
                add_votes(os.path.join(VAL_DIR, fn), prov, bucket, extra_data)
                break

    present.sort()

    rows: List[Dict[str, str]] = []
    majority: Dict[Pair, bool] = {}

    for (id1, t1, id2, t2), votes in bucket.items():
        ans_cnt = len(votes)
        yes_cnt = sum(v == "yes" for v in votes.values())
        pct_yes = 0.0 if ans_cnt == 0 else yes_cnt / ans_cnt
        maj = pct_yes >= cutoff
        majority[(id1, t1, id2, t2)] = maj

        row = dict(
            id1=id1, term1=t1, id2=id2, term2=t2,
            pct_yes=f"{pct_yes:.2f}",
            majority="t" if maj else "f",
        )
        for prov in present:
            row[prov] = votes.get(prov, "N/A")
        if include_strict:
            row["strict"] = extra_data.get((id1, t1, id2, t2), {}).get("strict", "N/A")
        rows.append(row)

    cols = ["id1", "term1", "id2", "term2"]
    if include_strict:
        cols.append("strict")
    cols += present + ["pct_yes", "majority"]
    out_path = os.path.join(OUT_DIR, outfile)
    pd.DataFrame(rows)[cols].to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[INFO] wrote {len(rows)} rows → {out_path}")
    return majority


def write_master(text_pass: Dict[Pair, bool], logic_pass: Dict[Pair, bool], out_csv: str):
    keep = {p for p, ok in text_pass.items() if ok} | {p for p, ok in logic_pass.items() if ok}

    rows: List[Dict[str, str]] = []
    for p in keep:
        id1, t1, id2, t2 = p
        rows.append(
            dict(
                id1=id1, term1=t1,
                id2=id2, term2=t2,
                text="t" if text_pass.get(p, False) else "f",
                logical="t" if logic_pass.get(p, False) else "f",
            )
        )

    df = pd.DataFrame(rows)[["id1", "id2", "logical", "text", "term1", "term2"]]
    df.to_csv(out_csv, index=False)

    print(f"[INFO] logical=t pairs: {(df.logical=='t').sum()}")
    print(f"[INFO] text=t pairs   : {(df.text=='t').sum()}")
    print(f"[INFO] logical & text : {((df.logical=='t') & (df.text=='t')).sum()}")
    print(f"[INFO] wrote {len(df)} unified rows → {out_csv}")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    text_pass  = votes_table(TEXT_RE,  "hpo_opposites_text_unified.csv",    args.cutoff, include_strict=False)
    logic_pass = votes_table(LOGIC_RE, "hpo_opposites_logical_unified.csv", args.cutoff, include_strict=True)
    write_master(
        text_pass, logic_pass,
        os.path.join(OUT_DIR, "hpo_opposites_unified.csv")
    )

if __name__ == "__main__":
    main()