#!/usr/bin/env python3
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")
VALIDATED_DIR = os.path.join(OUTPUT_DIR, "llm_validated")

TEXT_FILE = os.path.join(
    VALIDATED_DIR, "hpo_opposites_text_gemini_validated.csv"
)
LOGICAL_FILE = os.path.join(
    VALIDATED_DIR, "hpo_opposites_logical_gemini_validated.csv"
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hpo_opposites_unified.csv")


def _canonicalize(row: pd.Series) -> pd.Series:
    if row["id1"] > row["id2"]:
        row["id1"], row["id2"] = row["id2"], row["id1"]
        row["term1"], row["term2"] = row["term2"], row["term1"]
    return row


def _load(path: str, text_flag: str, logical_flag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["inverse_verified_by_llm"].str.lower() == "yes"].copy()
    df["text"] = text_flag
    df["logical"] = logical_flag
    return df.drop(columns=["inverse_verified_by_llm"], errors="ignore")


def main() -> None:
    rename = {
        "hpo_id1": "id1",
        "hpo_term1": "term1",
        "hpo_id2": "id2",
        "hpo_term2": "term2",
    }

    df_text = _load(TEXT_FILE, "t", "f").rename(columns=rename)
    df_logical = _load(LOGICAL_FILE, "f", "t").rename(columns=rename)

    combined = pd.concat([df_text, df_logical], ignore_index=True)
    combined = combined.apply(_canonicalize, axis=1)

    unified = (
        combined.groupby(["id1", "term1", "id2", "term2"], sort=False, as_index=False)
        .agg(
            logical=("logical", lambda s: "t" if (s == "t").any() else "f"),
            text=("text",    lambda s: "t" if (s == "t").any() else "f"),
        )
    )

    logical_total = (unified["logical"] == "t").sum()
    text_total = (unified["text"] == "t").sum()
    both_total = ((unified["logical"] == "t") & (unified["text"] == "t")).sum()

    unified[["id1", "id2", "logical", "text", "term1", "term2"]].to_csv(
        OUTPUT_FILE, index=False
    )

    print(f"[INFO] logical=t pairs: {logical_total}")
    print(f"[INFO] text=t pairs: {text_total}")
    print(f"[INFO] logical=t & text=t pairs: {both_total}")
    print(f"[INFO] Wrote {len(unified)} unified pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()