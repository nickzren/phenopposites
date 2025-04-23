#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
if not OUTPUT_DIR:
    raise RuntimeError("Environment variable OUTPUT_DIR is not set.")

OUTPUT_FILE = Path(OUTPUT_DIR) / "hpo_opposites_verified.csv"
KEEP_COLS = ["id1", "id2", "logical", "text", "inherit", "term1", "term2"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to validated CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, dtype=str)
    df = df[df["inverse_verified_by_llm"].str.lower() == "yes"][KEEP_COLS]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()