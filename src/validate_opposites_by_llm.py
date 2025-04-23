#!/usr/bin/env python3
"""
validate_opposites_by_llm.py
-------------------------------------------------
Validates inverse (Opposite‑of) term pairs with
multiple LLM providers and writes one result
file per provider to:

    data/output/llm_validated/<basename><SUFFIX>.csv
"""

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple

import boto3
from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes
from openai import OpenAI, OpenAIError, RateLimitError, Timeout
from together import Together
from tqdm import tqdm

load_dotenv()

PROVIDERS = ("openai", "deepseek", "aws", "together", "google")
FILE_SUFFIX = {
    "openai": "_openai_validated.csv",
    "deepseek": "_deepseek_validated.csv",
    "aws": "_claude_validated.csv",
    "together": "_llama_validated.csv",
    "google": "_gemini_validated.csv",
}
YES_NO = re.compile(r"\b(?:yes|no)\b", re.I)


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify inverse term pairs with an LLM")
    p.add_argument("--input_file", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--llm_provider", choices=PROVIDERS, default="openai")
    return p.parse_args()


def output_path(src: str, provider: str) -> str:
    base, _ = os.path.splitext(os.path.basename(src))
    out_dir = os.path.join("data", "output", "llm_validated")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base + FILE_SUFFIX[provider])


# --------------------------------------------------------------------------- #
#  Prompt helpers & CSV utilities                                             #
# --------------------------------------------------------------------------- #
def file_type_from_name(path: str) -> str:
    name = os.path.basename(path).lower()
    if "mondo" in name:
        return "mondo"
    if "hpo" in name:
        return "hpo"
    raise ValueError("Unable to infer file type (hpo|mondo) from file name")


def get_term_columns(headers: list[str], ftype: str) -> Tuple[str, str]:
    for prefix in (f"{ftype}_", ""):
        t1, t2 = f"{prefix}term1", f"{prefix}term2"
        if t1 in headers and t2 in headers:
            return t1, t2

    t1_candidates = [h for h in headers if h.endswith("term1")]
    t2_candidates = [h for h in headers if h.endswith("term2")]
    if len(t1_candidates) == 1 and len(t2_candidates) == 1:
        return t1_candidates[0], t2_candidates[0]

    raise ValueError("Could not determine term1/term2 columns")


def make_prompt(t1: str, t2: str, ftype: str) -> str:
    term_type = "human phenotype" if ftype == "hpo" else "human disease"
    return (
        f"Determine if these two {term_type} terms represent an inverse "
        f"(Opposite‑of) relationship.\nReply with exactly one word — yes or no.\n\n"
        f"Term 1: {t1}\nTerm 2: {t2}"
    )


# --------------------------------------------------------------------------- #
#  Client factories                                                           #
# --------------------------------------------------------------------------- #
def _missing(var: str) -> None:
    raise RuntimeError(f"{var} missing in .env")


def get_client(provider: str) -> Tuple[Any, str]:
    if provider == "openai":
        return (
            OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or _missing("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            ),
            os.getenv("OPENAI_API_MODEL_ID") or _missing("OPENAI_API_MODEL_ID"),
        )
    if provider == "deepseek":
        return (
            OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY") or _missing("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_API_BASE_URL"),
            ),
            os.getenv("DEEPSEEK_API_MODEL_ID") or _missing("DEEPSEEK_API_MODEL_ID"),
        )
    if provider == "aws":
        return (
            boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION")),
            os.getenv("AWS_CLAUDE_MODEL_ID") or _missing("AWS_CLAUDE_MODEL_ID"),
        )
    if provider == "together":
        return (
            Together(api_key=os.getenv("TOGETHER_API_KEY") or _missing("TOGETHER_API_KEY")),
            os.getenv("TOGETHER_LLAMA_MODEL_ID") or _missing("TOGETHER_LLAMA_MODEL_ID"),
        )
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or _missing("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key), os.getenv("GOOGLE_MODEL_ID") or _missing("GOOGLE_MODEL_ID")
    raise ValueError("Unsupported provider")


# --------------------------------------------------------------------------- #
#  LLM call                                                                   #
# --------------------------------------------------------------------------- #
def ask_llm(
    client, prompt: str, model_id: str, provider: str, retries: int = 4, backoff: int = 2
) -> str:
    for attempt in range(retries):
        try:
            if provider == "google":
                cfg = gtypes.GenerateContentConfig(
                    max_output_tokens=5, temperature=0.0, thinking_config=gtypes.ThinkingConfig(thinking_budget=0)
                )
                txt = client.models.generate_content(model=model_id, contents=[prompt], config=cfg).text
            elif provider == "aws":
                body = json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 5,
                        "temperature": 0,
                        "top_p": 1,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                    }
                )
                txt = json.loads(
                    client.invoke_model(
                        modelId=model_id, contentType="application/json", accept="application/json", body=body
                    )["body"].read()
                )["content"][0]["text"]
            else:  # OpenAI‑compatible
                txt = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                    top_p=1,
                ).choices[0].message.content

            if m := YES_NO.search(txt.lower()):
                return m.group(0)
            time.sleep(backoff)
        except (RateLimitError, Timeout, OpenAIError):
            time.sleep(backoff * (2**attempt))
        except Exception:
            time.sleep(backoff)
    return "N/A"


# --------------------------------------------------------------------------- #
#  CSV driver                                                                 #
# --------------------------------------------------------------------------- #
def process_csv(infile: str, outfile: str, client, model_id: str, provider: str, workers: int):
    with open(infile, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        ftype = file_type_from_name(infile)
        term1_col, term2_col = get_term_columns(rdr.fieldnames, ftype)

    prompts = [make_prompt(r[term1_col], r[term2_col], ftype) for r in rows]
    results = ["N/A"] * len(prompts)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(ask_llm, client, p, model_id, provider): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futs), total=len(prompts), desc=f"{provider} queries"):
            results[futs[fut]] = fut.result()

    fieldnames = list(rows[0].keys()) + ["inverse_verified_by_llm"]
    with open(outfile, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, res in zip(rows, results):
            row["inverse_verified_by_llm"] = res
            writer.writerow(row)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    client, model_id = get_client(args.llm_provider)
    out_file = output_path(args.input_file, args.llm_provider)
    process_csv(args.input_file, out_file, client, model_id, args.llm_provider, args.workers)


if __name__ == "__main__":
    main()