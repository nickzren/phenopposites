#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Any

import boto3
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, Timeout, OpenAIError
from together import Together
from tqdm import tqdm

load_dotenv()

YES_NO = re.compile(r"\b(?:yes|no)\b", re.I)
PROVIDERS = ("openai", "deepseek", "aws-claude", "together-llama")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify inverse term pairs with an LLM")
    p.add_argument("--input_file", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--llm_provider", choices=PROVIDERS, default="openai")
    return p.parse_args()

def output_path(src: str) -> str:
    base, ext = os.path.splitext(os.path.basename(src))
    out_dir = os.getenv("OUTPUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_llm_validated{ext}")

def detect_file_type(headers) -> str:
    if {"hpo_id1", "hpo_term1"} <= set(headers):
        return "hpo"
    if {"mondo_id1", "mondo_term1"} <= set(headers):
        return "mondo"
    raise ValueError("Unsupported CSV format")

def make_prompt(t1: str, t2: str, ftype: str) -> str:
    term_type = "human phenotype" if ftype == "hpo" else "human disease"
    return (
        f"Determine if these two {term_type} terms represent an inverse (Opposite‑of) "
        f"relationship.\nReply with exactly one word — yes or no.\n\n"
        f"Term 1: {t1}\nTerm 2: {t2}"
    )

def _missing(var: str) -> None:
    raise RuntimeError(f"{var} missing in .env")

def get_client(provider: str) -> Tuple[Any, str, str]:
    """Return (client, model_id, provider)."""
    if provider == "openai":
        model = os.getenv("OPENAI_API_MODEL_ID") or _missing("OPENAI_API_MODEL_ID")
        return (
            OpenAI(
                api_key=os.getenv("OPENAI_API_KEY") or _missing("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            ),
            model,
            provider,
        )

    if provider == "deepseek":
        model = os.getenv("DEEPSEEK_API_MODEL_ID") or _missing("DEEPSEEK_API_MODEL_ID")
        return (
            OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY") or _missing("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_API_BASE_URL"),
            ),
            model,
            provider,
        )

    if provider == "aws-claude":
        model = os.getenv("AWS_CLAUDE_MODEL_ID") or _missing("AWS_CLAUDE_MODEL_ID")
        return (
            boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION")),
            model,
            provider,
        )

    if provider == "together-llama":
        model = (
            os.getenv("TOGETHER_LLAMA_MODEL_ID") or _missing("TOGETHER_LLAMA_MODEL_ID")
        )
        return (Together(api_key=os.getenv("TOGETHER_API_KEY") or _missing(
            "TOGETHER_API_KEY")), model, provider)

    raise ValueError("Unsupported provider")

def ask_llm(
    client,
    prompt: str,
    model_id: str,
    provider: str,
    retries: int = 4,
    backoff: int = 2,
) -> str:
    """Return 'yes', 'no', or 'N/A'."""
    for attempt in range(retries):
        try:
            if provider == "aws-claude":
                body = json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 5,
                        "temperature": 0,
                        "top_p": 1,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                    }
                )
                txt = json.loads(
                    client.invoke_model(
                        modelId=model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=body,
                    )["body"].read()
                )["content"][0]["text"]
            else:  # OpenAI‑compatible providers
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

def process_csv(
    infile: str, outfile: str, client, model_id: str, provider: str, workers: int
):
    with open(infile, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        ftype = detect_file_type(reader.fieldnames)

    prompts = [
        make_prompt(r[f"{ftype}_term1"], r[f"{ftype}_term2"], ftype) for r in rows
    ]
    results = ["N/A"] * len(prompts)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(ask_llm, client, p, model_id, provider): i
            for i, p in enumerate(prompts)
        }
        for fut in tqdm(as_completed(futures), total=len(prompts), desc="LLM queries"):
            results[futures[fut]] = fut.result()

    fieldnames = list(rows[0].keys()) + ["inverse_verified_by_llm"]
    with open(outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row, res in zip(rows, results):
            row["inverse_verified_by_llm"] = res
            w.writerow(row)

def main() -> None:
    args = parse_args()
    client, model_id, provider = get_client(args.llm_provider)
    process_csv(
        args.input_file,
        output_path(args.input_file),
        client,
        model_id,
        provider,
        args.workers,
    )


if __name__ == "__main__":
    main()