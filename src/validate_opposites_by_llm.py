#!/usr/bin/env python3
"""
validate_opposites_by_llm.py

Supported providers
-------------------
openai         – any OpenAI‑compatible endpoint
deepseek       – DeepSeek’s OpenAI‑style API
aws            – Claude models on AWS Bedrock
together       – Together AI (Llama‑4 family, etc.)
google         – Google Gemini Flash

Required .env keys
------------------
OPENAI_API_KEY,        OPENAI_API_MODEL_ID
DEEPSEEK_API_KEY,      DEEPSEEK_API_MODEL_ID
AWS_REGION,            AWS_CLAUDE_MODEL_ID
TOGETHER_API_KEY,      TOGETHER_LLAMA_MODEL_ID
GOOGLE_API_KEY,        GOOGLE_MODEL_ID
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
from openai import OpenAI, OpenAIError, RateLimitError, Timeout
from together import Together
from tqdm import tqdm
from google import genai
from google.genai import types as gtypes

load_dotenv()

YES_NO = re.compile(r"\b(?:yes|no)\b", re.I)
PROVIDERS = ("openai", "deepseek", "aws", "together", "google")


# --------------------------------------------------------------------------- #
#  CLI helpers                                                                #
# --------------------------------------------------------------------------- #
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
        f"Determine if these two {term_type} terms represent an inverse "
        f"(Opposite‑of) relationship.\nReply with exactly one word — yes or no.\n\n"
        f"Term 1: {t1}\nTerm 2: {t2}"
    )


# --------------------------------------------------------------------------- #
#  Provider plumbing                                                          #
# --------------------------------------------------------------------------- #
def _missing(var: str) -> None:
    raise RuntimeError(f"{var} missing in .env")


def get_client(provider: str) -> Tuple[Any, str, str]:
    """Return (client object, model_id, provider_tag)."""

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

    if provider == "aws":
        model = os.getenv("AWS_CLAUDE_MODEL_ID") or _missing("AWS_CLAUDE_MODEL_ID")
        return (
            boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION")),
            model,
            provider,
        )

    if provider == "together":
        model = os.getenv("TOGETHER_LLAMA_MODEL_ID") or _missing(
            "TOGETHER_LLAMA_MODEL_ID"
        )
        return (
            Together(api_key=os.getenv("TOGETHER_API_KEY") or _missing("TOGETHER_API_KEY")),
            model,
            provider,
        )

    if provider == "google":
        model = os.getenv("GOOGLE_MODEL_ID") or _missing("GOOGLE_MODEL_ID")
        api_key = os.getenv("GOOGLE_API_KEY") or _missing("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key), model, provider

    raise ValueError("Unsupported provider")


# --------------------------------------------------------------------------- #
#  LLM dispatcher                                                             #
# --------------------------------------------------------------------------- #
def ask_llm(
    client,
    prompt: str,
    model_id: str,
    provider: str,
    retries: int = 4,
    backoff: int = 2,
) -> str:
    for attempt in range(retries):
        try:
            if provider == "google":
                cfg = gtypes.GenerateContentConfig(
                    max_output_tokens=5,
                    temperature=0.0,
                    thinking_config=gtypes.ThinkingConfig(thinking_budget=0),
                )
                txt = client.models.generate_content(
                    model=model_id, contents=[prompt], config=cfg
                ).text

            elif provider == "aws":
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

            else:  # OpenAI‑compatible (OpenAI / DeepSeek / Together)
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
def process_csv(
    infile: str, outfile: str, client, model_id: str, provider: str, workers: int
):
    with open(infile, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        ftype = detect_file_type(rdr.fieldnames)

    prompts = [make_prompt(r[f"{ftype}_term1"], r[f"{ftype}_term2"], ftype) for r in rows]
    results = ["N/A"] * len(prompts)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(ask_llm, client, p, model_id, provider): i
            for i, p in enumerate(prompts)
        }
        for fut in tqdm(as_completed(futs), total=len(prompts), desc="LLM queries"):
            results[futs[fut]] = fut.result()

    fieldnames = list(rows[0].keys()) + ["inverse_verified_by_llm"]
    with open(outfile, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row, res in zip(rows, results):
            row["inverse_verified_by_llm"] = res
            w.writerow(row)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
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