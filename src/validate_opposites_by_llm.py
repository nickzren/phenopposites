#!/usr/bin/env python3
"""
validate_opposites_by_llm.py
Validate inverse (Opposite-of) term pairs with multiple LLMs and write one
result file per model to data/output/llm_validated/<basename><SUFFIX>.csv
"""

import argparse, csv, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Tuple

import boto3
from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError
from together import Together
from tqdm import tqdm

load_dotenv()
YES_NO = re.compile(r"\b(?:yes|no)\b", re.I)

SYSTEM_PROMPT = (
    "You will judge whether two HUMAN PHENOTYPE terms are inverse "
    "(Opposite-of) pairs. Inverse means they describe the SAME trait at "
    "opposite extremes. Reply with exactly one lower-case word: yes or no."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--llm", choices=["openai", "deepseek", "claude", "llama", "gemini", "grok", "qwen"], default="openai")
    a = parser.parse_args()
    return a


def output_path(src: str, llm: str) -> str:
    base, _ = os.path.splitext(os.path.basename(src))
    dst_dir = os.path.join("data", "output", "llm_validated")
    os.makedirs(dst_dir, exist_ok=True)
    suffix = f"_{llm}_validated.csv"
    return os.path.join(dst_dir, base + suffix)


def file_type_from_name(path: str) -> str:
    name = os.path.basename(path).lower()
    if "mondo" in name:
        return "mondo"
    if "hpo" in name:
        return "hpo"
    raise ValueError("Unable to infer file type")


def get_term_columns(headers: list[str], ftype: str) -> Tuple[str, str]:
    for prefix in (f"{ftype}_", ""):
        t1, t2 = f"{prefix}term1", f"{prefix}term2"
        if t1 in headers and t2 in headers:
            return t1, t2
    t1s = [h for h in headers if h.endswith("term1")]
    t2s = [h for h in headers if h.endswith("term2")]
    if len(t1s) == 1 and len(t2s) == 1:
        return t1s[0], t2s[0]
    raise ValueError("term1/term2 columns not found")


def make_user_prompt(t1: str, t2: str) -> str:
    return f"Term 1: {t1}\nTerm 2: {t2}"


def get_client(llm: str) -> Tuple[Any, str]:
    if llm == "openai":
        return (
            OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE_URL"),
            ),
            os.getenv("OPENAI_API_MODEL_ID"),
        )

    # DeepSeek (official API, OpenAI‑compatible)
    if llm == "deepseek":
        return (
            OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_API_BASE_URL"),
            ),
            os.getenv("DEEPSEEK_API_MODEL_ID"),
        )

    if llm in ("qwen", "llama"):
        return (
            Together(api_key=os.getenv("TOGETHER_API_KEY")),
            os.getenv(
                {
                    "qwen": "TOGETHER_QWEN_MODEL_ID",
                    "llama": "TOGETHER_LLAMA_MODEL_ID",
                }[llm]
            ),
        )

    if llm == "claude":
        return (
            boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION")),
            os.getenv("AWS_CLAUDE_MODEL_ID"),
        )

    if llm == "gemini":
        key = os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=key), os.getenv("GOOGLE_MODEL_ID")

    if llm == "grok":
        return (
            OpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url=os.getenv("XAI_API_BASE_URL"),
            ),
            os.getenv("XAI_API_MODEL_ID"),
        )

    raise ValueError("Unsupported LLM")


def ask_llm(client, user_prompt: str, model_id: str, llm: str, retries: int = 4, backoff: int = 2) -> str:
    for attempt in range(retries):
        try:
            if llm == "gemini":
                prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
                cfg = gtypes.GenerateContentConfig(temperature=0.0, thinking_config=gtypes.ThinkingConfig(thinking_budget=0))
                txt = client.models.generate_content(model=model_id, contents=[prompt], config=cfg).text
            elif llm == "claude":
                prompt = SYSTEM_PROMPT + "\n\n" + user_prompt
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 5,
                    "temperature": 0,
                    "top_p": 1,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                })
                txt = json.loads(
                    client.invoke_model(modelId=model_id, contentType="application/json", accept="application/json", body=body)["body"].read()
                )["content"][0]["text"]
            else:
                txt = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    top_p=1,
                    max_tokens=5,
                ).choices[0].message.content

            if m := YES_NO.search(txt.lower()):
                return m.group(0)
            time.sleep(backoff)
        except (RateLimitError, APIConnectionError, OpenAIError):
            time.sleep(backoff * (2 ** attempt))
        except Exception:
            time.sleep(backoff)
    return "N/A"


def process_csv(infile: str, outfile: str, client, model_id: str, llm: str, workers: int):
    with open(infile, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        ftype = file_type_from_name(infile)
        t1_col, t2_col = get_term_columns(rdr.fieldnames, ftype)

    prompts = [make_user_prompt(r[t1_col], r[t2_col]) for r in rows]
    results = ["N/A"] * len(prompts)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(ask_llm, client, p, model_id, llm): i for i, p in enumerate(prompts)}
        for fut in tqdm(as_completed(futs), total=len(prompts), desc=f"{llm} queries"):
            results[futs[fut]] = fut.result()

    fieldnames = list(rows[0].keys()) + ["inverse_verified_by_llm"]
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row, res in zip(rows, results):
            row["inverse_verified_by_llm"] = res
            w.writerow(row)


def main() -> None:
    args = parse_args()
    client, model_id = get_client(args.llm)
    out_path = output_path(args.input_file, args.llm)
    process_csv(args.input_file, out_path, client, model_id, args.llm, args.workers)


if __name__ == "__main__":
    main()