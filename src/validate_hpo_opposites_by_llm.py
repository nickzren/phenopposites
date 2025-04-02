import csv
import os
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inverse relationships between human phenotype terms using OpenAI or DeepSeek API.")
    parser.add_argument("--input_file", required=True, help="Input CSV file path")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel LLM requests")
    parser.add_argument("--llm_provider", choices=['openai', 'deepseek'], default='openai', help="LLM provider to use (default: openai)")
    return parser.parse_args()

def generate_output_path(input_file):
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_dir = os.getenv("OUTPUT_DIR")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{name}_llm_validated{ext}")

def create_prompt(term1, term2):
    prompt_instructions = (
        "Determine if these two human phenotype terms represent an inverse (Opposite-of) relationship. "
        "Answer yes or no.\n"
    )
    return f"{prompt_instructions}Term 1: {term1}\nTerm 2: {term2}"

def get_client(provider):
    if provider == 'openai':
        api_base_url = os.getenv("OPENAI_API_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        api_model = os.getenv("OPENAI_API_MODEL")
    else:  # deepseek
        api_base_url = os.getenv("DEEPSEEK_API_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_model = os.getenv("DEEPSEEK_API_MODEL")

    return OpenAI(api_key=api_key, base_url=api_base_url), api_model

def query_llm(client, prompt, model, retries=3):
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0
            )
            answer = response.choices[0].message.content.strip().lower()
            if answer == "yes":
                return "t"
            elif answer == "no":
                return "f"
        except Exception:
            pass
    return "N/A"

def process_csv(input_file, output_file, client, model, workers):
    with open(input_file, 'r', encoding='utf-8') as infile:
        rows = list(csv.DictReader(infile))

    prompts = [create_prompt(row["hpo_term1"], row["hpo_term2"]) for row in rows]
    results = ["N/A"] * len(prompts)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {executor.submit(query_llm, client, prompt, model): i for i, prompt in enumerate(prompts)}
        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc='Processing LLM queries'):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = "N/A"

    fieldnames = list(rows[0].keys()) + ["inverse_verified_by_llm"]
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row, result in zip(rows, results):
            row["inverse_verified_by_llm"] = result
            writer.writerow(row)

def main():
    args = parse_args()
    client, model = get_client(args.llm_provider)
    print(f"Using LLM Provider: {args.llm_provider.upper()}, Model: {model}")
    output_file = generate_output_path(args.input_file)
    process_csv(args.input_file, output_file, client, model, args.workers)

if __name__ == "__main__":
    main()