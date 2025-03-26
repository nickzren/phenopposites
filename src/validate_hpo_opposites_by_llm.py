import csv
import os
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inverse relationships between human phenotype terms using an LLM.")
    parser.add_argument("--input_file", required=True, help="Input CSV file path")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel LLM requests")
    return parser.parse_args()

def generate_output_path(input_file):
    base, ext = os.path.splitext(input_file)
    return f"{base}_llm_validated{ext}"

def create_prompt(term1, term2):
    return (
        "Are these two human phenotype terms inversely related phenotypes? "
        "Answer only 't' or 'f'.\n"
        f"Term 1: {term1}\nTerm 2: {term2}"
    )

def query_llm(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer if answer in {"t", "f"} else "f"

def process_csv(input_file, output_file, workers):
    with open(input_file, 'r', encoding='utf-8') as infile:
        rows = list(csv.DictReader(infile))

    prompts = [create_prompt(row["hpo_term1"], row["hpo_term2"]) for row in rows]

    results = ["f"] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {executor.submit(query_llm, prompt): i for i, prompt in enumerate(prompts)}

        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc='Processing LLM queries'):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = "f"

    fieldnames = list(rows[0].keys()) + ["inverse_verified_by_llm"]
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row, result in zip(rows, results):
            row["inverse_verified_by_llm"] = result
            writer.writerow(row)

def main():
    args = parse_args()
    output_file = generate_output_path(args.input_file)
    process_csv(args.input_file, output_file, args.workers)

if __name__ == "__main__":
    main()