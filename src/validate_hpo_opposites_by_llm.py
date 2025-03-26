import csv
import os
import argparse
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inverse relationships between human phenotype terms using an LLM.")
    parser.add_argument("--input_file", required=True, help="Input CSV file path")
    return parser.parse_args()

def generate_output_path(input_file):
    base, ext = os.path.splitext(input_file)
    return f"{base}_llm_validated{ext}"

def query_llm(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0
    )
    answer = response.choices[0].message.content.strip().lower()
    return answer if answer in ('yes', 'no') else 'no'

def process_csv(input_file, output_file, prompt_template):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + ['inverse_verified_by_llm'])

        writer.writeheader()

        for row in tqdm(reader, desc='Processing phenotype terms'):
            prompt = prompt_template.format(term1=row['hpo_term1'], term2=row['hpo_term2'])
            row['inverse_verified_by_llm'] = query_llm(prompt)
            writer.writerow(row)

def main():
    args = parse_args()
    output_file = generate_output_path(args.input_file)

    prompt_template = (
        "Are these two human phenotype terms inversely related phenotypes? "
        "Answer only 'yes' or 'no'.\n"
        "Term 1: {term1}\nTerm 2: {term2}"
    )

    process_csv(args.input_file, output_file, prompt_template)

if __name__ == "__main__":
    main()