# phenopposites
Project to generate phenotype opposite_of relationships in PATO, MPO and HPO. We also investigate the effect including this reltionship in ontology-based algorithms like semantic similarity.

 * lexically derived - these were obtained by looking for pairs of labels that differ according to an antonym pair (see [antonyms file](https://github.com/Phenomics/phenopposites/blob/master/antonyms/text-patterns.txt))
 * logically derived - using the OWL definitions in HP/MP, and the [is opposite of](http://purl.obolibrary.org/obo/RO_0002604) relationships in PATO. See for example [increased height](https://www.ebi.ac.uk/ols/ontologies/pato/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FPATO_0000570).

# publication
Currently there is only a preprint version for this project. See http://biorxiv.org/content/early/2017/02/16/108977


# Automation & Data Refresh

This fork of **phenopposites** was created to modernize and extend the original repository. The primary goal is to systematically generate opposite-of relationships in phenotype ontologies using an automated pipeline. 

### Prerequisites

- Git
- Miniconda (with Mamba)

### Setup

**1. Initialize conda environment:**
   ```sh
   cd phenopposites
   mamba env create -f environment.yml
   ```
**2. Run:**
   ```sh
   cd phenopposites
   conda activate phenopposites

   # ensures up-to-date ontology data is fetched before processing
   bash scripts/download.sh

   # extracts opposite terms for PATO using short id
   python src/generate_pato_opposites_shortid.py

   # identifies opposite terms using text-based antonym replacement
   python src/hpo_opposites_text_matching.py 

   # validate opposite terms using LLM
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_text.csv --llm openai
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_text.csv --llm claude
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_text.csv --llm deepseek
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_text.csv --llm gemini
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_text.csv --llm grok
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_text.csv --llm llama
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_text.csv --llm qwen

   # extracts opposite terms using logical definitions from ontology structure
   python src/hpo_opposites_logical_matching.py

   # validate opposite terms using LLM
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_logical.csv --llm openai
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_logical.csv --llm claude
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_logical.csv --llm deepseek
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_logical.csv --llm gemini
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_logical.csv --llm grok
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_logical.csv --llm llama
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_logical.csv --llm qwen

   # merges text-based and logical opposite pairs into a single dataset
   python src/hpo_opposites_unify.py  

   # inherits opposite-of relationships to descendant terms
   python src/hpo_opposites_propagate.py

   # validate opposite terms using LLM
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_inherited.csv --llm openai
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_inherited.csv --llm claude
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_inherited.csv --llm deepseek
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_inherited.csv --llm gemini
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_inherited.csv --llm grok
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_inherited.csv --llm llama 
   python src/validate_opposites_by_llm.py --input_file ./data/output/hpo_opposites_inherited.csv --llm qwen 

   # extract final verified opposite terms
   python src/hpo_opposites_extract_verified.py 
   ```