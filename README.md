# CV Sorting using LLMs (CS02)

Ranks a folder of CVs (.pdf/.docx/.txt) against a job description using local LLMs served by [Ollama](https://ollama.com/download). No API key is needed.

## Requirements

- Python 3.9+
- Ollama with the following models:
  - `granite4.2:3b` – structured extraction
  - `qwen3.5:9b` – requirement-level judge
  - `nomic-embed-text` – embeddings

```bash
pip install -r requirements.txt
ollama pull granite4.2:3b
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
```

## Usage

Run from the `Codebase/` directory:

```bash
# Rank CVs
python main.py --jd ./job_description.txt --cvs ./cvs

# Rank and evaluate against relevance labels (CSV header "file,relevance", values 0-3)
python main.py --jd ./job_description.txt --cvs ./cvs --labels ./labels.csv --out ./output

# Rank, then refine interactively
python main.py --jd ./job_description.txt --cvs ./cvs --interactive
```

See [Codebase/execution.txt](Codebase/execution.txt) for all options, or run `python main.py --help`.
