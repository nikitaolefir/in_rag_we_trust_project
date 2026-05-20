# In RAG We Trust?

Experiment on the robustness of a RAG pipeline against poisoned retrieval contexts.
Tests 4 prompt strategies against 3 retrieval conditions on 100 HotpotQA questions.

## Pipeline

HotpotQA → dense retrieval (MiniLM + FAISS) → optional poisoning of top-k → prompting → local LLM (Llama 3.1 8B via Ollama) → classification.

- **Conditions:** `clean`, `entity_swap` (spaCy NER swap on first 2 of 5 passages), `contradiction` (LLM-rewritten passage with reversed claims).
- **Strategies:** `S0_baseline`, `S1_consistency`, `S2_abstention`, `S3_combined`.
- **Metrics:** correct / abstain / hallucination classification, exact match, token F1, pairwise McNemar.

## Layout

```
config.py              # Paths, model names, hyperparameters
requirements.txt
scripts/run.py         # Main driver: 4 strategies x 3 conditions x N questions
src/
  data.py              # HotpotQA loading, corpus building, JSON I/O
  retrieval.py         # SentenceTransformer + FAISS IndexFlatIP
  poisoning.py         # entity_swap, contradiction
  prompts.py           # The 4 strategy instructions + prompt formatter
  llm.py               # OpenAI-compatible client to Ollama with diskcache
  metrics.py           # Normalization, EM/F1, abstention regex, classify, McNemar
notebooks/
  01_demo.ipynb        # Single-question walkthrough of the pipeline
  02_analysis.ipynb    # Loads runs.jsonl, computes rates, writes figures
data/
  samples.json         # 100 sampled HotpotQA questions
  corpus.json          # Deduplicated passages from the sampled contexts
  llm_cache/           # diskcache for LLM responses
results/runs.jsonl     # One record per (question, strategy, condition); append-only with resume
paper/figures/         # rates.csv, rates_bar.png, rates_heatmap.png
Report.pdf
```

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b-instruct-q4_K_M

python -m src.data                  # builds data/samples.json + data/corpus.json
python scripts/run.py               # appends to results/runs.jsonl, resumable
jupyter notebook notebooks/02_analysis.ipynb
```

## Configuration

See [config.py](config.py). Key knobs: `N_QUESTIONS=100`, `TOP_K=5`, `N_POISONED=2`, `SEED=42`, `LLM_SEED=42`, `TEMPERATURE=0.0`.
