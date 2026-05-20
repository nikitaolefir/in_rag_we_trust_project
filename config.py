from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = DATA_DIR / "llm_cache"

for d in (DATA_DIR, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_QUESTIONS = 100
SEED = 42
TOP_K = 5
N_POISONED = 2

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
NUM_CTX = 8192
TEMPERATURE = 0.0
LLM_SEED = 42
MAX_TOKENS = 512

STRATEGIES = ["S0_baseline", "S1_consistency", "S2_abstention", "S3_combined"]
CONDITIONS = ["clean", "entity_swap", "contradiction"]

RUNS_JSONL = RESULTS_DIR / "runs.jsonl"
CORPUS_JSON = DATA_DIR / "corpus.json"
SAMPLES_JSON = DATA_DIR / "samples.json"
