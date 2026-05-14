import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy
from tqdm import tqdm

from config import (
    EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY, CACHE_DIR,
    NUM_CTX, TEMPERATURE, LLM_SEED, MAX_TOKENS,
    TOP_K, N_POISONED,
    STRATEGIES, CONDITIONS,
    SAMPLES_JSON, CORPUS_JSON, RUNS_JSONL,
)
from src.data import load_json
from src.retrieval import build_index, search
from src.poisoning import build_entity_pool, poison_entity_swap, poison_contradiction
from src.prompts import format_prompt
from src.llm import LLM


def apply_condition(passages, condition, nlp, pool, llm, rng):
    if condition == "clean":
        return passages
    out = []
    for i, p in enumerate(passages):
        if i < N_POISONED:
            if condition == "entity_swap":
                out.append(poison_entity_swap(p, nlp, pool, rng))
            elif condition == "contradiction":
                out.append(poison_contradiction(p, llm))
        else:
            out.append(p)
    return out


def load_done(path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["strategy"], r["condition"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    samples = load_json(SAMPLES_JSON)
    if args.limit:
        samples = samples[:args.limit]
    corpus = load_json(CORPUS_JSON)

    nlp = spacy.load("en_core_web_sm")
    pool = build_entity_pool(corpus, nlp)
    index, model = build_index(corpus, EMBEDDING_MODEL)
    llm = LLM(LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY, CACHE_DIR,
              NUM_CTX, TEMPERATURE, LLM_SEED, MAX_TOKENS)

    done = load_done(RUNS_JSONL)
    retrieved = {s["id"]: search(index, model, corpus, s["question"], TOP_K) for s in samples}

    total = len(STRATEGIES) * len(CONDITIONS) * len(samples)
    pbar = tqdm(total=total, initial=len(done))
    with open(RUNS_JSONL, "a") as out:
        for condition in CONDITIONS:
            for s in samples:
                rng = random.Random(hash((s["id"], condition)) & 0xFFFFFFFF)
                passages = apply_condition(retrieved[s["id"]], condition, nlp, pool, llm, rng)
                for strategy in STRATEGIES:
                    if (s["id"], strategy, condition) in done:
                        continue
                    prompt = format_prompt(strategy, s["question"], passages)
                    answer = llm(prompt)
                    out.write(json.dumps({
                        "id": s["id"],
                        "question": s["question"],
                        "gold": s["answer"],
                        "strategy": strategy,
                        "condition": condition,
                        "answer": answer,
                        "passages_meta": [
                            {k: p.get(k) for k in ("passage_id", "title", "poisoned", "swap") if k in p}
                            for p in passages
                        ],
                    }) + "\n")
                    out.flush()
                    pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
