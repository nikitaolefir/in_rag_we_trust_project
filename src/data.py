import json
import random
from datasets import load_dataset


def load_hotpotqa(n, seed):
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    samples = []
    for i in rng.sample(range(len(ds)), n):
        ex = ds[int(i)]
        passages = [
            {"title": t, "text": "".join(s)}
            for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
        ]
        samples.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],
            "level": ex["level"],
            "supporting_titles": list(set(ex["supporting_facts"]["title"])),
            "passages": passages,
        })
    return samples


def build_corpus(samples):
    seen = {}
    corpus = []
    for s in samples:
        for p in s["passages"]:
            key = (p["title"], p["text"])
            if key in seen:
                continue
            seen[key] = len(corpus)
            corpus.append({"passage_id": len(corpus), "title": p["title"], "text": p["text"]})
    return corpus


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import N_QUESTIONS, SEED, SAMPLES_JSON, CORPUS_JSON

    samples = load_hotpotqa(N_QUESTIONS, SEED)
    corpus = build_corpus(samples)
    save_json(samples, SAMPLES_JSON)
    save_json(corpus, CORPUS_JSON)
    print(len(samples), "questions,", len(corpus), "passages")
