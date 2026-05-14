import re
from collections import defaultdict

SWAPPABLE = {"PERSON", "ORG", "GPE", "DATE", "NORP", "LOC", "EVENT", "WORK_OF_ART"}


def build_entity_pool(corpus, nlp):
    pool = defaultdict(set)
    for p in corpus:
        for ent in nlp(p["text"]).ents:
            if ent.label_ in SWAPPABLE and len(ent.text) > 1:
                pool[ent.label_].add(ent.text)
    return {k: list(v) for k, v in pool.items()}


def poison_entity_swap(doc, nlp, pool, rng):
    ents = [e for e in nlp(doc["text"]).ents if e.label_ in SWAPPABLE]
    if not ents:
        return {**doc, "poisoned": "entity_swap_failed"}
    target = rng.choice(ents)
    candidates = [x for x in pool.get(target.label_, []) if x != target.text]
    if not candidates:
        return {**doc, "poisoned": "entity_swap_failed"}
    replacement = rng.choice(candidates)
    new_text = re.sub(re.escape(target.text), replacement, doc["text"])
    return {**doc, "text": new_text, "poisoned": "entity_swap",
            "swap": {"from": target.text, "to": replacement, "label": target.label_}}


def poison_contradiction(doc, llm_call):
    prompt = (
        "Rewrite the passage below so that its main factual claims are reversed or "
        "contradicted, while preserving the original length and writing style. Keep "
        "the same named entities and topic. Do not add disclaimers or meta-commentary. "
        "Output only the rewritten passage.\n\n"
        f"Passage:\n{doc['text']}"
    )
    return {**doc, "text": llm_call(prompt).strip(), "poisoned": "contradiction"}
