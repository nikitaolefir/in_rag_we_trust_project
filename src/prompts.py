INSTRUCTIONS = {
    "S0_baseline": "Answer the question using the provided documents.",

    "S1_consistency": (
        "Before answering, check whether the documents agree on the relevant facts. "
        "If they conflict, identify the conflict explicitly. Then answer the question "
        "using only information that is consistent across documents."
    ),

    "S2_abstention": (
        "Answer the question using the provided documents. If the documents are "
        "insufficient or contradictory, respond exactly: 'I cannot answer reliably.'"
    ),

    "S3_combined": (
        "Before answering, check whether the documents agree on the relevant facts. "
        "If they conflict, identify the conflict. If the documents are insufficient or "
        "contradictory, respond exactly: 'I cannot answer reliably.' Otherwise, answer "
        "using only information that is consistent across documents."
    ),
}


def format_prompt(strategy, question, passages):
    docs = "\n\n".join(f"[Doc {i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages))
    return f"{INSTRUCTIONS[strategy]}\n\nDocuments:\n{docs}\n\nQuestion: {question}\nAnswer:"
