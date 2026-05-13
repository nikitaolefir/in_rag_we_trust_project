import hashlib
import diskcache
from openai import OpenAI


class LLM:
    def __init__(self, model, base_url, api_key, cache_dir,
                 num_ctx, temperature, seed, max_tokens):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.cache = diskcache.Cache(str(cache_dir))
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens

    def __call__(self, prompt):
        key = hashlib.sha256(f"{self.model}|{self.temperature}|{self.seed}|{prompt}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
            extra_body={"options": {"num_ctx": self.num_ctx}},
        )
        text = resp.choices[0].message.content
        self.cache[key] = text
        return text


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import (LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY, CACHE_DIR,
                        NUM_CTX, TEMPERATURE, LLM_SEED, MAX_TOKENS)

    llm = LLM(LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY, CACHE_DIR,
              NUM_CTX, TEMPERATURE, LLM_SEED, MAX_TOKENS)
    print(llm("Say 'pong' and nothing else."))
