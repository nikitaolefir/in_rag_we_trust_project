import re
import string
from collections import Counter
from scipy.stats import binomtest

PUNCT = str.maketrans("", "", string.punctuation)
ARTICLES = re.compile(r"\b(a|an|the)\b")

ABSTAIN_PATTERNS = [
    r"\bi cannot answer reliably\b",
    r"\bi (don'?t|do not) know\b",
    r"\bcannot (be )?(answer|determine)",
    r"\binsufficient (information|context|evidence|data)\b",
    r"\bnot (enough|sufficient) (information|context|evidence)\b",
    r"\b(documents?|sources?) (are|seem to be|appear) (contradictory|inconsistent|insufficient)\b",
    r"\bunable to (answer|determine)\b",
    r"\bno (definitive|clear) answer\b",
]


def normalize(s):
    s = ARTICLES.sub(" ", s.lower()).translate(PUNCT)
    return " ".join(s.split())


def exact_match(pred, gold):
    return int(normalize(pred) == normalize(gold))


def contains_gold(pred, gold):
    ng = normalize(gold)
    if not ng:
        return 0
    return int(re.search(rf"\b{re.escape(ng)}\b", normalize(pred)) is not None)


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def is_abstention(pred):
    p = pred.lower()
    return any(re.search(pat, p) for pat in ABSTAIN_PATTERNS)


def classify(pred, gold):
    if is_abstention(pred):
        return "abstain"
    if contains_gold(pred, gold) or token_f1(pred, gold) >= 0.6:
        return "correct"
    return "hallucination"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, p=0.5).pvalue
