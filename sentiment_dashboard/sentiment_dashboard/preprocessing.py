"""Text preprocessing for sentiment classification.

Pipeline (applied in this order):
    1. Remove HTML tags / entities          ``<br />`` -> space
    2. Lowercase
    3. Remove URLs
    4. Expand contractions                  ``don't`` -> ``do not``
    5. Remove punctuation, digits, symbols
    6. Tokenise on whitespace
    7. Remove stopwords (negations are KEPT, e.g. "not", "no")
    8. Lemmatise                            ``movies`` -> ``movie``

NLTK (stopwords + WordNet lemmatiser) is used when available. If the NLTK data
cannot be downloaded (e.g. no internet), small built-in fallbacks are used so
the app still runs. ``get_backend_info()`` reports which one is active, and the
training script stores it with the model so mismatches can be detected.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache
from typing import Iterable, List

import pandas as pd

# ----------------------------------------------------------------------------
# Regular expressions
# ----------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_SPACES_RE = re.compile(r"\s+")

_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]

# Words that flip sentiment. Standard stopword lists contain them, but removing
# them would turn "not good" into "good", so we always keep them.
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "neither"}

_FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "yours", "yourself", "yourselves",
}

# ----------------------------------------------------------------------------
# Lazy initialisation of NLTK resources
# ----------------------------------------------------------------------------
_STATE = {"ready": False, "stopwords": set(_FALLBACK_STOPWORDS), "lemmatizer": None,
          "stopwords_source": "built-in list", "lemmatizer_source": "built-in suffix rules"}


def _init_resources() -> None:
    """Load NLTK stopwords / WordNet once; silently fall back if unavailable."""
    if _STATE["ready"]:
        return
    _STATE["ready"] = True
    try:
        import nltk  # noqa: WPS433
    except ImportError:
        return

    def _ensure(resource: str, path: str) -> bool:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            try:
                return bool(nltk.download(resource, quiet=True))
            except Exception:  # network problems, permissions, ...
                return False

    if _ensure("stopwords", "corpora/stopwords"):
        try:
            from nltk.corpus import stopwords
            _STATE["stopwords"] = set(stopwords.words("english"))
            _STATE["stopwords_source"] = "NLTK stopwords"
        except Exception:
            pass

    if _ensure("wordnet", "corpora/wordnet"):
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("tests")  # raises LookupError if data is unusable
            _STATE["lemmatizer"] = lemmatizer
            _STATE["lemmatizer_source"] = "NLTK WordNet lemmatizer"
        except Exception:
            pass


def get_backend_info() -> dict:
    """Describe which stopword list / lemmatiser is currently active."""
    _init_resources()
    using_nltk = (_STATE["stopwords_source"].startswith("NLTK")
                  and _STATE["lemmatizer_source"].startswith("NLTK"))
    return {
        "stopwords": _STATE["stopwords_source"],
        "lemmatizer": _STATE["lemmatizer_source"],
        "using_nltk": using_nltk,
    }


def get_stopwords() -> set:
    _init_resources()
    return {w for w in _STATE["stopwords"] if w not in NEGATIONS}


# ----------------------------------------------------------------------------
# Lemmatisation
# ----------------------------------------------------------------------------
def _fallback_lemma(word: str) -> str:
    """Very small rule-based lemmatiser used only when WordNet is unavailable."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    return word


@lru_cache(maxsize=100_000)
def lemmatize_word(word: str) -> str:
    _init_resources()
    lemmatizer = _STATE["lemmatizer"]
    if lemmatizer is None:
        return _fallback_lemma(word)
    return lemmatizer.lemmatize(lemmatizer.lemmatize(word, "v"), "n")


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def clean_text(text: object) -> str:
    """Steps 1-5: strip HTML, lowercase, remove URLs/contractions/punctuation."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = html.unescape(str(text))
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = _URL_RE.sub(" ", text)
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def tokenize(text: object) -> List[str]:
    """Steps 1-8: return the final list of cleaned tokens."""
    stop = get_stopwords()
    tokens = [t for t in clean_text(text).split() if len(t) > 1 and t not in stop]
    return [lemmatize_word(t) for t in tokens]


def preprocess_text(text: object) -> str:
    """Full preprocessing pipeline; returns a single space-joined string."""
    return " ".join(tokenize(text))


def preprocess_series(texts: Iterable[object]) -> pd.Series:
    """Apply :func:`preprocess_text` to many reviews (keeps the index)."""
    series = texts if isinstance(texts, pd.Series) else pd.Series(list(texts))
    return series.map(preprocess_text)
