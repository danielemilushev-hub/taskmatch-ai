"""Synthetic instruction-following problems (IFEval-style): every constraint
here is mechanically checkable by counting/regex on the raw response, so
grading never needs an LLM judge -- only whether the model followed the
literal instruction, not whether the content itself is "good."
"""

from __future__ import annotations

import random

_TOPICS = [
    "renewable energy",
    "the history of chess",
    "how bicycles work",
    "the water cycle",
    "urban gardening",
    "the invention of the printing press",
    "coral reefs",
    "how bread is made",
    "the phases of the moon",
    "how libraries are organized",
]

_NO_PREAMBLE = (
    "Respond with ONLY the requested content -- no preamble like 'Sure, here is...', "
    "no meta-commentary, no markdown formatting, no quotation marks around the whole thing."
)


def _gen_paragraph_count(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    n = rng.randint(2, 4)
    return {
        "id": f"paragraph_count_{idx}",
        "constraint_type": "paragraph_count",
        "params": {"n": n},
        "prompt": (
            f"Write about {topic} in exactly {n} paragraphs, separated by a blank line. "
            f"{_NO_PREAMBLE}"
        ),
    }


def _gen_forbidden_letter(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    letter = rng.choice("aeiost")
    return {
        "id": f"forbidden_letter_{idx}",
        "constraint_type": "forbidden_letter",
        "params": {"letter": letter},
        "prompt": (
            f"Write a short paragraph (3-5 sentences) about {topic}. "
            f"Do not use the letter '{letter}' anywhere in your response, not even once. "
            f"{_NO_PREAMBLE}"
        ),
    }


def _gen_ending_phrase(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    phrase = rng.choice(
        [
            "That concludes my answer.",
            "This is the end of the response.",
            "Thank you for reading.",
        ]
    )
    return {
        "id": f"ending_phrase_{idx}",
        "constraint_type": "ending_phrase",
        "params": {"phrase": phrase},
        "prompt": (
            f"Write a short paragraph about {topic}. Your response must end with exactly "
            f"this phrase, word for word, as the final text: \"{phrase}\" {_NO_PREAMBLE}"
        ),
    }


def _gen_word_count_max(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    n = rng.choice([20, 30, 40])
    return {
        "id": f"word_count_max_{idx}",
        "constraint_type": "word_count_max",
        "params": {"n": n},
        "prompt": (
            f"Explain {topic} in at most {n} words total. {_NO_PREAMBLE}"
        ),
    }


def _gen_keyword_count(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    word = rng.choice(["important", "process", "energy", "system", "example"])
    n = rng.randint(2, 3)
    return {
        "id": f"keyword_count_{idx}",
        "constraint_type": "keyword_count",
        "params": {"word": word, "n": n},
        "prompt": (
            f"Write a short paragraph about {topic}, using the word '{word}' at least "
            f"{n} times. {_NO_PREAMBLE}"
        ),
    }


def _gen_all_lowercase(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    return {
        "id": f"all_lowercase_{idx}",
        "constraint_type": "all_lowercase",
        "params": {},
        "prompt": (
            f"Write a short sentence about {topic}, entirely in lowercase letters -- "
            f"no capital letters anywhere, not even at the start of the sentence or for "
            f"proper nouns. {_NO_PREAMBLE}"
        ),
    }


def _gen_no_commas(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    return {
        "id": f"no_commas_{idx}",
        "constraint_type": "no_commas",
        "params": {},
        "prompt": (
            f"Write a short paragraph (2-3 sentences) about {topic} without using any "
            f"commas anywhere in your response. {_NO_PREAMBLE}"
        ),
    }


def _gen_starts_with(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(_TOPICS)
    word = rng.choice(["Interestingly", "Notably", "Historically", "Essentially"])
    return {
        "id": f"starts_with_{idx}",
        "constraint_type": "starts_with",
        "params": {"word": word},
        "prompt": (
            f"Write a short paragraph about {topic}. The response must start with the "
            f"exact word '{word}'. {_NO_PREAMBLE}"
        ),
    }


_GENERATORS = [
    _gen_paragraph_count,
    _gen_forbidden_letter,
    _gen_ending_phrase,
    _gen_word_count_max,
    _gen_keyword_count,
    _gen_all_lowercase,
    _gen_no_commas,
    _gen_starts_with,
]


def generate_problems(num_problems: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for i in range(num_problems):
        generator = _GENERATORS[i % len(_GENERATORS)]
        problems.append(generator(rng, i // len(_GENERATORS)))
    return problems
