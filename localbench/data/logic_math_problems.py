"""Synthetic logic/math problem generation -- answers are always known because
we generate them, so grading is exact match, no reference dataset needed.
"""

from __future__ import annotations

import random

_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]

_ANSWER_INSTRUCTION = (
    "Think step by step if you need to, but end your response with a final "
    "line in exactly this format (no extra words on that line):\nAnswer: <value>"
)


def _gen_arithmetic(rng: random.Random, idx: int) -> dict:
    a = rng.randint(-50, 50)
    b = rng.randint(1, 20)
    c = rng.randint(-20, 20)
    op1 = rng.choice(["+", "-", "*"])
    op2 = rng.choice(["+", "-", "*"])
    expr = f"{a} {op1} {b} {op2} {c}"
    answer = eval(expr)  # noqa: S307 -- expr built entirely from our own random ints/ops
    return {
        "id": f"arithmetic_{idx}",
        "category": "arithmetic",
        "prompt": (
            f"What is {expr}? Follow standard operator precedence "
            f"(multiplication before addition/subtraction).\n\n{_ANSWER_INSTRUCTION}"
        ),
        "answer_type": "int",
        "expected": answer,
    }


def _gen_comparison_chain(rng: random.Random, idx: int) -> dict:
    names = rng.sample(_NAMES, 3)
    trait = rng.choice(["taller than", "older than", "faster than", "heavier than"])
    facts = [
        f"{names[0]} is {trait} {names[1]}.",
        f"{names[1]} is {trait} {names[2]}.",
    ]
    ask_true = rng.random() < 0.5
    if ask_true:
        question = f"Is {names[0]} {trait} {names[2]}?"
        expected = "yes"
    else:
        question = f"Is {names[2]} {trait} {names[0]}?"
        expected = "no"
    prompt = " ".join(facts) + f" {question} Answer Yes or No.\n\n{_ANSWER_INSTRUCTION}"
    return {
        "id": f"comparison_chain_{idx}",
        "category": "comparison_chain",
        "prompt": prompt,
        "answer_type": "yes_no",
        "expected": expected,
    }


def _gen_boolean_expression(rng: random.Random, idx: int) -> dict:
    tokens = []
    num_literals = rng.randint(2, 3)
    for i in range(num_literals):
        if i > 0:
            tokens.append(rng.choice(["and", "or"]))
        literal = rng.choice(["True", "False"])
        if rng.random() < 0.3:
            tokens.append("not")
        tokens.append(literal)
    expr = " ".join(tokens)
    expected = eval(expr)  # noqa: S307 -- expr built entirely from our own True/False/and/or/not tokens
    return {
        "id": f"boolean_expression_{idx}",
        "category": "boolean_expression",
        "prompt": (
            f"Evaluate this boolean expression: {expr}\n\n{_ANSWER_INSTRUCTION} "
            f"(the value should be exactly True or False)"
        ),
        "answer_type": "bool",
        "expected": str(expected),
    }


_GENERATORS = [_gen_arithmetic, _gen_comparison_chain, _gen_boolean_expression]


def generate_problems(num_problems: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    problems = []
    for i in range(num_problems):
        generator = _GENERATORS[i % len(_GENERATORS)]
        problems.append(generator(rng, i // len(_GENERATORS)))
    return problems
