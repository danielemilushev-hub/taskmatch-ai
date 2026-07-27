"""Procedurally generated JSON-schema compliance problems.

The fixed set was only five hand-written problems, which is far too few to
measure anything: a perfect 5/5 has a 95% confidence interval reaching down
to 57%. It also always asked for the same five shapes, so a model could in
principle have seen them.

Here the schema is assembled from randomised field types and constraints, and
the prompt is generated *from that schema* rather than written alongside it,
so the task description can never drift from what is actually validated --
the usual way a hand-maintained suite silently starts grading unfairly.

Grading needs no reference answer: the schema is the ground truth, and
jsonschema decides. Every problem also sets additionalProperties: false, so
a model that pads the object with extra keys is caught rather than passed.
"""

from __future__ import annotations

import json
import random

_STR_POOL = ["name", "title", "label", "city", "author", "category", "summary"]
_INT_POOL = ["count", "quantity", "age", "score", "priority", "year", "size"]
_NUM_POOL = ["rating", "price", "weight", "ratio", "temperature", "latitude"]
_BOOL_POOL = ["active", "resolved", "verified", "archived", "public", "urgent"]
_ARR_POOL = ["tags", "items", "keywords", "authors", "steps", "categories"]
_ENUM_VALUES = [
    ["low", "medium", "high"],
    ["draft", "published", "archived"],
    ["red", "green", "blue"],
    ["small", "medium", "large"],
    ["pending", "approved", "rejected"],
]


def _string_field(rng: random.Random, name: str):
    schema = {"type": "string"}
    desc = [f"'{name}' (a string"]
    if rng.random() < 0.5:
        n = rng.randint(3, 8)
        schema["minLength"] = n
        desc.append(f"at least {n} characters long")
    desc[-1] += ")"
    return schema, " ".join(desc)


def _pattern_field(rng: random.Random, name: str):
    kind = rng.choice(["code", "sku", "id"])
    if kind == "code":
        pattern, human = r"^[A-Z]{3}-\d{4}$", "three uppercase letters, a hyphen, then exactly 4 digits"
    elif kind == "sku":
        pattern, human = r"^[A-Z]{2}\d{6}$", "two uppercase letters followed by exactly 6 digits"
    else:
        pattern, human = r"^\d{4}-\d{2}$", "four digits, a hyphen, then exactly 2 digits"
    return {"type": "string", "pattern": pattern}, f"'{name}' (a string matching {human})"


def _integer_field(rng: random.Random, name: str):
    lo = rng.randint(0, 20)
    hi = lo + rng.randint(5, 80)
    return (
        {"type": "integer", "minimum": lo, "maximum": hi},
        f"'{name}' (an integer between {lo} and {hi} inclusive)",
    )


def _number_field(rng: random.Random, name: str):
    lo = rng.choice([0, 1, -10])
    hi = lo + rng.choice([5, 10, 100])
    return (
        {"type": "number", "minimum": lo, "maximum": hi},
        f"'{name}' (a number between {lo} and {hi} inclusive)",
    )


def _boolean_field(rng: random.Random, name: str):
    return {"type": "boolean"}, f"'{name}' (a boolean)"


def _enum_field(rng: random.Random, name: str):
    values = rng.choice(_ENUM_VALUES)
    return (
        {"type": "string", "enum": list(values)},
        f"'{name}' (exactly one of: {', '.join(repr(v) for v in values)})",
    )


def _array_field(rng: random.Random, name: str):
    n = rng.randint(2, 4)
    return (
        {"type": "array", "items": {"type": "string"}, "minItems": n},
        f"'{name}' (an array of at least {n} strings)",
    )


def _nested_field(rng: random.Random, name: str):
    inner_str = rng.choice(_STR_POOL)
    inner_int = rng.choice(_INT_POOL)
    return (
        {
            "type": "object",
            "properties": {inner_str: {"type": "string"}, inner_int: {"type": "integer"}},
            "required": [inner_str, inner_int],
            "additionalProperties": False,
        },
        f"'{name}' (an object containing exactly '{inner_str}' (string) and '{inner_int}' (integer))",
    )


_BUILDERS = [
    (_string_field, _STR_POOL),
    (_pattern_field, ["code", "sku", "reference"]),
    (_integer_field, _INT_POOL),
    (_number_field, _NUM_POOL),
    (_boolean_field, _BOOL_POOL),
    (_enum_field, ["status", "level", "colour", "state"]),
    (_array_field, _ARR_POOL),
    (_nested_field, ["details", "meta", "owner"]),
]

_SUBJECTS = [
    "a product listing", "a support ticket", "a calendar event", "a user profile",
    "a shipping record", "a book entry", "a sensor reading", "a job posting",
]


def _make_problem(rng: random.Random, idx: int) -> dict:
    n_fields = rng.randint(3, 5)
    builders = rng.sample(_BUILDERS, n_fields)

    props: dict = {}
    descriptions: list[str] = []
    used: set[str] = set()
    for builder, pool in builders:
        name = rng.choice([n for n in pool if n not in used] or pool)
        used.add(name)
        schema, desc = builder(rng, name)
        props[name] = schema
        descriptions.append(desc)

    schema = {
        "type": "object",
        "properties": props,
        "required": sorted(props.keys()),
        # strict: catches a model that invents extra keys, which a lenient
        # schema would silently accept
        "additionalProperties": False,
    }

    subject = rng.choice(_SUBJECTS)
    task = (
        f"Generate a JSON object describing {subject}. It must contain exactly "
        f"these {len(props)} fields and no others:\n"
        + "\n".join(f"- {d}" for d in descriptions)
    )

    return {"id": f"schema_{idx}_{len(props)}fields", "task": task, "schema": schema}


def generate_problems(num_problems: int = 20, seed: int = 42) -> list[dict]:
    """Deterministic for a given seed, so a run is reproducible."""
    rng = random.Random(seed)
    return [_make_problem(rng, i) for i in range(num_problems)]


if __name__ == "__main__":  # quick manual inspection
    for p in generate_problems(3, seed=1):
        print(p["id"])
        print(p["task"])
        print(json.dumps(p["schema"], indent=2))
        print()
