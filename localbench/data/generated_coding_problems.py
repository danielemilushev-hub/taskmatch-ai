"""Procedurally generated coding problems.

The fixed problem set (factorial, fibonacci, is_prime, is_palindrome, binary
search) has a serious validity problem: those exact functions appear in
essentially every training corpus, so a model can pass them from memorisation
without demonstrating any coding ability. Whatever that measures, it is recall,
not competence.

These are composed instead from randomised parameters -- an operation, a
filter predicate, a threshold, an ordering -- so the specific combination a
run asks for almost certainly never appeared in training. The underlying
skills are the same ordinary ones (iterate, filter, accumulate, handle the
empty case); only the exact task is novel.

Ground truth is computed by executing a reference implementation here in the
harness, never hand-written, so the expected answer cannot drift from the
stated task.
"""

from __future__ import annotations

import random

# (label used in the prompt, predicate over an int)
_FILTERS = [
    ("even", lambda x: x % 2 == 0),
    ("odd", lambda x: x % 2 == 1),
    ("positive", lambda x: x > 0),
    ("negative", lambda x: x < 0),
    ("divisible by 3", lambda x: x % 3 == 0),
    ("greater than 10", lambda x: x > 10),
    ("less than -5", lambda x: x < -5),
    ("a multiple of 5", lambda x: x % 5 == 0),
]

# (label, reducer over a list of ints, identity value for the empty case)
_REDUCERS = [
    ("sum", lambda xs: sum(xs), 0),
    ("product", lambda xs: _product(xs), 1),
    ("count", lambda xs: len(xs), 0),
    ("maximum", lambda xs: max(xs) if xs else 0, 0),
    ("minimum", lambda xs: min(xs) if xs else 0, 0),
]


def _product(xs: list[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


def _filter_reduce(rng: random.Random, idx: int) -> dict:
    """"Return the <reducer> of all <filter> numbers in the list."""
    fname, fpred = rng.choice(_FILTERS)
    rname, rfunc, identity = rng.choice(_REDUCERS)
    fn = f"solve_{idx}"

    def reference(nums: list[int]) -> int:
        return rfunc([n for n in nums if fpred(n)])

    cases: list[list[int]] = [
        [rng.randint(-20, 40) for _ in range(rng.randint(4, 9))] for _ in range(4)
    ]
    cases.append([])  # the empty list -- the case models most often get wrong
    cases.append([rng.randint(-20, 40)])

    return {
        "id": f"filter_reduce_{idx}",
        "entry_point": fn,
        "prompt": (
            f"Write a Python function `{fn}(nums)` that takes a list of integers "
            f"and returns the {rname} of all values in the list that are {fname}.\n"
            f"If no values match, return {identity}."
        ),
        "tests": [{"args": [c], "expected": reference(c)} for c in cases],
    }


def _transform_join(rng: random.Random, idx: int) -> dict:
    """String transform with a randomised rule -- tests exact spec-following."""
    sep = rng.choice(["-", "|", "::", ","])
    case = rng.choice(["upper", "lower", "title"])
    order = rng.choice(["as given", "reversed", "sorted alphabetically"])
    min_len = rng.randint(2, 4)
    fn = f"solve_{idx}"

    def reference(words: list[str]) -> str:
        kept = [w for w in words if len(w) >= min_len]
        if order == "reversed":
            kept = kept[::-1]
        elif order == "sorted alphabetically":
            kept = sorted(kept)
        kept = [getattr(w, case)() for w in kept]
        return sep.join(kept)

    pool = ["apple", "be", "cat", "dog", "elephant", "fox", "go", "hat", "ice", "jam", "kiwi", "lm"]
    cases = [rng.sample(pool, rng.randint(3, 6)) for _ in range(4)]
    cases.append([])
    cases.append([rng.choice(pool)])

    return {
        "id": f"transform_join_{idx}",
        "entry_point": fn,
        "prompt": (
            f"Write a Python function `{fn}(words)` that takes a list of strings and:\n"
            f"1. keeps only words with at least {min_len} characters\n"
            f"2. orders the kept words: {order}\n"
            f"3. converts each to {case}case\n"
            f"4. joins them with '{sep}' and returns the resulting string.\n"
            f"Return an empty string if no words remain."
        ),
        "tests": [{"args": [c], "expected": reference(c)} for c in cases],
    }


def _dict_aggregate(rng: random.Random, idx: int) -> dict:
    """Group/aggregate over records -- closer to real data-wrangling work."""
    threshold = rng.randint(20, 60)
    keep_above = rng.choice([True, False])
    rname, rfunc, identity = rng.choice(_REDUCERS[:3])
    fn = f"solve_{idx}"
    cmp_label = "greater than" if keep_above else "less than"

    def reference(records: list[dict]) -> dict:
        out: dict[str, int] = {}
        for r in records:
            val = r["value"]
            if (val > threshold) if keep_above else (val < threshold):
                out.setdefault(r["group"], []).append(val)
        return {k: rfunc(v) for k, v in out.items()}

    groups = ["a", "b", "c"]
    cases = [
        [
            {"group": rng.choice(groups), "value": rng.randint(0, 100)}
            for _ in range(rng.randint(4, 8))
        ]
        for _ in range(4)
    ]
    cases.append([])
    cases.append([{"group": "a", "value": rng.randint(0, 100)}])

    return {
        "id": f"dict_aggregate_{idx}",
        "entry_point": fn,
        "prompt": (
            f"Write a Python function `{fn}(records)` where records is a list of "
            f"dicts, each with keys 'group' (a string) and 'value' (an integer).\n"
            f"Keep only records whose value is {cmp_label} {threshold}, then return a "
            f"dict mapping each group name to the {rname} of its kept values.\n"
            f"Groups with no kept records must not appear in the result. "
            f"Return an empty dict if nothing matches."
        ),
        "tests": [{"args": [c], "expected": reference(c)} for c in cases],
    }


def _running_sequence(rng: random.Random, idx: int) -> dict:
    """Stateful pass over a list -- catches off-by-one and accumulator bugs."""
    fn = f"solve_{idx}"
    mode = rng.choice(["running total", "running maximum", "difference from previous"])
    skip_first = rng.choice([True, False])

    def reference(nums: list[int]) -> list[int]:
        vals = nums[1:] if skip_first else nums
        out: list[int] = []
        acc = 0
        best = None
        for i, n in enumerate(vals):
            if mode == "running total":
                acc += n
                out.append(acc)
            elif mode == "running maximum":
                best = n if best is None else max(best, n)
                out.append(best)
            else:
                out.append(0 if i == 0 else n - vals[i - 1])
        return out

    cases = [[rng.randint(-15, 30) for _ in range(rng.randint(3, 7))] for _ in range(4)]
    cases.append([])
    cases.append([rng.randint(-15, 30)])

    skip_note = " Ignore the first element of the input list." if skip_first else ""
    if mode == "difference from previous":
        detail = (
            "each element is that value minus the previous one, with 0 for the first position"
        )
    else:
        detail = f"each element is the {mode} up to and including that position"

    return {
        "id": f"running_sequence_{idx}",
        "entry_point": fn,
        "prompt": (
            f"Write a Python function `{fn}(nums)` that takes a list of integers and "
            f"returns a new list where {detail}.{skip_note}\n"
            f"Return an empty list if there is nothing to process."
        ),
        "tests": [{"args": [c], "expected": reference(c)} for c in cases],
    }


_GENERATORS = [_filter_reduce, _transform_join, _dict_aggregate, _running_sequence]


def generate_problems(num_problems: int = 12, seed: int = 42) -> list[dict]:
    """Deterministic for a given seed, so a run is reproducible, while a
    different seed yields a genuinely different problem set."""
    rng = random.Random(seed)
    problems = []
    for i in range(num_problems):
        problems.append(_GENERATORS[i % len(_GENERATORS)](rng, i))
    return problems
