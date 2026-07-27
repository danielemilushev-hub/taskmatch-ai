"""HumanEval-style coding problems with reference solutions and test cases.

Each problem's generated code is executed in a subprocess (see
suites/coding_suite.py) against `tests` — the reference_solution is not run
against the model's code, it exists so this data file's own correctness can
be verified (see verify_problems() below).
"""

PROBLEMS = [
    {
        "id": "add",
        "prompt": "Write a Python function `def add(a, b):` that returns the sum of a and b.",
        "entry_point": "add",
        "reference_solution": "def add(a, b):\n    return a + b\n",
        "tests": [
            {"args": [1, 2], "expected": 3},
            {"args": [-1, 1], "expected": 0},
            {"args": [-5, -7], "expected": -12},
            {"args": [0, 0], "expected": 0},
        ],
    },
    {
        "id": "is_palindrome",
        "prompt": (
            "Write a Python function `def is_palindrome(s):` that returns True if "
            "the string s reads the same forwards and backwards, and False otherwise. "
            "Do not ignore case or spaces — compare the string exactly."
        ),
        "entry_point": "is_palindrome",
        "reference_solution": "def is_palindrome(s):\n    return s == s[::-1]\n",
        "tests": [
            {"args": ["racecar"], "expected": True},
            {"args": ["hello"], "expected": False},
            {"args": [""], "expected": True},
            {"args": ["a"], "expected": True},
            {"args": ["abba"], "expected": True},
        ],
    },
    {
        "id": "factorial",
        "prompt": (
            "Write a Python function `def factorial(n):` that returns n! (the factorial "
            "of n) for a non-negative integer n. factorial(0) is 1."
        ),
        "entry_point": "factorial",
        "reference_solution": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "tests": [
            {"args": [0], "expected": 1},
            {"args": [1], "expected": 1},
            {"args": [5], "expected": 120},
            {"args": [10], "expected": 3628800},
        ],
    },
    {
        "id": "fibonacci",
        "prompt": (
            "Write a Python function `def fibonacci(n):` that returns the nth Fibonacci "
            "number, 0-indexed, where fibonacci(0) == 0 and fibonacci(1) == 1."
        ),
        "entry_point": "fibonacci",
        "reference_solution": (
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        "tests": [
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [2], "expected": 1},
            {"args": [10], "expected": 55},
            {"args": [15], "expected": 610},
        ],
    },
    {
        "id": "is_prime",
        "prompt": (
            "Write a Python function `def is_prime(n):` that returns True if n is a "
            "prime number and False otherwise. n is a non-negative integer."
        ),
        "entry_point": "is_prime",
        "reference_solution": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True\n"
        ),
        "tests": [
            {"args": [0], "expected": False},
            {"args": [1], "expected": False},
            {"args": [2], "expected": True},
            {"args": [17], "expected": True},
            {"args": [18], "expected": False},
            {"args": [97], "expected": True},
        ],
    },
    {
        "id": "reverse_words",
        "prompt": (
            "Write a Python function `def reverse_words(s):` that takes a sentence s "
            "with words separated by single spaces and returns a new string with the "
            "words in reverse order, still separated by single spaces. "
            "e.g. 'hello world foo' -> 'foo world hello'."
        ),
        "entry_point": "reverse_words",
        "reference_solution": "def reverse_words(s):\n    return ' '.join(s.split(' ')[::-1])\n",
        "tests": [
            {"args": ["hello world"], "expected": "world hello"},
            {"args": ["hello world foo"], "expected": "foo world hello"},
            {"args": ["single"], "expected": "single"},
        ],
    },
    {
        "id": "count_vowels",
        "prompt": (
            "Write a Python function `def count_vowels(s):` that returns the number of "
            "vowels (a, e, i, o, u, case-insensitive) in string s."
        ),
        "entry_point": "count_vowels",
        "reference_solution": (
            "def count_vowels(s):\n"
            "    return sum(1 for ch in s.lower() if ch in 'aeiou')\n"
        ),
        "tests": [
            {"args": ["hello"], "expected": 2},
            {"args": ["AEIOU"], "expected": 5},
            {"args": ["xyz"], "expected": 0},
            {"args": [""], "expected": 0},
        ],
    },
    {
        "id": "max_subarray_sum",
        "prompt": (
            "Write a Python function `def max_subarray_sum(nums):` that returns the "
            "largest sum of any contiguous, non-empty subarray of the list nums "
            "(Kadane's algorithm). nums contains at least one element and may contain "
            "negative numbers."
        ),
        "entry_point": "max_subarray_sum",
        "reference_solution": (
            "def max_subarray_sum(nums):\n"
            "    best = current = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        current = max(x, current + x)\n"
            "        best = max(best, current)\n"
            "    return best\n"
        ),
        "tests": [
            {"args": [[1, 2, 3, 4]], "expected": 10},
            {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"args": [[-1, -2, -3]], "expected": -1},
            {"args": [[5]], "expected": 5},
        ],
    },
]


def verify_problems() -> None:
    """Sanity-check every reference_solution actually passes its own tests.

    Not called during normal benchmark runs — run manually after editing this
    file to catch authoring mistakes before they're blamed on the model.
    """
    for problem in PROBLEMS:
        namespace: dict = {}
        exec(problem["reference_solution"], namespace)
        fn = namespace[problem["entry_point"]]
        for test in problem["tests"]:
            actual = fn(*test["args"])
            assert actual == test["expected"], (
                f"{problem['id']}: reference solution gave {actual!r}, "
                f"expected {test['expected']!r} for args {test['args']!r}"
            )
    print(f"All {len(PROBLEMS)} reference solutions pass their own tests.")


if __name__ == "__main__":
    verify_problems()
