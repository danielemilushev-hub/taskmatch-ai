"""Procedurally generated multi-turn conversation problems.

Tests conversational state tracking, entity binding, persistent constraint
retention across intermediate distractor turns, and multi-turn state updates.
"""

from __future__ import annotations

import random
from typing import Any

_NAMES = ["Alice Smith", "Marcus Vance", "Elena Rostova", "David Chen", "Amina Al-Mansoor"]
_PROJECTS = ["PROJECT-NEBULA", "AURORA-9", "TITAN-CORE", "CYPHER-7", "APOLLO-X"]
_SECTORS = ["Sector 4", "Quadrant B", "Zone Delta", "Floor 14", "Substation 8"]
_CODES = ["ALPHA-772", "OMEGA-914", "SIGMA-303", "KAPPA-451", "ZEPHYR-88"]
_FORBIDDEN_WORDS = ["elephant", "banana", "galaxy", "pyramid", "diamond"]


def generate_multi_turn_problems(num_problems: int, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    problems = []

    generators = [
        _gen_secret_recall_dialogue,
        _gen_persistent_prefix_dialogue,
        _gen_forbidden_word_dialogue,
        _gen_state_update_dialogue,
        _gen_entity_binding_dialogue,
    ]

    for i in range(num_problems):
        gen_fn = generators[i % len(generators)]
        problem = gen_fn(rng, i)
        problems.append(problem)

    return problems


def _gen_secret_recall_dialogue(rng: random.Random, idx: int) -> dict[str, Any]:
    code = rng.choice(_CODES)
    project = rng.choice(_PROJECTS)

    turns = [
        f"Hello. We are initializing our session. Note that the clearance code for {project} is '{code}'. Please acknowledge.",
        "Before we proceed, what is the largest mammal currently living on Earth? Answer in one short sentence.",
        f"What was the clearance code for {project} that I gave you earlier? Output ONLY the exact clearance code.",
    ]

    return {
        "id": f"multi_turn_recall_{idx}",
        "turns": turns,
        "eval_type": "exact_contains",
        "expected": code,
        "description": f"Recall of clearance code across 3 turns",
    }


def _gen_persistent_prefix_dialogue(rng: random.Random, idx: int) -> dict[str, Any]:
    tag = rng.choice(["[INTERNAL_ONLY]", "[SECURE_LOG]", "[VERIFIED]", "[AUDIT_TRAIL]"])

    turns = [
        f"For this entire session, you MUST begin EVERY single response with the exact tag '{tag}'. Acknowledge this rule.",
        "What is 15 multiplied by 14? Answer briefly.",
        "Name the capital city of Australia.",
    ]

    return {
        "id": f"multi_turn_prefix_{idx}",
        "turns": turns,
        "eval_type": "starts_with",
        "expected": tag,
        "description": f"Persistent prefix constraint across 3 turns",
    }


def _gen_forbidden_word_dialogue(rng: random.Random, idx: int) -> dict[str, Any]:
    word = rng.choice(_FORBIDDEN_WORDS)

    turns = [
        f"Important instruction: throughout this entire conversation, never use the word '{word}' in any of your responses, even if directly asked. Acknowledge.",
        "Give me a quick 2-sentence tip on effective time management.",
        f"Describe a {word} without ever writing the actual word '{word}'. Keep it under 2 sentences.",
    ]

    return {
        "id": f"multi_turn_forbidden_{idx}",
        "turns": turns,
        "eval_type": "forbidden_word",
        "expected": word,
        "description": f"Persistent negative constraint across 3 turns",
    }


def _gen_state_update_dialogue(rng: random.Random, idx: int) -> dict[str, Any]:
    item = rng.choice(["laptop", "camera", "drone", "microscope"])
    init_qty = rng.randint(10, 20)
    sold_qty = rng.randint(2, 5)
    added_qty = rng.randint(3, 7)
    final_qty = init_qty - sold_qty + added_qty

    turns = [
        f"We have an inventory of {init_qty} {item}s in stock. Acknowledge.",
        f"We just sold {sold_qty} {item}s to a client. Also, what is the chemical formula for water?",
        f"A new shipment arrived with {added_qty} more {item}s. Exactly how many {item}s do we have in total right now? Answer with just the number.",
    ]

    return {
        "id": f"multi_turn_state_{idx}",
        "turns": turns,
        "eval_type": "numerical_exact",
        "expected": str(final_qty),
        "description": f"Multi-turn arithmetic state tracking",
    }


def _gen_entity_binding_dialogue(rng: random.Random, idx: int) -> dict[str, Any]:
    name = rng.choice(_NAMES)
    sector = rng.choice(_SECTORS)
    project = rng.choice(_PROJECTS)

    turns = [
        f"Please register the following team member: {name} has been assigned to {project} in {sector}.",
        "What is the approximate distance from the Earth to the Moon in kilometers?",
        f"Which sector is {name} working in? Output ONLY the sector name.",
    ]

    return {
        "id": f"multi_turn_entity_{idx}",
        "turns": turns,
        "eval_type": "exact_contains",
        "expected": sector,
        "description": f"Entity-attribute binding across turns",
    }
