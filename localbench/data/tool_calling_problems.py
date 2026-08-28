"""Procedurally generated tool / function calling problems.

Evaluates:
1. Tool Selection: Did the model call the right tool when multiple tools are provided?
2. Argument Extraction & Schema Adherence: Are arguments valid JSON and conforming to the parameter types, enums, and required fields?
3. Parameter Precision: Did the model extract the correct entities specified in the prompt?
4. Negative Control / Tool Refusal: When given tools but asked a general knowledge question, did the model refrain from calling tools unnecessarily?
"""

from __future__ import annotations

import random
from typing import Any

_CITIES = ["Tokyo", "Berlin", "San Francisco", "London", "Paris", "Toronto", "Sydney", "Singapore"]
_TABLES = ["users", "orders", "products", "invoices", "transactions", "employees"]
_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]
_CABIN_CLASSES = ["economy", "premium_economy", "business", "first"]
_DEVICES = ["living_room_light", "kitchen_thermostat", "bedroom_fan", "garage_door", "hallway_lamp"]


def _get_tools_catalog() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch current weather conditions and forecast for a specific location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name or coordinates"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        "days": {"type": "integer", "minimum": 1, "maximum": 7},
                    },
                    "required": ["location", "unit"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Query structured database tables with filters and limits.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "enum": _TABLES},
                        "query": {"type": "string", "description": "Search keyword or filter term"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["table", "query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert money between currencies using live market rates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "minimum": 0.01},
                        "from_currency": {"type": "string", "enum": _CURRENCIES},
                        "to_currency": {"type": "string", "enum": _CURRENCIES},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "control_device",
                "description": "Turn smart home devices on or off and set levels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string", "enum": _DEVICES},
                        "state": {"type": "string", "enum": ["on", "off"]},
                        "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                    "required": ["device_id", "state"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_flight",
                "description": "Search and book airline tickets between cities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Departure city"},
                        "destination": {"type": "string", "description": "Arrival city"},
                        "passengers": {"type": "integer", "minimum": 1, "maximum": 9},
                        "cabin_class": {"type": "string", "enum": _CABIN_CLASSES},
                    },
                    "required": ["origin", "destination", "passengers"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_mortgage",
                "description": "Calculate monthly payment and interest for a fixed-rate loan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "principal": {"type": "number", "minimum": 1000},
                        "annual_rate": {"type": "number", "minimum": 0.1, "maximum": 30.0},
                        "term_years": {"type": "integer", "enum": [10, 15, 20, 30]},
                    },
                    "required": ["principal", "annual_rate", "term_years"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def generate_tool_calling_problems(num_problems: int, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tools_catalog = _get_tools_catalog()
    problems = []

    generators = [
        _gen_weather_problem,
        _gen_db_search_problem,
        _gen_currency_problem,
        _gen_device_problem,
        _gen_flight_problem,
        _gen_mortgage_problem,
        _gen_negative_control_problem,
    ]

    for i in range(num_problems):
        gen_fn = generators[i % len(generators)]
        problem = gen_fn(rng, i, tools_catalog)
        problems.append(problem)

    return problems


def _gen_weather_problem(rng: random.Random, idx: int, catalog: list[dict]) -> dict[str, Any]:
    city = rng.choice(_CITIES)
    unit = rng.choice(["celsius", "fahrenheit"])
    days = rng.randint(1, 5)
    prompt = f"What is the {days}-day weather forecast for {city} in {unit}?"
    distractors = [t for t in catalog if t["function"]["name"] != "get_weather"]
    tools = [catalog[0]] + rng.sample(distractors, 2)
    rng.shuffle(tools)

    return {
        "id": f"tool_weather_{idx}",
        "prompt": prompt,
        "tools": tools,
        "expected_tool": "get_weather",
        "is_negative": False,
        "validator_type": "weather",
        "expected_args": {
            "location": city,
            "unit": unit,
            "days": days,
        },
    }


def _gen_db_search_problem(rng: random.Random, idx: int, catalog: list[dict]) -> dict[str, Any]:
    table = rng.choice(_TABLES)
    query = rng.choice(["active_status", "pending_review", "vip_customer", "electronics_q3", "urgent_ticket"])
    limit = rng.choice([5, 10, 25, 50])
    prompt = f"Please search the '{table}' database table for records matching '{query}', limiting the results to {limit} rows."
    distractors = [t for t in catalog if t["function"]["name"] != "search_database"]
    target = [t for t in catalog if t["function"]["name"] == "search_database"][0]
    tools = [target] + rng.sample(distractors, 2)
    rng.shuffle(tools)

    return {
        "id": f"tool_db_{idx}",
        "prompt": prompt,
        "tools": tools,
        "expected_tool": "search_database",
        "is_negative": False,
        "validator_type": "search_database",
        "expected_args": {
            "table": table,
            "query": query,
            "limit": limit,
        },
    }


def _gen_currency_problem(rng: random.Random, idx: int, catalog: list[dict]) -> dict[str, Any]:
    amount = round(rng.uniform(10.0, 5000.0), 2)
    from_curr, to_curr = rng.sample(_CURRENCIES, 2)
    prompt = f"Convert {amount} {from_curr} into {to_curr}."
    target = [t for t in catalog if t["function"]["name"] == "convert_currency"][0]
    distractors = [t for t in catalog if t["function"]["name"] != "convert_currency"]
    tools = [target] + rng.sample(distractors, 2)
    rng.shuffle(tools)

    return {
        "id": f"tool_currency_{idx}",
        "prompt": prompt,
        "tools": tools,
        "expected_tool": "convert_currency",
        "is_negative": False,
        "validator_type": "convert_currency",
        "expected_args": {
            "amount": amount,
            "from_currency": from_curr,
            "to_currency": to_curr,
        },
    }


def _gen_device_problem(rng: random.Random, idx: int, catalog: list[dict]) -> dict[str, Any]:
    device = rng.choice(_DEVICES)
    state = rng.choice(["on", "off"])
    brightness = rng.randint(20, 90) if state == "on" and "light" in device else None
    if brightness:
        prompt = f"Turn {state} the {device.replace('_', ' ')} and set its brightness to {brightness}%."
    else:
        prompt = f"Turn {state} the {device.replace('_', ' ')}."
    target = [t for t in catalog if t["function"]["name"] == "control_device"][0]
    distractors = [t for t in catalog if t["function"]["name"] != "control_device"]
    tools = [target] + rng.sample(distractors, 2)
    rng.shuffle(tools)

    return {
        "id": f"tool_device_{idx}",
        "prompt": prompt,
        "tools": tools,
        "expected_tool": "control_device",
        "is_negative": False,
        "validator_type": "control_device",
        "expected_args": {
            "device_id": device,
            "state": state,
            "brightness": brightness,
        },
    }


def _gen_flight_problem(rng: random.Random, idx: int, catalog: list[dict]) -> dict[str, Any]:
    orig, dest = rng.sample(_CITIES, 2)
    passengers = rng.randint(1, 4)
    cabin = rng.choice(_CABIN_CLASSES)
    prompt = f"Book a flight from {orig} to {dest} for {passengers} passenger(s) in {cabin} class."
    target = [t for t in catalog if t["function"]["name"] == "book_flight"][0]
    distractors = [t for t in catalog if t["function"]["name"] != "book_flight"]
    tools = [target] + rng.sample(distractors, 2)
    rng.shuffle(tools)

    return {
        "id": f"tool_flight_{idx}",
        "prompt": prompt,
        "tools": tools,
        "expected_tool": "book_flight",
        "is_negative": False,
        "validator_type": "book_flight",
        "expected_args": {
            "origin": orig,
            "destination": dest,
            "passengers": passengers,
            "cabin_class": cabin,
        },
    }


def _gen_mortgage_problem(rng: random.Random, idx: int, catalog: list[dict]) -> dict[str, Any]:
    principal = rng.choice([250000, 350000, 500000, 750000])
    annual_rate = rng.choice([3.5, 4.25, 5.0, 6.5])
    term = rng.choice([15, 20, 30])
    prompt = f"Calculate the monthly mortgage payment for a loan of ${principal:,} at {annual_rate}% annual interest over {term} years."
    target = [t for t in catalog if t["function"]["name"] == "calculate_mortgage"][0]
    distractors = [t for t in catalog if t["function"]["name"] != "calculate_mortgage"]
    tools = [target] + rng.sample(distractors, 2)
    rng.shuffle(tools)

    return {
        "id": f"tool_mortgage_{idx}",
        "prompt": prompt,
        "tools": tools,
        "expected_tool": "calculate_mortgage",
        "is_negative": False,
        "validator_type": "calculate_mortgage",
        "expected_args": {
            "principal": principal,
            "annual_rate": annual_rate,
            "term_years": term,
        },
    }


def _gen_negative_control_problem(rng: random.Random, idx: int, catalog: list[dict]) -> dict[str, Any]:
    qa_pairs = [
        ("What is the chemical symbol for Gold?", "Au"),
        ("Who wrote the play Romeo and Juliet?", "William Shakespeare"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("How many continents are there on Earth?", "7"),
        ("In what year was the United Nations established?", "1945"),
    ]
    question, _ = rng.choice(qa_pairs)
    prompt = f"{question} (Answer directly in text. Do not call any tools.)"
    tools = rng.sample(catalog, 3)

    return {
        "id": f"tool_negative_{idx}",
        "prompt": prompt,
        "tools": tools,
        "expected_tool": None,
        "is_negative": True,
        "validator_type": "negative_control",
        "expected_args": {},
    }
