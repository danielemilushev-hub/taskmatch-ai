"""JSON schema compliance problems: prompt -> schema the model's JSON must satisfy."""

PROBLEMS = [
    {
        "id": "flat_object",
        "task": "Generate a JSON object describing a fictional person with fields: "
        "name (string), age (integer), email (string).",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
            "additionalProperties": True,
        },
    },
    {
        "id": "enum_constraint",
        "task": "Generate a JSON object for a support ticket with fields: "
        "id (integer), priority (one of: low, medium, high, critical), resolved (boolean).",
        "schema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "resolved": {"type": "boolean"},
            },
            "required": ["id", "priority", "resolved"],
        },
    },
    {
        "id": "nested_object",
        "task": "Generate a JSON object for an order with fields: order_id (integer), "
        "and customer (an object with name (string) and address (an object with "
        "street (string) and zip_code (string)).",
        "schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "integer"},
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "zip_code": {"type": "string"},
                            },
                            "required": ["street", "zip_code"],
                        },
                    },
                    "required": ["name", "address"],
                },
            },
            "required": ["order_id", "customer"],
        },
    },
    {
        "id": "array_of_objects",
        "task": "Generate a JSON object with a single field 'items' that is an array "
        "of exactly 3 objects, each with fields: sku (string) and quantity (integer, "
        "minimum 1).",
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer", "minimum": 1},
                        },
                        "required": ["sku", "quantity"],
                    },
                }
            },
            "required": ["items"],
        },
    },
    {
        "id": "number_range_and_pattern",
        "task": "Generate a JSON object for a product review with fields: "
        "rating (integer between 1 and 5 inclusive), and reviewer_code (a string "
        "matching the pattern of exactly 3 uppercase letters followed by 4 digits, "
        "e.g. 'ABC1234').",
        "schema": {
            "type": "object",
            "properties": {
                "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                "reviewer_code": {"type": "string", "pattern": "^[A-Z]{3}[0-9]{4}$"},
            },
            "required": ["rating", "reviewer_code"],
        },
    },
]
