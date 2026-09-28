import json

from support_assistant.graph import run_assistant


EXAMPLES = [
    "How much is the delivery fee for an order below INR 149?",
    "What is the capital of India?",
]


if __name__ == "__main__":
    print("=" * 60)
    print("SUPPORT ASSISTANT EXAMPLES")
    print("=" * 60)

    for query in EXAMPLES:
        response = run_assistant(query)
        print(f"\nQuery: {query}")
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
