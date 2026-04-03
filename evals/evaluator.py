from typing import List


CORE_GROUPS: List[List[str]] = [
    ["dataset", "data", "traces"],
    ["error analysis", "analyze errors", "failure analysis"]
]

OPTIONAL_GROUPS: List[List[str]] = [
    ["evaluator", "evaluation", "judge"],
    ["llm-as-a-judge", "llm judge"],
    ["regex", "code-based", "assertion"],
    ["iteration", "refinement", "improve"]
]


def normalize(text: str) -> str:
    return text.lower()


def matches_group(text: str, group: List[str]) -> bool:
    return any(keyword in text for keyword in group)


def evaluate_output(output: str) -> bool:
    text = normalize(output)

    core_ok = all(matches_group(text, g) for g in CORE_GROUPS)
    optional_ok = any(matches_group(text, g) for g in OPTIONAL_GROUPS)

    return core_ok and optional_ok