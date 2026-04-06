import re
from typing import List


def normalize(text: str) -> str:
    return text.lower()


def has_answer(output: str) -> bool:
    cleaned = output.strip()
    return bool(cleaned) and len(cleaned.split()) >= 4


def has_support_reference(output: str) -> bool:
    patterns = [
        r"\b(source|reference|citation)\b",
        r"\baccording to\b",
        r"\bas stated\b",
        r"\bparagraph\b",
        r"\bsentence\b",
        r"\bsection\b",
        r"\bpage\b",
        r'\".+?\"',
        r"'.+?'",
    ]
    return any(re.search(pattern, output, re.I) for pattern in patterns)


def has_consistent_format(output: str) -> bool:
    answer_match = re.search(r"answer\s*:\s*.+", output, re.I | re.S)
    source_match = re.search(r"(source|reference)\s*:\s*.+", output, re.I | re.S)
    return bool(answer_match and source_match)


def evaluate_output(output: str) -> bool:
    return has_answer(output) and has_support_reference(output) and has_consistent_format(output)