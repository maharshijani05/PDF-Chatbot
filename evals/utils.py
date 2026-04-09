import re


def strip_markdown_json(text: str) -> str:
    """Strip markdown code fences from a JSON string."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()
