import json
import re
from typing import Dict, List


def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def strip_markdown_json(text: str) -> str:
    """Strip markdown code fences from a JSON string."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()
