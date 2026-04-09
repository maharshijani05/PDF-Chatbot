import json
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from evals.utils import strip_markdown_json


EVALUATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the following answer on 3 binary criteria (PASS or FAIL).
Note: format and source attribution are checked separately via deterministic string checks.

1. **Answer Quality**: Does the answer directly address the question? Is it relevant and substantive?
   If the answer is not available, it should be a clear "I don't know" without irrelevant information.

2. **Hallucination Check**: Does the answer stick exclusively to document content without introducing
   false or fabricated information?

3. **Completeness**: Does the answer fully address the question if relevant, otherwise "I don't know"?

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object, no markdown or extra text:
{{
  "answer_quality": "PASS" or "FAIL",
  "hallucination_check": "PASS" or "FAIL",
  "completeness": "PASS" or "FAIL",
  "failed_criteria": ["list of criteria that failed, empty if all pass"],
  "notes": "<brief explanation of failures if any>"
}}
""")

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
_eval_chain = EVALUATION_PROMPT | _llm


def evaluate_with_llm(question: str, answer: str) -> tuple[bool, Dict]:
    try:
        response = _eval_chain.invoke({"question": question, "answer": answer})
        result = json.loads(strip_markdown_json(response.content))

        criteria = {
            "answer_quality": result.get("answer_quality", "FAIL"),
            "hallucination_check": result.get("hallucination_check", "FAIL"),
            "completeness": result.get("completeness", "FAIL"),
        }
        all_pass = all(v == "PASS" for v in criteria.values())
        criteria["failed_criteria"] = result.get("failed_criteria", [])
        criteria["notes"] = result.get("notes", "")
        return all_pass, criteria

    except Exception as e:
        return False, {
            "answer_quality": "FAIL",
            "hallucination_check": "FAIL",
            "completeness": "FAIL",
            "failed_criteria": ["evaluation_error"],
            "notes": f"Evaluation error: {e}",
        }
