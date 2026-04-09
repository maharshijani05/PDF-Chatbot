import sys
import json
import time
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import bootstrap_environment
bootstrap_environment()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from rag import build_qa_chain
from evals.utils import load_dataset, strip_markdown_json


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


def run_llm_judge_evaluation(dataset_path: str) -> None:
    data = load_dataset(dataset_path)
    qa_chain = build_qa_chain()

    passed = 0
    failed = 0

    print("=" * 80)
    print("LLM-AS-JUDGE EVALUATION")
    print("=" * 80)

    for idx, item in enumerate(data, start=1):
        question = item.get("question", "")
        if not question:
            print(f"[SKIP] Test {idx}: No question")
            continue

        print(f"\n[TEST {idx}] {question}")
        print("Waiting 10 seconds...")
        time.sleep(10)

        try:
            answer = qa_chain.invoke(question)
            print(f"\n  Answer:\n  {answer}")

            all_pass, criteria = evaluate_with_llm(question, answer)
            status = "PASS" if all_pass else "FAIL"

            if all_pass:
                passed += 1
            else:
                failed += 1

            print(f"  [{status}]")
            for criterion, value in criteria.items():
                if criterion not in ("failed_criteria", "notes"):
                    print(f"    [{'✓' if value == 'PASS' else '✗'}] {criterion.replace('_', ' ').title()}: {value}")

            if criteria.get("failed_criteria"):
                print(f"  Failed: {', '.join(criteria['failed_criteria'])}")
            if criteria.get("notes"):
                print(f"  Notes: {criteria['notes']}")

        except Exception as e:
            print(f"[ERROR] Test {idx}: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = passed + failed
    print(f"Passed: {passed}/{total} ({passed / total * 100:.1f}%)" if total else "No tests ran.")


if __name__ == "__main__":
    run_llm_judge_evaluation("evals/dataset.json")
