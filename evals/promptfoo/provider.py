"""
Promptfoo custom Python provider.
Pipeline: question -> RAG chain -> answer -> LLM judge -> JSON result

Provider config options (set via promptfooconfig.yaml):
  skip_judge: true | false  — skip LLM judge (used by regression tests)
"""
import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import bootstrap_environment
bootstrap_environment()

from rag import build_qa_chain
from evals.evaluator_llm import evaluate_with_llm

_qa_chain = None


def call_api(prompt: str, options: dict, context: dict) -> dict:
    global _qa_chain
    if _qa_chain is None:
        _qa_chain = build_qa_chain()

    skip_judge = options.get("config", {}).get("skip_judge", False)

    question = prompt.strip()
    if not question:
        return {"output": json.dumps({"error": "No question provided"})}

    # Step 1: RAG answer
    time.sleep(10)
    answer = _qa_chain.invoke(question)

    # Regression path: return raw answer, no judge
    if skip_judge:
        return {"output": json.dumps({"answer": answer})}

    # Step 2: LLM judge
    time.sleep(10)
    all_pass, criteria = evaluate_with_llm(question, answer)

    # Step 3: buffer before promptfoo fires the llm-rubric grader
    time.sleep(10)

    return {
        "output": json.dumps({
            "passed": all_pass,
            "answer": answer,
            "answer_quality": criteria["answer_quality"],
            "hallucination_check": criteria["hallucination_check"],
            "completeness": criteria["completeness"],
            "failed_criteria": criteria.get("failed_criteria", []),
            "notes": criteria.get("notes", ""),
        }, indent=2)
    }
