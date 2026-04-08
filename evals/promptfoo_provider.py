"""
Promptfoo custom Python provider.
Pipeline: question -> RAG chain (strict or concise variant) -> answer -> LLM judge -> JSON result
"""
import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import bootstrap_environment
bootstrap_environment()

from rag import build_qa_chain
from evals.evaluator_llm import evaluate_with_llm

_qa_chains: dict = {}


def call_api(prompt: str, options: dict, context: dict) -> dict:
    config = options.get("config", {})
    variant = config.get("variant", "strict")

    if variant not in _qa_chains:
        _qa_chains[variant] = build_qa_chain(variant=variant)

    question = prompt.strip()
    if not question:
        return {"output": json.dumps({"error": "No question provided"})}

    # Step 1: get RAG answer
    time.sleep(10)
    answer = _qa_chains[variant].invoke(question)

    # Step 2: LLM judge evaluates the answer
    time.sleep(10)
    all_pass, criteria = evaluate_with_llm(question, answer)

    return {
        "output": json.dumps({
            "passed": all_pass,
            "answer": answer,
            "answer_quality": criteria["answer_quality"],
            "source_attribution": criteria["source_attribution"],
            "hallucination_check": criteria["hallucination_check"],
            "format_consistency": criteria["format_consistency"],
            "completeness": criteria["completeness"],
            "failed_criteria": criteria.get("failed_criteria", []),
            "notes": criteria.get("notes", ""),
        }, indent=2)
    }
