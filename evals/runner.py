import json
import time
from typing import Dict, List
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import bootstrap_environment
bootstrap_environment()

from evals.evaluator import evaluate_output
from rag import build_qa_chain


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def run_live_evaluation(dataset_path: str) -> None:
    data = load_dataset(dataset_path)

    qa_chain = build_qa_chain()

    passed = 0
    failed = 0

    for idx, item in enumerate(data, start=1):
        question = item["input"]

        try:
            model_output = qa_chain.invoke(question)
        except Exception as e:
            print(f"[ERROR] Test {idx}: {e}")
            failed += 1
            continue

        predicted = evaluate_output(model_output)

        status = "PASS" if predicted else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"[{status}] Test {idx}")
        print(f"Q: {question}")
        print(f"A: {model_output}\n")

        # Delay to avoid rate limits
        if idx < len(data):
            print("Waiting 5 seconds...\n")
            time.sleep(5)

    print("----- SUMMARY -----")
    print(f"Total: {len(data)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    run_live_evaluation("evals/dataset.json")