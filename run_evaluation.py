#!/usr/bin/env python
"""
Main evaluation runner with mode selection between traditional and LLM-as-judge.
Usage: python run_evaluation.py [--mode traditional|llm-judge]
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--mode",
        choices=["traditional", "llm-judge"],
        default="llm-judge",
        help="Evaluation mode (default: llm-judge)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "llm-judge":
        from evals.evaluator_llm import run_llm_judge_evaluation
        run_llm_judge_evaluation("evals/dataset.json")
    else:
        from evals.runner import run_live_evaluation
        run_live_evaluation("evals/dataset.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
