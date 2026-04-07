#!/usr/bin/env python
"""
Main evaluation runner with mode selection between traditional, LLM-as-judge, and RAGAS.
Usage: python run_evaluation.py [--mode traditional|llm-judge|ragas]
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
        choices=["traditional", "llm-judge", "ragas"],
        default="ragas",
        help="Evaluation mode (default: ragas)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "llm-judge":
        from evals.evaluator_llm import run_llm_judge_evaluation
        run_llm_judge_evaluation("evals/dataset.json")
    elif args.mode == "traditional":
        from evals.runner import run_live_evaluation
        run_live_evaluation("evals/dataset.json")
    else:  # ragas
        from evals.evaluator_ragas import run_ragas_evaluation
        run_ragas_evaluation("evals/dataset.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
