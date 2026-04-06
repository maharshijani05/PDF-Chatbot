#!/usr/bin/env python
"""
LLM-as-Judge evaluator for the RAG system.
Uses an LLM to evaluate RAG responses on 5 binary criteria.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import bootstrap_environment
bootstrap_environment()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


EVALUATION_PROMPT = ChatPromptTemplate.from_template("""
You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Evaluate the following answer on 5 binary criteria (PASS or FAIL):

1. **Answer Quality**: Does the answer directly address the question? Is it relevant and substantive? (not just "I don't know" when information is available)

2. **Source Attribution**: Does the answer include a source reference (e.g., "Source:", "According to...", quotes, page references)? 

3. **Hallucination Check**: Does the answer stick exclusively to document content without introducing false or fabricated information?

4. **Format Consistency**: Does the answer follow a structured format with clear Answer and Source sections?

5. **Completeness**: Does the answer fully address the question asked?

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object, no markdown or extra text:
{{
  "answer_quality": "PASS" or "FAIL",
  "source_attribution": "PASS" or "FAIL",
  "hallucination_check": "PASS" or "FAIL",
  "format_consistency": "PASS" or "FAIL",
  "completeness": "PASS" or "FAIL",
  "failed_criteria": ["list of criteria that failed, empty if all pass"],
  "notes": "<brief explanation of failures if any>"
}}
""")


def evaluate_with_llm(question: str, answer: str) -> Tuple[bool, Dict]:
    """
    Use LLM to evaluate a RAG response against 5 binary criteria.
    Returns: (all_pass, criteria_dict)
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )
    
    chain = EVALUATION_PROMPT | llm
    
    try:
        response = chain.invoke({
            "question": question,
            "answer": answer
        })
        
        # Parse the JSON response
        response_text = response.content.strip()
        
        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        result = json.loads(response_text)
        
        # Check if all criteria passed
        criteria = {
            "answer_quality": result.get("answer_quality", "FAIL"),
            "source_attribution": result.get("source_attribution", "FAIL"),
            "hallucination_check": result.get("hallucination_check", "FAIL"),
            "format_consistency": result.get("format_consistency", "FAIL"),
            "completeness": result.get("completeness", "FAIL")
        }
        
        all_pass = all(v == "PASS" for v in criteria.values())
        
        criteria["failed_criteria"] = result.get("failed_criteria", [])
        criteria["notes"] = result.get("notes", "")
        
        return all_pass, criteria
        
    except Exception as e:
        return False, {
            "answer_quality": "FAIL",
            "source_attribution": "FAIL",
            "hallucination_check": "FAIL",
            "format_consistency": "FAIL",
            "completeness": "FAIL",
            "failed_criteria": ["evaluation_error"],
            "notes": f"Evaluation error: {str(e)}"
        }


def run_llm_judge_evaluation(dataset_path: str) -> None:
    """Run LLM-as-judge evaluation on dataset."""
    from evals.runner import load_dataset
    from rag import build_qa_chain
    import time
    
    data = load_dataset(dataset_path)
    qa_chain = build_qa_chain()
    
    results = []
    passed = 0
    failed = 0
    
    print("=" * 80)
    print("LLM-AS-JUDGE EVALUATION (Binary Criteria)")
    print("=" * 80)
    
    for idx, item in enumerate(data, start=1):
        question = item.get("question", "")
        if not question:
            print(f"[SKIP] Test {idx}: No question")
            continue
        
        try:
            print(f"\n[TEST {idx}] {question}")
            
            # Get RAG response
            answer = qa_chain.invoke(question)
            print(f"\n  📝 ANSWER:")
            print(f"  {answer}")
            
            # Evaluate with LLM
            all_pass, criteria = evaluate_with_llm(question, answer)
            
            results.append({
                "question": question,
                "answer": answer,
                "passed": all_pass,
                "criteria": criteria,
                "category": item.get("category", "normal")
            })
            
            if all_pass:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL"
            
            print(f"{status}")
            
            # Show detailed criteria
            print("  Criteria Results:")
            for criterion, value in criteria.items():
                if criterion not in ["failed_criteria", "notes"]:
                    icon = "✓" if value == "PASS" else "✗"
                    criterion_name = criterion.replace("_", " ").title()
                    print(f"    [{icon}] {criterion_name}: {value}")
            
            if criteria.get("failed_criteria"):
                print(f"  ❌ Failed Criteria: {', '.join(criteria['failed_criteria'])}")
            
            if criteria.get("notes"):
                print(f"\n  📋 EVALUATION NOTES:")
                print(f"  {criteria['notes']}")
            
            # Delay to avoid rate limits
            if idx < len(data):
                print("  Waiting 10 seconds...")
                time.sleep(10)
            
        except Exception as e:
            print(f"[ERROR] Test {idx}: {e}")
            failed += 1
            results.append({
                "question": question,
                "passed": False,
                "criteria": {
                    "answer_quality": "FAIL",
                    "source_attribution": "FAIL",
                    "hallucination_check": "FAIL",
                    "format_consistency": "FAIL",
                    "completeness": "FAIL",
                    "failed_criteria": ["execution_error"],
                    "notes": str(e)
                },
                "category": item.get("category", "normal")
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    pass_rate = (passed / len(results) * 100) if results else 0
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}/{len(results)} ({pass_rate:.1f}%)")
    print(f"Failed: {failed}/{len(results)}")
    
    # Detailed results
    print("\nDETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"\n{i}. [{status}] [{result['category']}]")
        print(f"   Question: {result['question']}")
        print(f"   Answer: {result['answer']}")
        
        criteria = result["criteria"]
        for criterion, value in criteria.items():
            if criterion not in ["failed_criteria", "notes"]:
                icon = "✓" if value == "PASS" else "✗"
                criterion_name = criterion.replace("_", " ").title()
                print(f"   [{icon}] {criterion_name}: {value}")
        
        if criteria.get("failed_criteria"):
            print(f"   Failed Criteria: {', '.join(criteria['failed_criteria'])}")
        
        
        if criteria.get("notes"):
            print(f"   Reason: {criteria['notes']}")


if __name__ == "__main__":
    run_llm_judge_evaluation("evals/dataset.json")
