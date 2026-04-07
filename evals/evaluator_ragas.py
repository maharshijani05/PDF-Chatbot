import json
import sys
import time
from pathlib import Path
from datasets import Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import bootstrap_environment, GOOGLE_API_KEY
bootstrap_environment()

from rag import build_retriever, invoke_with_context
from evals.utils import load_dataset, strip_markdown_json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class _JSONCleaningLLM(ChatGoogleGenerativeAI):
    """Strips markdown code fences from Gemini JSON responses."""

    def invoke(self, input, config=None, **kwargs):
        result = super().invoke(input, config, **kwargs)
        if hasattr(result, "content"):
            result.content = strip_markdown_json(result.content)
        return result


def collect_rag_data(dataset_path: str) -> Dataset:
    data = load_dataset(dataset_path)
    retriever = build_retriever()

    questions, answers, contexts = [], [], []

    for idx, item in enumerate(data):
        question = item.get("input") or item.get("question")
        if not question:
            continue

        print(f"Waiting 10 seconds before question {idx + 1}...")
        time.sleep(10)

        answer, context_list = invoke_with_context(question, retriever)

        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)

        print(f"Processed {idx + 1}/{len(data)}")

    return Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts})


def run_ragas_evaluation(dataset_path: str) -> None:
    print("Collecting RAG data...")
    dataset = collect_rag_data(dataset_path)

    print("Running RAGAS evaluation with Gemini...")

    llm = _JSONCleaningLLM(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    metrics = [faithfulness, answer_relevancy]
    for metric in metrics:
        metric.llm = llm
        metric.embeddings = embeddings

    scores = {}
    for i, metric in enumerate(metrics):
        if i > 0:
            print("Waiting 10 seconds before next metric...")
            time.sleep(10)

        metric_name = metric.name
        print(f"Evaluating {metric_name}...")

        try:
            result = evaluate(dataset, metrics=[metric])
            score = getattr(result, metric_name, None)
            if score is None and isinstance(getattr(result, "scores", None), dict):
                score = result.scores.get(metric_name)
            if score is not None:
                scores[metric_name] = score
                print(f"{metric_name}: {score:.4f}")
            else:
                print(f"Could not extract score for {metric_name}")
        except Exception as e:
            print(f"Error evaluating {metric_name}: {e}")

    print("\nRAGAS Evaluation Results:")
    print("=" * 50)
    for name, score in scores.items():
        print(f"{name}: {score:.4f}")

    if scores:
        lowest = min(scores, key=scores.get)
        print(f"\nLowest scoring metric: {lowest} ({scores[lowest]:.4f})")

    with open("ragas_results.json", "w") as f:
        json.dump(scores or {"error": "All evaluations failed"}, f, indent=2)

    print("Results saved to ragas_results.json")
