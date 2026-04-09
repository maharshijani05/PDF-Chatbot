import asyncio
import json
import sys
import time
from pathlib import Path
from datasets import Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import bootstrap_environment, GOOGLE_API_KEY
bootstrap_environment()

from rag import invoke_with_context
from evals.utils import load_dataset, strip_markdown_json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class _RateLimitedGemini(ChatGoogleGenerativeAI):
    """
    Throttles every LLM call at the lowest level (_generate/_agenerate) so the
    rate limit applies regardless of whether RAGAS calls invoke, generate, or agenerate.
    Also strips markdown fences from JSON responses at the same level.
    ~8 RPM, safely under the 10 RPM free-tier limit.
    """

    @staticmethod
    def _clean(result):
        from langchain_core.outputs import ChatGeneration
        for item in result.generations:
            gens = item if isinstance(item, list) else [item]
            for gen in gens:
                if not isinstance(gen, ChatGeneration):
                    continue
                content = gen.message.content
                if isinstance(content, str) and "```" in content:
                    cleaned = strip_markdown_json(content)
                    gen.message = gen.message.model_copy(update={"content": cleaned})
                    gen.text = cleaned
        return result

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(7)
        return self._clean(super()._generate(messages, stop, run_manager, **kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(7)
        return self._clean(await super()._agenerate(messages, stop, run_manager, **kwargs))


def collect_rag_data(dataset_path: str) -> Dataset:
    data = load_dataset(dataset_path)
    questions, answers, contexts = [], [], []

    for idx, item in enumerate(data):
        question = item.get("question")
        if not question:
            continue

        print(f"Waiting 10 seconds before question {idx + 1}...")
        time.sleep(10)

        answer, context_list = invoke_with_context(question)

        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)

        print(f"Processed {idx + 1}/{len(data)}")

    return Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts})


def run_ragas_evaluation(dataset_path: str) -> None:
    print("Collecting RAG data...")
    dataset = collect_rag_data(dataset_path)

    print("Running RAGAS evaluation with Gemini...")

    llm = _RateLimitedGemini(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
        convert_system_message_to_human=True,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
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
            score = None
            try:
                raw = result[metric_name]  # dict-like access (RAGAS >= 0.2)
                score = sum(raw) / len(raw) if isinstance(raw, list) else float(raw)
            except Exception:
                score = getattr(result, metric_name, None)
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
