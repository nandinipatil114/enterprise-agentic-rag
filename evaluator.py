"""Evaluate the agent: Faithfulness, Answer Relevancy, Context Recall (custom LLM judge or Ragas)."""
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from agent import answer_question
from config import BASE_DIR, LLM_MODEL, get_embeddings, get_llm

RESULTS_PATH = str(BASE_DIR / "eval_results.json")
METRICS = ["faithfulness", "answer_relevancy", "context_recall"]

GOLDEN_SET = [
    {"question": "How many days of annual leave do employees get, and how many can be carried over?",
     "reference": "Employees get 24 days of paid annual leave per year, and up to 10 unused days can be carried over (expiring March 31)."},
    {"question": "What is the minimum password length and how often must passwords be rotated?",
     "reference": "Passwords must be at least 14 characters and rotated every 90 days, with MFA mandatory."},
    {"question": "What is the maximum hotel rate per night in San Francisco?",
     "reference": "San Francisco is a tier-1 city, so hotels are capped at $250 per night."},
    {"question": "If monthly uptime drops to 98.5%, what service credit applies?",
     "reference": "A 25% service credit applies because uptime is below 99.0% but at least 95.0%."},
    {"question": "How quickly must a Severity 1 security incident be acknowledged?",
     "reference": "Severity 1 incidents must be acknowledged within 15 minutes."},
    {"question": "Who needs to pre-approve a $6,000 expense?",
     "reference": "Expenses above $5,000 require Director pre-approval."},
]


def _clean(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else round(max(0.0, min(1.0, v)), 3)


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _ragas_scores(rows: list[dict]) -> list[dict]:
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "retrieved_contexts": r["retrieved_contexts"] or ["No context retrieved."],
                "response": r["response"],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1), LLMContextRecall()],
        llm=LangchainLLMWrapper(get_llm()),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        run_config=RunConfig(max_workers=2, timeout=180),
        show_progress=False,
        raise_exceptions=False,
    )
    return [{m: _clean(s.get(m)) for m in METRICS} for s in result.scores]


class Judge(BaseModel):
    faithfulness: float = Field(description="0-1: fraction of answer claims supported by the contexts")
    answer_relevancy: float = Field(description="0-1: how directly the answer addresses the question")
    context_recall: float = Field(description="0-1: fraction of the reference's facts present in the contexts")


def _custom_scores(rows: list[dict]) -> list[dict]:
    """LLM-as-judge scoring; the most reliable option on Groq."""
    judge = get_llm().with_structured_output(Judge, method="function_calling")
    scores = []
    for r in rows:
        prompt = (
            "You are a strict RAG evaluator. Score each metric from 0.0 to 1.0.\n\n"
            f"QUESTION: {r['question']}\nREFERENCE ANSWER: {r['reference']}\n"
            f"MODEL ANSWER: {r['response']}\nRETRIEVED CONTEXTS:\n" + "\n---\n".join(r["retrieved_contexts"] or ["(none)"])
        )
        try:
            verdict = judge.invoke(prompt)
        except Exception:
            verdict = None
        scores.append({m: _clean(getattr(verdict, m)) if verdict else None for m in METRICS})
        time.sleep(0.5)  # be gentle with Groq's per-minute limits
    return scores


def run_evaluation(backend: str = "custom", on_progress=None) -> dict:
    rows = []
    for i, item in enumerate(GOLDEN_SET, 1):
        out = answer_question(item["question"])
        rows.append(
            {
                "question": item["question"],
                "reference": item["reference"],
                "response": out["answer"],
                "retrieved_contexts": out["contexts"],
                "rewrites": out["rewrites"],
            }
        )
        if on_progress:
            on_progress(i / (len(GOLDEN_SET) + 1), f"Answered: {item['question']}")
    if on_progress:
        on_progress(len(GOLDEN_SET) / (len(GOLDEN_SET) + 1), f"Scoring with {backend}...")
    scores = _ragas_scores(rows) if backend == "ragas" else _custom_scores(rows)
    for row, score in zip(rows, scores):
        row.update(score)
    result = {
        "backend": backend,
        "model": LLM_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {m: _mean([r[m] for r in rows]) for m in METRICS},
        "rows": rows,
    }
    Path(RESULTS_PATH).write_text(json.dumps(result, indent=2))
    if on_progress:
        on_progress(1.0, "Done")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the agentic RAG pipeline")
    parser.add_argument("--backend", choices=["custom", "ragas"], default="custom")
    args = parser.parse_args()
    res = run_evaluation(args.backend)
    print(f"\nEvaluation summary ({res['backend']}, {res['model']}):")
    for metric, value in res["summary"].items():
        print(f"  {metric:<18} {'n/a' if value is None else f'{value:.3f}'}")
    print(f"\nDetailed results saved to {RESULTS_PATH}")