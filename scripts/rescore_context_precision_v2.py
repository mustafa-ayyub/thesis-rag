#!/usr/bin/env python3
"""Rescore System A context precision with partial credit and write v2 outputs."""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create partial-credit context precision v2 for System A."
    )
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.evaluation.context_precision_v2 import PartialCreditContextPrecision
    from src.evaluation.ragas_scorer import RagasScores
    from src.evaluation.runner import EvaluationRunner
    from src.saia_rate_limiter import SaiaRateLimiter
    from src.settings import load_settings

    cfg = load_settings(args.config)
    results_dir = Path(cfg.evaluation.system_a_results_dir)
    answers_path = results_dir / cfg.evaluation.answers_filename
    scores_v1_path = results_dir / cfg.evaluation.ragas_scores_filename
    scores_v2_path = results_dir / "ragas_scores_v2.json"
    intermediate_path = results_dir / "context_precision_v2.json"
    final_v2_path = results_dir / "evaluation_final_v2.json"

    queries = json.loads(Path(cfg.data.queries_path).read_text(encoding="utf-8"))
    first_runs: dict[str, dict] = {}
    with answers_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["run"] == 1:
                first_runs[record["query_id"]] = record

    scores_v1 = json.loads(scores_v1_path.read_text(encoding="utf-8"))
    intermediate: dict[str, dict] = {}
    if intermediate_path.exists():
        intermediate = json.loads(intermediate_path.read_text(encoding="utf-8"))

    scorer = PartialCreditContextPrecision(
        cfg,
        SaiaRateLimiter.from_settings(cfg),
    )

    for query in queries:
        query_id = query["query_id"]
        if query["query_type"] == "no_retrieval":
            continue
        if scores_v1.get(query_id) is None:
            continue
        if (
            query_id in intermediate
            and intermediate[query_id].get("context_precision") is not None
        ):
            continue

        record = first_runs[query_id]
        contexts = [
            (
                f"{chunk['doc_name']}, p.{chunk['page_number']}",
                chunk["text"],
            )
            for chunk in record["retrieved_chunks"]
        ]
        score, verdicts = scorer.score(
            question=record["question"],
            reference=query["ground_truth"],
            contexts=contexts,
        )
        intermediate[query_id] = {
            "context_precision": score,
            "verdicts": verdicts,
        }
        intermediate_path.write_text(
            json.dumps(intermediate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Partial-credit context precision %s: %.4f verdicts=%s",
            query_id,
            score,
            verdicts,
        )

    scores_v2 = dict(scores_v1)
    for query_id, payload in intermediate.items():
        if scores_v2.get(query_id) is None:
            continue
        updated = RagasScores.model_validate(scores_v2[query_id])
        updated.context_precision = payload["context_precision"]
        scores_v2[query_id] = updated.model_dump()

    scores_v2_path.write_text(
        json.dumps(scores_v2, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    runner = EvaluationRunner(cfg)
    runner._ragas_scores_path = scores_v2_path
    runner._final_path = final_v2_path
    runner.merge()

    print(f"Wrote {scores_v2_path}")
    print(f"Wrote {final_v2_path}")


if __name__ == "__main__":
    main()
