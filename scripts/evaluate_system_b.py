#!/usr/bin/env python3
"""Run System B generation, RAGAS scoring, and final merge.

Usage:
    python scripts/evaluate_system_b.py --phase generate
    python scripts/evaluate_system_b.py --phase score
    python scripts/evaluate_system_b.py --phase merge
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_PHASES = ("generate", "score", "merge")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate System B (MCP-wrapped RAG agent)."
    )
    parser.add_argument("--phase", required=True, choices=_PHASES)
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.evaluation.runner_system_b import SystemBEvaluationRunner
    from src.settings import load_settings

    cfg = load_settings(args.config)
    runner = SystemBEvaluationRunner(cfg)

    if args.phase == "generate":
        runner.generate()
    elif args.phase == "score":
        runner.score()
    else:
        runner.merge()


if __name__ == "__main__":
    main()
