from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.batch import run_batch
from pre_cab.provider_factory import build_provider
from pre_cab.schemas import Strictness


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable batch validation for Normal CRs")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/batch_results.jsonl"))
    parser.add_argument("--attachments", type=Path, default=None)
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--provider", choices=["auto", "groq", "huggingface"], default="auto")
    parser.add_argument("--restart", action="store_true", help="Overwrite prior checkpoint instead of resuming")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("Input must contain a JSON list of CR records")

    model = build_provider(args.provider) if args.llm else None
    progress = run_batch(
        records,
        output_path=args.output,
        attachment_root=args.attachments,
        strictness=Strictness(args.strictness),
        model=model,
        resume=not args.restart,
    )
    print(json.dumps(progress.__dict__, indent=2))


if __name__ == "__main__":
    main()
