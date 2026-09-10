from __future__ import annotations

import argparse
import json
from pathlib import Path

from pre_cab.batch import run_batch
from pre_cab.input_loader import load_cr_records
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
    parser.add_argument("--include-non-normal", action="store_true", help="Process every change type (Normal only is the default)")
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run attachment evidence validation; metadata-only screening is used when no attachments are supplied.",
    )
    args = parser.parse_args()

    try:
        records = load_cr_records(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to load CR JSON: {exc}") from exc

    model = build_provider(args.provider) if args.llm else None
    progress = run_batch(
        records,
        output_path=args.output,
        attachment_root=args.attachments,
        stage1_only=not args.full_validation and args.attachments is None,
        strictness=Strictness(args.strictness),
        model=model,
        normal_only=not args.include_non_normal,
        resume=not args.restart,
    )
    print(json.dumps(progress.__dict__, indent=2))


if __name__ == "__main__":
    main()
