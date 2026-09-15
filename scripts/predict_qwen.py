"""Run the Qwen prediction stage against a new CR using learned organization knowledge."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_cab.brain_loop import AgenticReasoningLoop
from pre_cab.env_loader import load_dotenv
from pre_cab.input_loader import load_cr_records, normalize_cr_record, record_type, source_id
from pre_cab.runtime_provider import build_runtime_provider
from pre_cab.schemas import AgentContext, Strictness


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Predict CAB readiness for a new CR with learned local Qwen knowledge")
    parser.add_argument("input", type=Path, help="JSON file containing one new CR (or a list; first CR is used unless --limit is set)")
    parser.add_argument("--knowledge", type=Path, default=Path("training/output/qwen_learning.json"))
    parser.add_argument("--provider", choices=["ollama"], default="ollama")
    parser.add_argument("--strictness", choices=[s.value for s in Strictness], default="balanced")
    parser.add_argument("--mode", choices=["single", "dual"], default="single")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen_prediction"))
    args = parser.parse_args()

    if not args.knowledge.exists():
        raise SystemExit(f"Learned knowledge not found: {args.knowledge}. Run training/learn_qwen.py first.")

    # brain_loop accepts both names for backwards compatibility; set both so the
    # learned artifact is guaranteed to reach the reasoning payload.
    os.environ["PRE_CAB_LEARNED_KNOWLEDGE"] = str(args.knowledge)
    os.environ["PRE_CAB_TRAINING_CONTEXT"] = str(args.knowledge)
    os.environ["PRE_CAB_REASONING_MODE"] = args.mode

    records = load_cr_records(args.input)
    records = [record for record in records if record_type(record) != "emergency"]
    if not records:
        raise SystemExit("No Normal CR records found in the prediction input.")
    records = records[: max(1, args.limit)]

    model = build_runtime_provider(args.provider)
    loop = AgenticReasoningLoop(model=model, memory=None, mode=args.mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    for index, raw in enumerate(records, 1):
        cr = normalize_cr_record(raw)
        number = source_id(cr)
        print(f"[PREDICT {index}/{len(records)}] {number} -> {model.model_name}", flush=True)
        context = AgentContext(cr=cr, strictness=Strictness(args.strictness))
        result = loop.run(context)
        try:
            initial = json.loads(result.initial.text)
        except json.JSONDecodeError:
            initial = {"raw_model_output": result.initial.text}
        item: dict[str, Any] = {
            "cr_number": number,
            "model": model.model_name,
            "mode": result.mode,
            "prediction": initial,
            "knowledge_artifact": str(args.knowledge),
            "learned_stage_is_separate": True,
        }
        if result.critique is not None:
            try:
                item["critique"] = json.loads(result.critique.text)
            except json.JSONDecodeError:
                item["critique"] = {"raw_model_output": result.critique.text}
        outputs.append(item)
        (args.output_dir / f"{number}_qwen_prediction.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(initial, ensure_ascii=False, indent=2))

    (args.output_dir / "batch_results.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
