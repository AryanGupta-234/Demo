"""Kaggle-friendly smoke test and holdout evaluation for the GPT-OSS v1 adapter."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_REQUIRED_KEYS = {"facts", "technical_reasoning", "testing_reasoning", "risk_reasoning", "uncertainties", "contradictions", "cab_questions", "recommendations", "prediction"}
_LABELS = ("PASS", "CONDITIONAL", "NOT_READY")
_REASONING_KEYS = ("facts", "technical_reasoning", "testing_reasoning", "risk_reasoning")


def classification_report(golds: list[str], preds: list[str | None], labels: tuple[str, ...] = _LABELS) -> dict[str, Any]:
    """Per-class precision/recall/F1 + macro-F1, robust to unparsed predictions
    (counted as wrong for every gold label, never silently dropped -- accuracy
    alone hides collapse onto the majority class under label imbalance)."""
    confusion: dict[str, Counter[str]] = {g: Counter() for g in labels}
    for gold, pred in zip(golds, preds):
        if gold not in confusion:
            continue
        confusion[gold][pred or "UNPARSED"] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fn = sum(v for k, v in confusion[label].items() if k != label)
        fp = sum(confusion[g][label] for g in labels if g != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = tp + fn
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
    return {
        "per_class": per_class,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "confusion_matrix": {g: dict(row) for g, row in confusion.items()},
    }


def reasoning_quality(result: dict[str, Any] | None) -> dict[str, bool]:
    """Cheap, deterministic reasoning-quality signals: is the JSON schema-valid
    and are the reasoning sections actually populated, or did the model just
    emit an empty/boilerplate shell around a prediction label."""
    if not result:
        return {"schema_valid": False, "reasoning_populated": False}
    schema_valid = _REQUIRED_KEYS.issubset(result)
    reasoning_populated = any(bool(_lines(result.get(key))) for key in _REASONING_KEYS)
    return {"schema_valid": schema_valid, "reasoning_populated": reasoning_populated}


def extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def load_model(model_name_or_adapter: str, max_seq_length: int):
    """Load with explicit balanced placement; Unsloth's automatic planner can
    reject a multi-GPU layout because it reserves output-head headroom.
    Works for either a saved LoRA adapter directory or a stock base model
    name -- pass --base-model instead of --adapter for the stock-comparison
    run in the 3-way evaluation (stock / stock+context / fine-tuned)."""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_name_or_adapter),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map="balanced",
    )
    FastLanguageModel.for_inference(model)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def build_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)


def _lines(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _agent_block(label: str, items: Any) -> list[str]:
    values = _lines(items)
    return [label] + ([f"→ {item}" for item in values] if values else ["→ no model finding returned"])


def render_agent_reasoning(result: dict[str, Any]) -> str:
    return "\n\n".join("\n".join(block) for block in [_agent_block("Technical Agent", result.get("technical_reasoning")), _agent_block("Testing Agent", result.get("testing_reasoning")), _agent_block("Risk Agent", result.get("risk_reasoning"))])


def render_cab_result(cr_id: str, result: dict[str, Any], historical: str | None) -> str:
    prediction = str(result.get("prediction") or "UNPARSED")
    confidence = result.get("confidence")
    confidence_text = f"{float(confidence):.0%}" if isinstance(confidence, (int, float)) else "N/A"
    labels = {"PASS": "✅ CAB READY", "CONDITIONAL": "⚠️ CONDITIONAL - REVIEW REQUIRED", "NOT_READY": "❌ NOT READY"}
    lines = ["PRE-CAB RESULT", "─" * 34, f"CR: {cr_id}", "", f"Prediction: {labels.get(prediction, prediction)}", f"Confidence: {confidence_text}", ""]
    for fact in _lines(result.get("facts"))[:10]:
        lines.append(fact if fact.startswith(("✅", "❌", "⚠️")) else f"✅ {fact}")
    lines += ["", "Uncertainties:"]
    uncertainties = _lines(result.get("uncertainties"))
    lines += [f"   • {x}" for x in uncertainties[:6]] or ["   • None explicitly reported."]
    lines += ["", "Contradictions:"]
    contradictions = _lines(result.get("contradictions"))
    lines += [f"   • {x}" for x in contradictions[:6]] or ["   • None detected."]
    lines += ["", "Historical analysis:", f"   Historical target outcome = {historical}." if historical else "   No historical target supplied.", "", "Potential CAB questions:"]
    questions = _lines(result.get("cab_questions"))
    lines += [f"   • {x}" for x in questions[:4]] or ["   • No additional questions returned."]
    lines += ["", "Recommendation:"]
    recommendations = _lines(result.get("recommendations"))
    lines += [f"   → {x}" for x in recommendations[:4]] or ["   → Complete final CAB evidence review."]
    return "\n".join(lines)


def validate_result(result: dict[str, Any] | None) -> bool:
    return bool(result and _REQUIRED_KEYS.issubset(result))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test or benchmark the trained Pre-CAB GPT-OSS v1 adapter")
    parser.add_argument("--adapter", type=Path, default=None, help="Path to a saved LoRA adapter directory.")
    parser.add_argument("--base-model", type=str, default=None,
                         help="Stock base model name instead of --adapter, for the un-fine-tuned comparison run.")
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=768,
                         help="384 was too tight: the un-fine-tuned stock model in particular tends to "
                         "narrate before/around the JSON and needs more room to reach the closing brace, "
                         "so the whole holdout set was ending unparsed (truncated mid-JSON).")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                         help="Match training/prepare_qlora.py's budget so holdout prompts aren't right-"
                         "truncated more aggressively at eval time than they were during training.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.adapter and not args.base_model:
        raise SystemExit("Provide either --adapter <path> or --base-model <name>.")
    if args.adapter and not args.adapter.exists():
        raise SystemExit(f"Adapter not found: {args.adapter}")
    if not args.holdout.exists(): raise SystemExit(f"Holdout not found: {args.holdout}")

    model, tokenizer = load_model(str(args.adapter) if args.adapter else args.base_model, args.max_seq_length)
    rows = load_rows(args.holdout, args.limit)
    if not rows: raise SystemExit("Holdout contains no usable records")
    actual: Counter[str] = Counter(); predicted: Counter[str] = Counter(); correct = scored = false_pass = false_fail = unparsed = 0
    output_rows: list[dict[str, Any]] = []
    golds_for_report: list[str] = []
    preds_for_report: list[str | None] = []
    schema_valid_count = reasoning_populated_count = 0

    for idx, row in enumerate(rows, 1):
        messages = row.get("messages") or []
        if len(messages) < 2:
            unparsed += 1; print(f"[{idx}/{len(rows)}] INVALID_INPUT missing messages"); continue
        prompt = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        # With a balanced device map there is no single reliable model.device.
        # Put inputs on the first CUDA device; Accelerate/Unsloth dispatches the model.
        if torch_cuda_available():
            inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        print(f"[{idx}/{len(rows)}] generating (prompt tokens={input_len}, max_new_tokens={args.max_new_tokens})...", flush=True)
        try:
            import torch
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, use_cache=True)
            decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        except Exception as exc:
            unparsed += 1; print(f"[{idx}/{len(rows)}] GENERATION_ERROR {type(exc).__name__}: {exc}", flush=True); continue
        result = extract_json(decoded)
        if not validate_result(result): unparsed += 1
        pred = str(result.get("prediction")) if result and result.get("prediction") else None
        gold = row.get("metadata", {}).get("historical_outcome")
        cr_id = row.get("metadata", {}).get("cr_id", f"row-{idx}")
        quality = reasoning_quality(result)
        schema_valid_count += quality["schema_valid"]
        reasoning_populated_count += quality["reasoning_populated"]
        if pred: predicted[pred] += 1
        if gold:
            actual[str(gold)] += 1
            golds_for_report.append(str(gold))
            preds_for_report.append(pred)
            if pred:
                scored += 1; correct += pred == gold
                if pred == "PASS" and gold != "PASS": false_pass += 1
                elif gold == "PASS" and pred != "PASS": false_fail += 1
        output_rows.append({"cr_id": cr_id, "prediction": pred, "historical_outcome": gold, "parsed": validate_result(result), "reasoning_quality": quality, "agent_reasoning": result, "raw_output": decoded})
        print("\n" + render_agent_reasoning(result or {}) + "\n", flush=True)
        print(render_cab_result(str(cr_id), result or {}, str(gold) if gold else None), flush=True)
        print("\n" + "=" * 90, flush=True)

    report = {
        "records": len(rows), "scored": scored, "parsed": len(rows) - unparsed, "unparsed_or_failed": unparsed,
        "accuracy": correct / scored if scored else 0.0,
        # Accuracy alone hides collapse onto the majority class under label imbalance
        # (PASS is typically the majority historical outcome); macro-F1 and per-class
        # precision/recall are the metrics that actually catch that failure mode.
        "classification": classification_report(golds_for_report, preds_for_report),
        "false_passes": false_pass, "false_pass_rate": false_pass / scored if scored else 0.0,
        "false_fails": false_fail, "false_fail_rate": false_fail / scored if scored else 0.0,
        "schema_valid_rate": schema_valid_count / len(rows) if rows else 0.0,
        "reasoning_populated_rate": reasoning_populated_count / len(rows) if rows else 0.0,
        "actual_distribution": dict(actual), "predicted_distribution": dict(predicted),
        "model": str(args.adapter) if args.adapter else args.base_model,
        "mode": "fine_tuned_adapter" if args.adapter else "stock_base_model",
        "max_seq_length": args.max_seq_length, "max_new_tokens": args.max_new_tokens,
    }
    output_path = args.output or Path("/kaggle/working/pre_cab_v1_smoke_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions_path = output_path.with_suffix(".jsonl")
    with predictions_path.open("w", encoding="utf-8") as f:
        for item in output_rows: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("\n=== PRE-CAB GPT-OSS V1 EVALUATION ==="); print(json.dumps(report, indent=2)); print(f"Report: {output_path}"); print(f"Predictions: {predictions_path}")
    return 0


def torch_cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    raise SystemExit(main())
