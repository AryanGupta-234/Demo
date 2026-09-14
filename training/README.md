# Adaptive Pre-CAB training

This folder builds the data/context side of the Pre-CAB brain. The workflow is deliberately data-driven: mine the historical export, keep only decision-relevant CR fields, map applicable requirements, include Work Notes as chronological auxiliary evidence, build a compact label-safe historical context pack, and then let the active Ollama model (`qwen2.5:7b-instruct`) reason over the mapped context. The prediction step remains separate from the historical reference records.

## Data rule: Work Notes are evidence, not labels

Notes / Work Notes can contain valuable operational reasoning, including testing, rollback, approval, rework, incident, scheduling, or implementation updates. They can also contain post-decision information. Therefore:

- actual Work Notes are embedded in every generated reasoning example under `work_notes`;
- Work Notes are explicitly marked as chronological auxiliary evidence;
- they must not silently overwrite the current CR field state;
- they are never used as the training label;
- known post-decision CR fields are excluded from the model-visible field map;
- the historical context pack hides individual record outcomes and keeps only aggregate outcome distributions by context bucket.

## Files

- `build_dataset.py` — normalizes the export, mines field usage/signals, maps the optimized CR fields + requirements + actual Work Notes, and creates train/validation/holdout JSONL.
- `build_context_pack.py` — turns the mapped training set into a compact, self-contained historical reference pack. It prefers embedded Work Notes and does not attach an individual historical outcome to its CR card.
- `train_qlora.py` — Kaggle QLoRA training path for the experimental fine-tuning workflow.
- `evaluate.py` — evaluates a saved adapter on the untouched holdout set.
- `evolve.py` — appends reviewed real-world feedback for later model evolution.

## Build the mapped dataset

```bash
python training/build_dataset.py /kaggle/input/pre-cab-data/change.json --output-dir /kaggle/working/pre_cab_dataset
```

The generated reasoning examples contain:

```text
current CR
  ├─ optimized/high-signal fields
  ├─ applicable field requirements
  └─ actual chronological Work Notes

        ↓

specialist reasoning targets
  ├─ technical
  ├─ testing
  ├─ risk
  └─ CAB questions / recommendations
```

For the historical reference context used by the reasoning brain:

```bash
python training/build_context_pack.py \
  /kaggle/working/pre_cab_dataset/train_reasoning.jsonl \
  --notes-jsonl /kaggle/working/pre_cab_dataset/notes_raw.jsonl \
  --output /kaggle/working/pre_cab_dataset/pre_cab_context_pack.json
```

Then point the runtime at that pack:

```bash
set PRE_CAB_TRAINING_CONTEXT=/kaggle/working/pre_cab_dataset/pre_cab_context_pack.json
```

For Ollama, the provider supports up to the configured model context. Set `OLLAMA_NUM_CTX` explicitly to match the context you intend to load; the adapter defaults to 32768 rather than forcing a large local memory allocation.

## Prediction workflow

The intended reasoning flow is:

```text
1. Current CR + mapped requirements + Work Notes
                ↓
2. Historical reference pack (label-safe per-record cards)
                ↓
3. Qwen analysis: facts, evidence, gaps, uncertainties, contradictions
                ↓
4. Prediction: PASS / CONDITIONAL / NOT_READY
```

Historical examples are reference material, not facts about the current CR. Individual historical predictions are never copied to the current CR. Aggregate bucket outcomes are context only; the current prediction must be derived from the current CR evidence.

## Known limitation: field requirements are currently global

`field_requirement_context()` in `build_dataset.py` reuses `config/field_requirements.generated.json`, which was mined from the entire real historical export rather than separately re-mined per train/validation/holdout split. This is not a direct per-record label leak, but it can make a holdout benchmark mildly optimistic until the requirement table is re-mined train-split-only.

## Learning / evolution design

The corpus intentionally contains several reasoning tasks instead of one hard-coded classifier:

- field selection and context minimization;
- requirement semantics;
- Work Notes interpretation;
- technical, testing, risk and impact reasoning;
- contradiction detection;
- CAB-question generation;
- decision supervision where a historical CAB outcome can be normalized confidently.

Never evaluate on records used to construct training examples, and never expose CAB outcome / closure fields to the prediction side of an individual historical example.
