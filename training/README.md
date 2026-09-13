# Adaptive Pre-CAB training

This folder is the training side of the Pre-CAB brain. It is deliberately data-driven: it mines the historical export, builds compact model contexts, creates multi-task supervision, trains a GPT-OSS 20B QLoRA adapter on Kaggle, evaluates on a held-out split, and keeps a feedback buffer for future evolution.

## Important data rule

Raw Notes / Work Notes are included as a dedicated `notes` signal stream because they often contain valuable operational reasoning. They are **not automatically safe as decision-time labels**: comments can contain post-CAB outcomes or other leakage. The dataset builder therefore keeps notes in `notes_raw.jsonl` / `notes_signals.jsonl`, while the main decision-training records exclude known post-decision fields and mark note usage explicitly.

## Files

- `build_dataset.py` — normalizes the export, mines field usage/signals, includes Notes/Work Notes, creates train/validation/holdout JSONL, and writes a manifest.
- `train_qlora.py` — Kaggle-ready Unsloth + TRL QLoRA training for GPT-OSS 20B.
- `evaluate.py` — evaluates a saved adapter on the untouched holdout set and reports decision accuracy plus false-pass / false-fail rates when historical labels exist.
- `evolve.py` — appends reviewed real-world feedback and creates a prioritized correction set for the next adapter version.

## Kaggle setup

1. Upload your approved/de-identified CR export to Kaggle as a private dataset.
2. Clone/download this repository or upload the `training/` directory.
3. Run dataset preparation locally or in Kaggle:

```bash
python training/build_dataset.py /kaggle/input/pre-cab-data/change.json --output-dir /kaggle/working/pre_cab_dataset
```

4. Train:

```bash
python training/train_qlora.py --dataset-dir /kaggle/working/pre_cab_dataset --output-dir /kaggle/working/pre_cab_gptoss20b_v1
```

5. Evaluate before promoting the adapter:

```bash
python training/evaluate.py --adapter /kaggle/working/pre_cab_gptoss20b_v1 --holdout /kaggle/working/pre_cab_dataset/holdout_reasoning.jsonl
```

## Learning / evolution design

The training corpus intentionally has several tasks instead of one hard-coded classifier:

- field selection: learn which fields are informative for a CR context;
- requirement semantics: learn REQUIRED / CONDITIONAL / RECOMMENDED / OPTIONAL / NOT_OBSERVED patterns;
- notes analysis: learn useful operational signals from historical notes without leaking post-decision outcomes;
- specialist reasoning: technical, testing, risk, impact, contradiction, and CAB-question patterns;
- decision supervision: only where a historical CAB outcome can be normalized confidently.

New reviewed outcomes are appended to `feedback.jsonl`. `evolve.py` prioritizes high-value mistakes, especially false PASS cases, so the next adapter version focuses on the failure modes that matter most.

Never evaluate on records used to construct training examples, and never expose CAB outcome / closure fields to the prediction side of a historical example.
