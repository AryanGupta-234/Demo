# Qwen2.5-7B-Instruct QLoRA -> Local Ollama

This path performs actual parameter-efficient fine-tuning. The Qwen base weights stay frozen and a LoRA adapter is trained on the mapped **training split only**.

## 1. Prepare on Windows

From the repo root after pulling `main`:

```powershell
python training/build_dataset.py <PATH_TO_REAL_SERVICENOW_JSON>
python training/prepare_qlora.py
```

`prepare_qlora.py` reads `training/output/{train,validation,holdout}_reasoning.jsonl` and writes:

```text
training/output/qwen_qlora_train.jsonl      # used for gradient updates
training/output/qwen_qlora_valid.jsonl      # eval_loss / checkpoint selection only
training/output/qwen_qlora_holdout.jsonl    # final unbiased evaluation only
training/output/qwen_qlora_prep_report.json # label distribution, note-truncation stats, cross-split leakage check
```

Every split gets the same field-preserving Work Notes/Comments shrink pass and a cross-split
CR-id disjointness check (it aborts if any CR id leaked across splits). Only
`qwen_qlora_train.jsonl` is ever used for gradient updates. If historical outcomes are heavily
skewed toward PASS, you may pass `--oversample-minority` to duplicate CONDITIONAL/NOT_READY
**training** rows toward a configurable ratio of the majority class -- off by default; check
`qwen_qlora_prep_report.json`'s `label_distribution` first and prefer relying on the macro-F1
metric in `training/evaluate.py` before reaching for oversampling.

Do **not** upload the original raw ServiceNow export to a public dataset or public GitHub repository.

## 2. Kaggle dataset

Create a private Kaggle Dataset and upload `qwen_qlora_train.jsonl` and `qwen_qlora_valid.jsonl`.
Keep `qwen_qlora_holdout.jsonl` for the final evaluation step (step 8) -- it can be uploaded to
the same private dataset since this training script never reads it.

In the Kaggle notebook, enable a GPU and make the dataset available under something like:

```text
/kaggle/input/pre-cab/qwen_qlora_train.jsonl
```

## 3. Install training stack

Run in the first Kaggle cell:

```python
!pip install -U "unsloth" "trl" "datasets" "bitsandbytes" "accelerate" "transformers" "peft"
```

Restart the kernel if Kaggle asks for it.

## 4. Run QLoRA

Copy `training/train_qwen_qlora.py` into the notebook or clone the repository if your private Kaggle workflow permits it. Then run:

```python
!python train_qwen_qlora.py \
  --train /kaggle/input/pre-cab/qwen_qlora_train.jsonl \
  --valid /kaggle/input/pre-cab/qwen_qlora_valid.jsonl \
  --output /kaggle/working/pre_cab_qwen \
  --model unsloth/Qwen2.5-7B-Instruct \
  --max-seq-length 4096 \
  --epochs 4 \
  --learning-rate 2e-4 \
  --batch-size 1 \
  --grad-accumulation 16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --early-stopping-patience 2
```

`--epochs 4` is an upper bound: training stops early once `eval_loss` on the validation split stops
improving for `--early-stopping-patience` evaluation rounds (evaluated once per epoch), and the
checkpoint with the best `eval_loss` is reloaded at the end (`load_best_model_at_end`). Before
training, the script tokenizes every example with the real Qwen tokenizer and, for anything longer
than `--max-seq-length`, shrinks Work Notes/Comments (never the fixed CR fields, never the
assistant target) via the same field-preserving logic as `prepare_qlora.py`; anything that still
doesn't fit is dropped and logged rather than silently truncated by the trainer. It also masks the
training loss to the assistant completion only, so gradient updates optimize the actual
reasoning/prediction target rather than reproducing the (large) input CR JSON.

The trainer writes:

```text
pre_cab_qwen/adapter/
pre_cab_qwen/merged_16bit/
pre_cab_qwen/gguf/
pre_cab_qwen/training_manifest.json   # config, length-audit stats, label distribution, leakage guarantees
```

If direct GGUF export is unsupported by the installed Unsloth build, use the `merged_16bit` directory and convert that model with a current llama.cpp build instead.

## 5. What is actually learned

The QLoRA objective is the assistant output in each training example. The mapped CR, Work Notes, Comments, requirements and signoff dispositions are the input; the normalized CAB reasoning/prediction JSON is the supervised target.

The training split is the only split used for gradient updates. Validation and holdout remain for evaluation.

## 6. Bring the GGUF to Windows

Download the produced `.gguf` file from Kaggle to the local machine. Keep it outside Git if the dataset is sensitive.

Create `Modelfile` beside the GGUF:

```text
FROM ./pre_cab_qwen.gguf
PARAMETER temperature 0.05
PARAMETER num_ctx 32768
SYSTEM You are the local Pre-CAB reasoning model. Return the required JSON output and never invent evidence.
```

Then create the Ollama model:

```powershell
ollama create pre-cab-qwen -f Modelfile
ollama list
```

Smoke test:

```powershell
ollama run pre-cab-qwen "Return exactly JSON with a field named prediction whose value is PASS."
```

## 7. Run the project against the fine-tuned model

Point the existing Ollama provider at the new model:

```powershell
$env:OLLAMA_MODEL = "pre-cab-qwen"
python scripts/predict_qwen.py <PATH_TO_NEW_CR_JSON> --provider ollama --mode dual
```

The prediction stage will then use the fine-tuned Qwen model locally.

## 8. Evaluation rule

For any capability number, evaluate on the untouched holdout set (`qwen_qlora_holdout.jsonl`).
Never fine-tune on holdout examples. Compare at minimum:

1. stock `unsloth/Qwen2.5-7B-Instruct`
2. stock Qwen + learned JSON context (`training/learn_qwen.py` output)
3. fine-tuned `pre-cab-qwen`

Use the same holdout cases and the same output schema for all three. Run `training/evaluate.py`
against the holdout set for both the stock model and the fine-tuned adapter:

```python
!python evaluate.py --base-model unsloth/Qwen2.5-7B-Instruct \
  --holdout /kaggle/input/pre-cab/qwen_qlora_holdout.jsonl --limit 0 \
  --output /kaggle/working/stock_holdout_report.json

!python evaluate.py --adapter /kaggle/working/pre_cab_qwen/adapter \
  --holdout /kaggle/input/pre-cab/qwen_qlora_holdout.jsonl --limit 0 \
  --output /kaggle/working/finetuned_holdout_report.json
```

Read `macro_f1` and `per_class` precision/recall before `accuracy` -- historical outcomes skew
toward PASS, so a model that always predicts PASS can still show high accuracy while having zero
recall on CONDITIONAL/NOT_READY. Also check `schema_valid_rate` and `reasoning_populated_rate`:
a fine-tuned model that raises `macro_f1` but collapses `reasoning_populated_rate` has learned to
guess the label without doing the analysis, which defeats the point of this project.
