# Qwen2.5-7B-Instruct QLoRA -> Local Ollama

This path performs actual parameter-efficient fine-tuning. The Qwen base weights stay frozen and a LoRA adapter is trained on the mapped **training split only**.

## 1. Prepare on Windows

From the repo root after pulling `main`:

```powershell
python training/build_dataset.py <PATH_TO_REAL_SERVICENOW_JSON>
python training/prepare_qlora.py
```

`prepare_qlora.py` reads only `training/output/train_reasoning.jsonl` by default and writes:

```text
training/output/qwen_qlora_train.jsonl
```

Do **not** upload the original raw ServiceNow export to a public dataset or public GitHub repository.

## 2. Kaggle dataset

Create a private Kaggle Dataset and upload only the generated `qwen_qlora_train.jsonl` file.

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
  --output /kaggle/working/pre_cab_qwen \
  --model unsloth/Qwen2.5-7B-Instruct \
  --max-seq-length 4096 \
  --epochs 2 \
  --learning-rate 2e-4 \
  --batch-size 2 \
  --grad-accumulation 8 \
  --lora-r 16
```

The trainer writes:

```text
pre_cab_qwen/adapter/
pre_cab_qwen/merged_16bit/
pre_cab_qwen/gguf/
pre_cab_qwen/training_manifest.json
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

For any capability number, evaluate on the untouched holdout set. Never fine-tune on holdout examples. Compare at minimum:

1. stock `qwen2.5:7b-instruct`
2. stock Qwen + learned JSON context
3. fine-tuned `pre-cab-qwen`

Use the same holdout cases and the same output schema for all three.
