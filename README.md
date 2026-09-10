# Pre-CAB Validator Demo

Model-independent proof of concept for validating **Normal** ServiceNow Change Requests before CAB.

## Scope

- Normal CRs are the primary supported change type.
- Emergency CRs are intentionally out of scope for V1.
- Standard CRs are secondary.
- GPT-OSS 120B is the only V1 reasoning model; Groq is the primary inference route and Hugging Face Inference Providers is an alternate route for the same model.
- The model gateway is provider-agnostic so the underlying inference provider/model can change without changing agent behavior.
- Three configurable strictness profiles: Lenient, Balanced, Strict.
- Unified memory combines structured, semantic, episodic, policy, evidence, and CAB-history knowledge.
- Multi-agent reasoning covers CR fields, context, technical impact, business/CAB impact, testing, evidence, risk, similarity/clone detection, and final decisioning.
- Attachment analysis is a second-stage gate: claims in the CR are checked against supporting documents.
- Final decisions are `PASS`, `CONDITIONAL`, or `NOT_READY`; the LLM provides reasoning, while deterministic policy gates protect critical approval logic.

## Ingestion boundary

The validator does **not** own the ServiceNow bulk-fetch or attachment-download workflow. The first integration layer is expected to fetch the CRs from ServiceNow and create the local CR workspace. Your existing API/file-management script can be used as that upstream input layer.

The contract is intentionally simple:

```text
ServiceNow API / fetcher
        ↓
fetch CR metadata + all attachments
        ↓
file-management layer
        ↓
private CR workspace
        ├── CR001234/
        │   ├── attachment-a.pdf
        │   ├── test-results.xlsx
        │   └── approval.pdf
        ├── CR001235/
        │   ├── uat.pdf
        │   └── rollback.pdf
        └── ...
        ↓
Pre-CAB Validator
        ↓
validation outputs
```

The validator treats each `<CR number>/` directory as the authoritative attachment boundary for that CR. It reads the files inside that directory and converts supported attachments into normalized evidence documents. It does not call ServiceNow, move/delete downloaded files, or inspect sibling CR folders. The upstream fetcher and file manager can therefore evolve independently of the validation algorithm.

For the current phase, manually placing PDFs/files into the same CR-specific directory structure is also supported. This is useful for testing the validator before the upstream ServiceNow fetcher is attached.

Sensitive production CR exports and downloaded attachments must remain in the controlled local environment and must **not** be committed to this public repository.

## Design principles

1. Technical detail is preserved. CAB reviewers get a clear summary plus expandable technical findings.
2. UAT is contextual, not universally mandatory. Requirements are inferred from the type and scope of the change.
3. A rollback plan can pass when it establishes a credible recovery path even if the procedure is brief; detail affects quality/confidence.
4. Similar historical CRs can become clone candidates, but clones always undergo delta validation.
5. The benchmark measures GPT-OSS 120B capability before any fine-tuning so improvements remain measurable.
6. Sensitive production data should not be committed to this public demo repository. Use synthetic data for demonstrations.
7. The reasoning brain explicitly separates observed facts, inferences, uncertainties, contradictions and recommendations and performs self-critique when the selected reasoning mode requests it.
8. Historical frequency is advisory evidence, not automatic policy. Governance rules remain explicit and reviewable.
9. Free-tier operation favors local preprocessing/retrieval and one high-value GPT-OSS 120B synthesis pass rather than many independent model calls.
10. Unreadable/scanned/image-only evidence is never silently treated as verified; it is surfaced for OCR/vision review.

## Run the manager demo

```text
pip install -e '.[all]'
uvicorn pre_cab.api:create_app --factory --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/`.

The demo UI keeps the CAB result prominent while retaining technical findings and evidence detail underneath.

## Local historical-data benchmark

Keep the real ServiceNow export outside Git. The benchmark preparation scripts read the local file, filter to Normal changes, remove outcome-bearing fields, and write sanitized model input plus a private ground-truth file.

```text
python scripts/prepare_benchmark.py real_data/cr_export.json
python scripts/profile_fields.py real_data/cr_export.json
```

For a fixed, reproducible Stage-1 benchmark sample:

```text
python scripts/run_benchmark.py real_data/cr_export.json --limit 0 --seed 7 --strictness balanced
```

For the GPT-OSS 120B Stage-1 baseline:

```text
python scripts/run_benchmark.py real_data/cr_export.json --limit 50 --seed 7 --strictness balanced --llm --provider auto
```

For the end-to-end benchmark (evidence-aware pipeline plus optional GPT-OSS 120B final reasoning):

```text
python scripts/run_real_benchmark.py real_data/cr_export.json --strictness balanced
python scripts/run_real_benchmark.py real_data/cr_export.json --strictness balanced --llm --provider auto
```

If the upstream ServiceNow fetcher/file manager has produced one private directory per CR, point the validator at the **parent directory**:

```text
python scripts/run_real_benchmark.py real_data/cr_export.json --strictness balanced --attachment-root real_data/cr_workspace
python scripts/run_real_benchmark.py real_data/cr_export.json --strictness balanced --attachment-root real_data/cr_workspace --llm --provider auto
```

Expected layout:

```text
real_data/cr_workspace/
├── CR001234/
│   ├── approval.pdf
│   └── test-results.pdf
├── CR001235/
│   └── uat.xlsx
└── ...
```

Use `--stage1-only` when you specifically want to isolate Stage 1. The end-to-end harness records the pipeline mode in the manifest so Stage-1-only and full-pipeline results are not confused.

The command writes a benchmark report, prediction JSONL, and dataset/configuration manifest. The manifest contains hashes/counts and runtime configuration, not raw CR contents. Never provide `ground_truth.private.json` to the model prompt or retrieval memory.

Layer ablation is available through:

```text
python scripts/run_ablation.py real_data/cr_export.json --limit 50 --seed 7 --strictness balanced
python scripts/run_ablation.py real_data/cr_export.json --limit 50 --seed 7 --strictness balanced --llm --provider auto
```

If private attachment evidence is available locally, add `--attachment-root <private-cr-workspace>`; attachment contents remain local.

## Current API

- `GET /health`
- `POST /v1/pre-cab/validate`

The validation endpoint accepts the CR plus extracted attachment records. The response contains both `cab_view` and `technical_view`.

## Current development state

### Implemented

- Typed CR, finding, requirement and agent-context contracts.
- Deterministic field validation and three strictness profiles.
- Contextual UAT/customer-approval/outage applicability signals.
- Credible rollback/recovery detection.
- Hybrid retrieval baseline and clone/delta analysis.
- Shared multi-agent context and configurable single/dual reasoning modes.
- Groq GPT-OSS 120B adapter with structured-output/reasoning support.
- Hugging Face GPT-OSS 120B adapter with provider-policy routing.
- Free-tier inference budget guardrails and model-response caching.
- Persistent local unified memory and replayable audit traces.
- PDF/XLSX/text attachment adapters and Stage-2 evidence verification.
- Claim/evidence verification and contradiction detection.
- Explicit unreadable/image/scanned evidence findings.
- Ranked evidence excerpts passed to final GPT synthesis.
- CAB + technical reporting and stable pipeline/API contracts.
- Read-only ServiceNow ingestion boundary with bounded retries, timeouts, and attachment-size limits.
- Leakage-safe Normal-CR benchmark preparation.
- Field population/dependency profiling from a local ServiceNow export.
- Reasoning-rich fine-tuning candidate generation with deterministic splits.
- Synthetic Normal-CR and attachment fixtures.
- Lightweight manager-demo dashboard.
- CI/test scaffolding.
- Benchmark dataset fingerprints, manifests and layer-ablation tooling.
- Reviewed requirement-level evaluation framework with per-label precision/recall/F1 and reviewer feedback/episodic-memory path.
- Explicit private CR-workspace handoff: the validator consumes one CR directory's downloaded attachments without owning upstream ServiceNow file management.

## Capability benchmark sequence

Run entirely locally against the controlled historical dataset:

1. Profile the Normal-CR schema and population patterns.
2. Normalize historical CAB outcomes into benchmark labels.
3. Remove outcome-bearing and post-decision fields from model input.
4. Build a fixed evaluation set and keep it hidden from prompt/memory during prediction.
5. Run deterministic Stage 1 first.
6. Compare memory/clone, evidence, and GPT-OSS 120B layers with the same seed/sample.
7. Measure accuracy, false-pass rate, false-fail rate, requirement-inference accuracy, evidence verification and clone quality.
8. Inspect failure cases and improve rules/retrieval/prompts before considering fine-tuning.
9. Fit/validate confidence calibration only after enough reviewed outcomes exist.

The real-data execution itself should happen in the user's controlled environment; the public repository contains only the benchmark tooling and synthetic fixtures.

## Free-tier mode

The intended V1 operating pattern is:

```text
ServiceNow fetcher / file manager
      ↓
Private CR workspace
      ↓
Local preprocessing
      ↓
Local retrieval + memory
      ↓
Deterministic specialist agents
      ↓
Stage-2 evidence verification
      ↓
One bounded GPT-OSS 120B reasoning/synthesis pass
      ↓
Conservative reconciliation (model may downgrade, never upgrade)
```

Provider routing:

```text
Groq GPT-OSS 120B
        ↓
Hugging Face GPT-OSS 120B (alternate)
```

API keys are environment variables only (`GROQ_API_KEY`, `HF_TOKEN`); they are never stored in source control.

## Fine-tuning path

Fine-tuning is deliberately deferred until the baseline benchmark is understood. The repository already contains a reviewable reasoning-example builder, but no model weights are changed by it.

Future paid path:

```text
Benchmark baseline
    ↓
Improve rules / retrieval / prompts
    ↓
Build reviewed reasoning examples
    ↓
Train / fine-tune GPT-OSS 120B
    ↓
Evaluate on untouched test set
    ↓
Compare against baseline
```

## Repository layout

```text
src/
  pre_cab/
    Specialized deterministic agents, reasoning, decisioning, documents,
    memory, model providers, retrieval, similarity, reporting and API

data/
  demo/              Synthetic demo dataset only

config/              Explicit Normal-CR governance rules
scripts/              Local demo, benchmark, profiling and evaluation utilities
tests/                Unit and integration regression tests
```