# Pre-CAB Validator Demo

Model-independent proof of concept for validating **Normal** ServiceNow Change Requests before CAB.

## Scope

- Normal CRs are the primary supported change type.
- Emergency CRs are intentionally out of scope for V1.
- Standard CRs are secondary.
- GPT-OSS 120B is the only V1 reasoning model; Groq is the primary free inference route and Hugging Face Inference Providers is an alternate route for the same model.
- The model gateway is provider-agnostic so the underlying inference provider/model can change without changing agent behavior.
- Three configurable strictness profiles: Lenient, Balanced, Strict.
- Unified memory combines structured, semantic, episodic, policy, evidence, and CAB-history knowledge.
- Multi-agent reasoning covers CR fields, context, technical impact, business/CAB impact, testing, evidence, risk, similarity/clone detection, and final decisioning.
- Attachment analysis is a second-stage gate: claims in the CR are checked against supporting documents.
- Final decisions are `PASS`, `CONDITIONAL`, or `NOT_READY`; the LLM provides reasoning, while deterministic policy gates protect critical approval logic.

## Design principles

1. Technical detail is preserved. CAB reviewers get a clear summary plus expandable technical findings.
2. UAT is contextual, not universally mandatory. Requirements are inferred from the type and scope of the change.
3. A rollback plan can pass when it establishes a credible recovery path even if the procedure is brief; detail affects quality/confidence.
4. Similar historical CRs can become clone candidates, but clones always undergo delta validation.
5. The benchmark measures GPT-OSS 120B capability before any fine-tuning so improvements remain measurable.
6. Sensitive production data should not be committed to this public demo repository. Use synthetic data for demonstrations.
7. The reasoning brain explicitly separates observed facts, inferences, uncertainties, contradictions and recommendations, and performs a self-critique pass before a recommendation.
8. Historical frequency is advisory evidence, not automatic policy. Governance rules remain explicit and reviewable.
9. Free-tier operation favors local preprocessing/retrieval and one high-value GPT-OSS 120B synthesis pass rather than many independent model calls.

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

Never provide `ground_truth.private.json` to the model prompt or retrieval memory.

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
- Shared multi-agent context and a two-pass reasoning brain.
- Groq GPT-OSS 120B adapter with structured-output/reasoning support.
- Hugging Face GPT-OSS 120B adapter with provider-policy routing.
- Free-tier inference budget guardrails.
- Persistent local unified memory and replayable audit traces.
- PDF/XLSX/text attachment adapters and Stage-2 evidence verification.
- Claim/evidence verification and contradiction detection.
- CAB + technical reporting and stable pipeline/API contracts.
- Read-only ServiceNow ingestion boundary.
- Leakage-safe Normal-CR benchmark preparation.
- Field population/dependency profiling from a local ServiceNow export.
- Reasoning-rich fine-tuning candidate generation with deterministic splits.
- Synthetic Normal-CR and attachment fixtures.
- Lightweight manager-demo dashboard.
- CI/test scaffolding.

## Next milestone: real-data capability benchmark

Run entirely locally against the controlled historical dataset:

1. Profile the 141-field Normal-CR schema and population patterns.
2. Normalize historical CAB outcomes into benchmark labels.
3. Remove outcome-bearing and post-decision fields from model input.
4. Build a fixed evaluation set and keep it hidden from prompt/memory during prediction.
5. Run GPT-OSS 120B with the current rules, retrieval, memory and reasoning loop.
6. Measure accuracy, false-pass rate, false-fail rate, requirement-inference accuracy, evidence verification and clone quality.
7. Inspect failure cases and improve rules/retrieval/prompts before considering fine-tuning.

## Free-tier mode

The intended V1 operating pattern is:

```text
Local preprocessing
      ↓
Local retrieval + memory
      ↓
Deterministic specialist agents
      ↓
One bounded GPT-OSS 120B reasoning/synthesis pass
      ↓
Deterministic decision gates
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
    agents/          Specialized agent implementations
    brain/           Reasoning state, hypotheses and self-critique
    decision/        Deterministic decision and strictness engine
    documents/       Attachment extraction and evidence verification
    memory/          Unified-memory interfaces and storage adapters
    models/          LLM provider abstraction + Groq/HF adapters
    retrieval/       Hybrid structured/semantic/keyword retrieval
    similarity/      Historical matching and clone/delta analysis
    reporting/       CAB-readable + technical result generation
    api/             HTTP API contracts and handlers
    config/          Runtime configuration

data/
  demo/              Synthetic demo dataset only

config/              Explicit Normal-CR governance rules
scripts/             Local demo, benchmark, and profiling utilities

tests/               Unit and integration regression tests
```
