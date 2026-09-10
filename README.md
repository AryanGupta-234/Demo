# Pre-CAB Validator Demo

Model-independent proof of concept for validating **Normal** ServiceNow Change Requests before CAB.

## Scope

- Normal CRs are the primary supported change type.
- Emergency CRs are intentionally out of scope for V1.
- Standard CRs are secondary.
- GPT-OSS 120B via Groq is the initial reasoning model.
- The model gateway remains provider-agnostic so the underlying model can change without changing agent behavior.
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
5. The benchmark phase measures GPT-OSS 120B capability before any fine-tuning so improvements remain measurable.
6. Sensitive production data should not be committed to this public demo repository. Use synthetic data for demonstrations.
7. The reasoning brain explicitly separates observed facts, inferences, uncertainties, contradictions and recommendations, and performs a self-critique pass before a recommendation.
8. Free-tier mode spends model calls only on high-value reasoning; deterministic agents, retrieval, parsing and policy checks run locally first.

## Free-tier development mode

The current V1 is designed around Groq's current free-plan limits for `openai/gpt-oss-120b`: 30 requests/minute, 1,000 requests/day, 8K tokens/minute, and 200K tokens/day. The code therefore budgets one bounded synthesis call per CR and requires retrieval/compaction before sending oversized context. See `src/pre_cab/budget.py`.

The model adapter uses GPT-OSS 120B reasoning mode and structured JSON-schema output. This gives the agents a stable machine-readable contract while keeping the model replaceable later.

## Current API

Install the optional API dependencies with `pip install -e '.[api]'`. The core HTTP surface is:

- `GET /health`
- `POST /v1/pre-cab/validate`

Example request shape:

```json
{
  "strictness": "balanced",
  "cr": {
    "Number": "CHG-DEMO-001",
    "Type": "Normal",
    "Short description": "Monthly OS security patching",
    "Description": "Patch production management servers and reboot them.",
    "Justification": "Remediate operating-system vulnerabilities.",
    "Implementation plan": "Apply approved patches, reboot, validate service health.",
    "Backout plan": "Restore the previous server image from backup.",
    "Test plan": "Validate service availability and patch level after reboot.",
    "Risk": "Low",
    "Configuration item": "demo-management-prod",
    "Conflict status": "No Conflict"
  }
}
```

The response contains both `cab_view` and `technical_view` so the technical layer is retained without making the CAB reviewer parse implementation detail first.

## Planned layout

```text
src/
  api/               FastAPI application and contracts
  agents/            Specialized agent implementations
  brain/             Reasoning state, hypotheses and self-critique
  decision/          Deterministic decision and strictness engine
  documents/         Attachment extraction and evidence verification
  memory/            Unified-memory interfaces and storage adapters
  models/            LLM provider abstraction + Groq adapter
  retrieval/         Hybrid SQL/vector/keyword retrieval
  schemas/           Shared typed contracts
  similarity/        Historical matching and clone/delta analysis
  validation/        CR field and contextual validation rules
  reporting/         CAB-readable + technical result generation
  config/             Runtime configuration

data/
  demo/              Synthetic demo dataset only

tests/
  unit/
  integration/
  evaluation/

scripts/
  benchmark/         Historical benchmark utilities
```

## Development phases

### Phase 1 — Foundation

Typed contracts, configuration, strictness profiles, model gateway, unified-memory interfaces, deterministic decision engine, and safe synthetic fixtures.

### Phase 2 — Retrieval + reasoning

Hybrid retrieval, similar-change/clone analysis, shared agent context, structured reasoning brain, self-critique and stable reporting contracts.

### Phase 3 — Evidence gate

Attachment discovery, PDF/XLSX/email extraction, claim-to-evidence verification, contradiction detection, and evidence quality scoring.

### Phase 4 — Service/API integration

Stable HTTP API, ServiceNow adapter boundary, audit events, authentication and request-level observability.

### Phase 5 — Benchmark

Replay historical Normal CRs without exposing their CAB outcomes to the model, then compare predictions with historical outcomes and measure false-pass rate.

### Phase 6 — Demo

CAB-first dashboard with technical drill-down, predicted CAB questions, clone suggestions, and audit trail.
