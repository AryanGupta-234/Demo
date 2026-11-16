# Pre-CAB Validator

A local-first, evidence-aware validation engine that checks Normal ServiceNow Change Requests (CRs) before they reach the Change Advisory Board (CAB).

## What it does

- Validates each CR deterministically (Level 1): required fields, implementation, rollback, risk, conflicts, testing and data-driven contextual requirements, with findings graded INFO, WARNING or BLOCKING.
- Inventories every evidence file in the CR's own workspace (`evidence/<CR number>/`), extracts PDF, DOCX, PPTX, XLSX, TXT, CSV, EML and JSON, and runs local OCR (Tesseract, PyMuPDF) on images and scanned pages.
- Verifies evidence (Level 2): checks the document names the current CR, supports its claimed purpose (UAT, approval, QA sign-off, rollback) and does not contradict other signals.
- Adds optional neural reasoning (Level 3) through Ollama, Groq or Hugging Face, with semantic memory and self-critique. The model can only make a result more conservative; it cannot turn a deterministic or evidence blocker into PASS.
- Returns PASS, CONDITIONAL or NOT_READY with CAB and technical reports, and records an audit trail in local SQLite.
- Includes a read-only (GET-only) ServiceNow REST client for validating a real CR by number.

## Built with

- Python 3.11+ (package `pre-cab-validator`, `src/pre_cab/`)
- pypdf, openpyxl, python-docx, python-pptx, PyMuPDF, pytesseract and Pillow for document extraction and OCR
- sentence-transformers for semantic memory
- Ollama, Groq or Hugging Face as model providers (optional)
- FastAPI and Uvicorn for the local API
- pytest, ruff and mypy; GitHub Actions CI on Python 3.11 to 3.13

## Run it

Requirements: Python 3.11 or newer and Git. Ollama and Tesseract are optional system applications, installed separately.

1. Clone and run the installer from the repository root:

       git clone https://github.com/AryanGupta-234/Demo.git
       cd Demo
       python install.py

   Default mode creates or reuses `.venv`, installs the lightweight dependencies and runs the offline manager demo (`scripts/demo.py`) on the synthetic CRs in `data/demo/`. No model or network is needed for it.

2. For the full AI, API and development stack plus the test suite:

       python install.py --full

   (Use `py install.py --full` if only `py` is available, or `python3` on Linux/WSL.)

3. Activate the environment: `.\.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Linux/WSL.

4. Validate your own CR export:

       python main.py ./data/one_cr.json --provider ollama

Manual installation instead of the installer:

    python -m venv .venv
    # activate it, then:
    python -m pip install --upgrade pip setuptools wheel
    pip install -e ".[all,dev]"

Selective extras: `pip install -e .` (core only), `".[docs]"`, `".[embeddings]"`, `".[api]"`, `".[llm]"`.

Check the install:

    python -m compileall src scripts main.py
    python -m pytest

## Project status

Working prototype. The full pipeline runs locally on demo and exported data and has an automated test suite. Production use against a real ServiceNow instance still needs organisation-specific work (see "Demo vs production" below).

---

## Reference

### Pipeline

    CR input
      -> normalization
      -> Level 1 deterministic validation
      -> specialist agents / blackboard
      -> CR-scoped evidence workspace
      -> file inventory + document extraction (+ OCR)
      -> Level 2 evidence verification
      -> semantic memory retrieval
      -> Level 3 neural reasoning
      -> adversarial critique when applicable
      -> final conservative decision gate
      -> CAB + technical NLG
      -> audit + episodic memory

`main.py` processes CRs sequentially, which keeps the default path predictable and budget friendly without disabling any layer. Emergency changes are excluded unless you pass `--include-emergency`. `src/pre_cab/parallel.py` offers a bounded worker pool that runs one complete `run_pre_cab()` per CR, preserving input order and per-CR evidence isolation.

### `main.py` options

    python main.py <cr_json> [--strictness ...] [--provider auto|groq|huggingface|ollama]
                   [--attachment-root PATH] [--evidence-file FILE ...]
                   [--output-dir artifacts/pre_cab] [--limit N] [--include-emergency]

Batch mode:

    python scripts/run_batch.py ./data/cr_export.json --attachments ./data/evidence --output artifacts/cr_results.jsonl

Offline diagnostic (dependencies, normalisation, Level-1 decision, workspace inventory, extraction failures; add `--json` for machine-readable output):

    python scripts/diagnose.py ./data/cr_export.json --evidence-root ./data/evidence

Local API (FastAPI on http://127.0.0.1:8000; set `PRE_CAB_ENABLE_LLM=1` to attach a model):

    python scripts/serve_demo.py

### Architecture by module

| Module | Responsibility |
|---|---|
| input_loader.py | Load and normalize CR exports |
| decision.py | Deterministic CR validation |
| field_requirement_engine.py | Data-driven contextual requirements |
| agents.py | Specialist CR agents |
| orchestrator.py | Agent orchestration and blackboard |
| evidence_workspace.py | CR workspace resolution and classification |
| local_attachments.py | File inventory and extraction |
| ocr.py | Local OCR and optional Ollama vision |
| rule_lifecycle.py | Holdout validation and rule lifecycle promotion |
| evidence.py | Evidence verification |
| evidence_agents.py | Document-purpose specialists |
| evidence_retrieval.py | Relevant evidence excerpt retrieval |
| brain.py | Reasoning contracts and prompts |
| brain_loop.py | Reasoning, critique and NLG |
| final_reasoning.py | Neural reconciliation |
| semantic_memory.py | Persistent semantic memory |
| memory_consolidation.py | Feedback and consolidation |
| pipeline.py | Public Level 1 -> Level 2 -> Level 3 pipeline |
| audit_store.py | Persistent audit trail |
| reporting.py | CAB and technical output |
| runtime_provider.py | Model provider bootstrap |
| servicenow.py / servicenow_readonly.py | ServiceNow adapter contract and GET-only REST client |

### Level 1: CR validation

Deterministic and needs no model. It evaluates CR type, short description, description, justification, implementation plan, backout plan, test plan, risk, configuration item, environment, conflict status, sign-off dispositions, contextual requirements and work-note signals.

    BLOCKING -> NOT_READY
    WARNING  -> CONDITIONAL
    none     -> PASS

Field requirements resolve as exact Category + Sub Category, then Category, then a global fallback, with levels REQUIRED, CONDITIONAL, RECOMMENDED, OPTIONAL and NOT_OBSERVED. Specialist agents (Field, Context, Technical, Business Impact, Testing, Risk, Evidence, Clone/Similarity, Cross-Agent Consistency) publish findings to a shared blackboard. The blackboard is advisory; deterministic policy remains authoritative.

### Evidence workspace and formats

Place downloaded files per CR:

    evidence/
      CHG001234/
        approval.pdf
        uat.xlsx
        implementation.docx
        rollback.pptx
        screenshot.png

The validator reads only the requested CR's workspace and never uses a sibling CR's evidence as proof. The older root-level CR-prefixed layout is also supported.

Every file is inventoried with a status (`supported`, `unsupported`, `too_large`, `stat_error`), so "not parsed" is never confused with "does not exist". Supported formats: PDF, DOCX, PPTX, TXT, Markdown, CSV, EML, JSON, XLSX, XLSM, and images (PNG, JPG/JPEG, WEBP, BMP, TIFF).

The default per-file extraction limit is 75 MiB. Change it with `PRE_CAB_MAX_ATTACHMENT_BYTES` (for example `157286400`). Oversized files stay in the inventory with an extraction error.

### OCR and vision

Raster images are OCR-processed with Tesseract; textless or mixed PDF pages are rendered with PyMuPDF and OCR-processed. OCR metadata (engine, confidence, page count, errors) goes into the evidence manifest. Tesseract is a separate system install; on Windows set `TESSERACT_CMD` if it is not on the default path. An optional Ollama vision model (`PRE_CAB_VISION_MODEL`) can describe images; its output is marked as derived analysis, not direct evidence. Unreadable evidence is surfaced as `EVIDENCE_UNREADABLE` or `EVIDENCE_VISION_REVIEW_REQUIRED`, never treated as proof.

### Level 2: evidence verification

Evidence is not accepted because a keyword exists. The verifier asks whether a relevant document exists, whether it identifies the current CR, whether it supports the claimed purpose, whether its status is positive, negative or unclear, whether it contradicts another signal, and whether it was actually readable. Categories: customer approval, UAT, test execution and results, QA sign-off, lower-environment validation, rollback and recovery, implementation or runbook evidence.

Key rules:

- A generic "UAT passed" document is not CR-specific proof; strong evidence names the CR (for example `CHG001234_UAT_Report.pdf`).
- Test plan, test execution and formal test results are separate. The parser prefers the latest explicit status ("Previous test failed ... Current test passed").
- Negative approval states (not approved, rejected, declined, pending, awaiting) are checked before positive terms.
- DEV, SIT, staging, UAT and Pre-PROD evidence cannot by itself prove production readiness; a PROD CR with only DEV evidence is flagged as a contradiction.
- Rollback is evaluated separately; "rollback if needed" is not a concrete recovery mechanism, and historical documents do not prove the current CR's rollback.

### Level 3: neural reasoning and the final gate

The model receives the CR, Level-1 findings, specialist observations, Level-2 state, ranked evidence excerpts, semantic memory, contradictions and uncertainties, and reasons in the order facts, inferences, uncertainties, contradictions, recommendations, decision. It is constrained not to invent current evidence and runs a self-critique (unverified claims, plan vs execution, memory mistaken for proof, credible rollback). NLG then writes CAB and technical narratives but cannot change the decision, confidence, facts, evidence status or uncertainty.

The final gate takes the most conservative result:

    Stage 1 = NOT_READY, Stage 2 = PASS      -> Final = NOT_READY
    Stage 1 = PASS,      Stage 2 = NOT_READY -> Final = NOT_READY

### Memory, feedback and rule lifecycle

Memory records are stored in SQLite with local embeddings; retrieval combines semantic similarity, lexical overlap, identifiers and authority/recency, then MMR. Repeated episodes consolidate into patterns marked `authoritative = false` and `current_cr_evidence = false`, so historical learning cannot become current approval evidence. Reviewer feedback (predicted vs actual decision, notes, corrected requirements) adjusts retrieval metadata, not the original evidence.

Mined rules follow `CANDIDATE -> VALIDATED -> ACTIVE -> DEPRECATED`, and are promoted only after validation against a separate Normal-CR holdout set:

    python scripts/mine_field_requirements.py historical_train.json --holdout historical_holdout.json --out config/field_requirements.generated.json
    python scripts/promote_field_requirements.py config/field_requirements.generated.json historical_holdout.json --out config/field_requirements.generated.json

### Model providers

Ollama (default `http://localhost:11434`):

    ollama pull qwen2.5:7b-instruct
    python main.py ./data/one_cr.json --provider ollama

`OLLAMA_MODEL`, `OLLAMA_BASE_URL` and `OLLAMA_NUM_CTX` configure it, and any compatible Ollama model can be used without changing the validator. Cloud providers: `GROQ_API_KEY` with `--provider groq`, `HF_TOKEN` with `--provider huggingface`, or `--provider auto`. See `docs/FREE_MODE.md` for the no-budget Groq and Hugging Face setup. The Qwen/QLoRA material in `training/` is an example training workflow, not a requirement.

Other settings: `PRE_CAB_SEMANTIC_MEMORY` (default true), `PRE_CAB_EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`, downloaded on first use), `PRE_CAB_SEMANTIC_WEIGHT` (0.65), `PRE_CAB_NLG_MODE=refine`, `PRE_CAB_MODEL_CACHE=1`, `PRE_CAB_MODEL_CACHE_PATH`, `PRE_CAB_BUDGET_GUARD=1`.

### Outputs

    artifacts/pre_cab/
      CHG001234_pre_cab.json
      CHG001234_pre_cab.txt
      batch_results.json

The JSON includes the final decision, confidence, findings, evidence manifest, workspace inventory, extraction errors, agent state, reasoning, model information and audit ID. Each run's audit record holds the CR number, strictness, stage and final decisions, model, evidence manifest, reasoning and findings.

### Demo vs production

| Area | Current demonstration | Real / production work |
|---|---|---|
| CR input | Local JSON fixture | ServiceNow REST CR retrieval |
| Authentication | None required | ServiceNow credentials / approved auth mechanism |
| Attachments | Local `evidence/<CR>/` folder | ServiceNow attachment API |
| Validation, agents, evidence checks | Real | Same |
| Neural reasoning | Configurable local or cloud model | Same, subject to deployment policy |
| Memory and audit | Local SQLite | Replaceable production persistence |
| ServiceNow writes | Not performed | Must be implemented and authorised separately |

Validate a real CR (read-only; it never approves the CR or writes back):

    $env:SERVICENOW_BASE_URL="https://your-instance.service-now.com"
    $env:SERVICENOW_USERNAME="..."
    $env:SERVICENOW_PASSWORD="..."
    python scripts/validate_servicenow_change.py CHG001234 --provider ollama

The client retries transient transport, 429 and 5xx failures, enforces attachment size limits and reads credentials from environment variables only. Still needed for production: an approved auth method (Basic is a reference implementation), instance and query configuration, attachment retention policy, secret management, logging and observability, rate limits, production audit storage, network controls, any write-back workflow, and scheduling.

### Failure containment

- No embedding model: falls back to lexical retrieval.
- No LLM: deterministic validation continues.
- Broken document: stays inventoried with its extraction error.
- Malformed model JSON: bounded repair, otherwise the deterministic result is kept.
- Evidence blocker plus model PASS: the blocker wins (NOT_READY).

### Security

Keep production data outside Git. Never commit production CR exports, customer approvals, screenshots, ServiceNow attachments, credentials, API keys, `.env`, `.pre_cab/` databases or production artifacts. Suggested ignore entries: `.env`, `.pre_cab/`, `artifacts/`, `real_data/`, `evidence/`, `attachments/`, `*.sqlite3`. Historical memory is context, not current evidence.

### Testing and evaluation

The test suite in `tests/` covers field validation, requirement resolution, evidence verification, CR identity matching, unreadable evidence, environment contradictions, clone analysis, pipeline merging, neural downgrade protection, caching, audit and memory persistence, feedback and workspace behaviour.

Measure more than language quality: false PASS rate, false NOT_READY rate, requirement precision/recall/F1, evidence accuracy, contradiction detection, retrieval relevance, malformed-response rate, reasoning calls and latency. Layer ablations (deterministic only, plus agents, evidence, memory, neural reasoning, NLG) show which layer helps.

### Planned work

1. Connect the upstream ServiceNow downloader to the CR workspace contract.
2. Add page, cell and slide-level evidence provenance.
3. Persist useful agent patterns without persisting current approval as policy.
4. Add a fixed historical benchmark with leakage controls.
5. Add automatic regression checks for false-PASS cases.
6. Add bounded parallel document extraction per workspace.
7. Parse document status from structured tables where available.
8. Add evidence freshness and provenance metadata.

### Core principle

The goal is not to make the LLM decide everything. It is deterministic policy, specialist reasoning, current evidence, semantic memory, adversarial reasoning and controlled language, combined into Pre-CAB validation that is smarter but still auditable, traceable and reproducible.

---

Part of [Aryan Gupta's projects](https://github.com/AryanGupta-234). Building [NOBAD](https://nobad-platform.aassworld99.workers.dev/): software with a point of view.
