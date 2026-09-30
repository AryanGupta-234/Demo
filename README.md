
# Pre-CAB Validator — Full System Guide

Local-first, evidence-aware validation and reasoning engine for Normal ServiceNow Change Requests.

## Current integration boundary

The repository preserves the **real ServiceNow integration boundary** while also providing a local/demo path.

There are two supported ingestion modes:

    DEMO / LOCAL

    local CR JSON + evidence/<CR number>/
                    |
                    v
             this validator

    REAL SERVICE NOW

    ServiceNow REST API
             |
             v
    read-only API adapter
             |
             +--> CR fields
             +--> attachment metadata
             +--> attachment bytes
             |
             v
    this validator

The production-facing integration structure is already present in:

    src/pre_cab/servicenow.py
        provider-neutral ServiceNowAdapter contract

    src/pre_cab/servicenow_readonly.py
        bounded GET-only ServiceNow REST client

    scripts/validate_servicenow_change.py
        real ServiceNow CR validation entrypoint

The core validator remains independent of ServiceNow credentials and network behavior. The API adapter is responsible for authentication, CR retrieval and attachment retrieval; the validation pipeline remains responsible for validation, evidence verification, reasoning and the final gate.

For the current local demonstration phase, downloaded files can still be placed under:

    evidence/
      CHG001234/
        approval.pdf
        uat.xlsx
        implementation.docx
        rollback.pptx
        screenshot.png

The validator reads only the requested CR workspace. It never uses a sibling CR's evidence as proof for the current CR.

For the current phase, place downloaded files under:

    evidence/
      CHG001234/
        approval.pdf
        uat.xlsx
        implementation.docx
        rollback.pptx
        screenshot.png

The validator reads only the requested CR workspace. It never uses a sibling CR's evidence as proof for the current CR.

---


## Canonical end-to-end command

The primary local/demo entrypoint is:

    python main.py "C:\Users\aassw\Downloads\test1.json" --provider ollama

This is the canonical **single-command end-to-end path**. The command loads and normalizes the CR dataset, excludes Emergency changes by default, initializes the configured Ollama model, enables semantic memory when configured, and then runs the linked Pre-CAB pipeline for each in-scope CR:

    CR input
      -> normalization
      -> Level 1 deterministic validation
      -> specialist agents / blackboard
      -> CR-scoped evidence workspace
      -> file inventory + document extraction
      -> Level 2 evidence verification
      -> semantic memory retrieval
      -> Level 3 neural reasoning
      -> adversarial critique when applicable
      -> final conservative decision gate
      -> CAB + technical NLG
      -> audit + episodic memory

The command is intentionally **sequential at the CR level**. This keeps the default demo path simple, predictable and provider/budget friendly. It does not disable any validation layer.

Emergency changes are excluded by default:

    python main.py "C:\path\to\crs.json" --provider ollama --include-emergency

The semantic embedding model may download from Hugging Face on first use when semantic memory is enabled. This is separate from the Ollama generation model.

### Optional bounded parallel processing

The repository also contains an alternate programmatic parallel executor:

    src/pre_cab/parallel.py

It uses a bounded worker pool and runs one complete `run_pre_cab()` pipeline per CR while preserving input order in the returned results. Each CR keeps its own evidence workspace, so parallel execution does not intentionally mix evidence across CRs.

Use parallel execution when throughput matters, for example in a controlled batch/service worker. The canonical `python main.py ...` path remains the reference demo entrypoint.

## 1. Complete execution flow

    CR JSON
      |
      v
    normalize input
      |
      v
    LEVEL 1 — deterministic CR validation
      |
      +-- field requirements
      +-- contextual applicability
      +-- implementation
      +-- rollback
      +-- risk
      +-- conflict
      +-- testing
      +-- specialist agents
      +-- shared agent blackboard
      |
      v
    Level-1 result finalized
      |
      v
    CR-scoped evidence workspace
      |
      +-- inventory EVERY file
      +-- identify supported/unsupported files
      +-- extract supported documents
      +-- OCR textless/scanned pages
      +-- optional Ollama vision analysis for images
      +-- record parser failures
      +-- flag unresolved visual evidence
      |
      v
    LEVEL 2 — evidence verification
      |
      +-- current CR identity
      +-- testing / UAT
      +-- customer approval
      +-- QA signoff
      +-- lower-environment validation
      +-- rollback corroboration
      +-- contradictions
      |
      v
    LEVEL 3 — neural reasoning
      |
      +-- semantic memory
      +-- historical patterns
      +-- specialist observations
      +-- evidence excerpts
      +-- uncertainty analysis
      +-- adversarial critique
      +-- structured JSON reasoning
      +-- optional NLG realization
      |
      v
    conservative final decision gate
      |
      +-- PASS
      +-- CONDITIONAL
      +-- NOT_READY
      |
      +------------------+
      |                  |
      v                  v
    audit              memory
                       |
                       +-- episodic experience
                       +-- reviewer feedback
                       +-- semantic vectors
                       +-- consolidation

The important ordering is:

    Level 1 -> evidence discovery/extraction -> Level 2 -> neural reasoning -> final gate

---

## 2. Architecture by module

| Module | Responsibility |
|---|---|
| input_loader.py | Load and normalize CR exports |
| decision.py | Deterministic CR validation |
| field_requirement_engine.py | Data-driven contextual requirements |
| agents.py | Specialist CR agents |
| orchestrator.py | Agent orchestration and blackboard |
| evidence_workspace.py | CR workspace resolution/classification |
| local_attachments.py | File inventory and extraction |
| ocr.py | Local OCR and optional Ollama vision execution |
| rule_lifecycle.py | Holdout validation and automatic rule lifecycle promotion |
| evidence.py | Evidence verification |
| evidence_agents.py | Document-purpose specialists |
| evidence_retrieval.py | Relevant evidence excerpt retrieval |
| brain.py | Reasoning contracts/prompts |
| brain_loop.py | Reasoning, critique and NLG |
| final_reasoning.py | Neural reconciliation |
| semantic_memory.py | Persistent semantic memory |
| memory_consolidation.py | Feedback and consolidation |
| pipeline.py | Public Level 1 → Level 2 → Level 3 pipeline |
| audit_store.py | Persistent audit trail |
| reporting.py | CAB and technical output |
| runtime_provider.py | Model provider bootstrap |
| scripts/diagnose.py | Offline diagnostic |

---

# 3. Level 1 — CR validation

Level 1 is deterministic first.

It evaluates:

- CR type
- short description
- description
- justification
- implementation plan
- backout plan
- test plan
- risk
- configuration item
- environment
- conflict status
- signoff dispositions
- contextual requirements
- work-note signals

Finding severities:

    INFO
    WARNING
    BLOCKING

Decision mapping:

    BLOCKING -> NOT_READY
    WARNING  -> CONDITIONAL
    none     -> PASS

The neural model is not required for this stage.

---

# 4. Data-driven field requirements

FieldRequirementEngine resolves rules in this order:

    exact Category + Sub Category
            |
            v
    Category fallback
            |
            v
    global fallback

Requirement levels:

    REQUIRED
    CONDITIONAL
    RECOMMENDED
    OPTIONAL
    NOT_OBSERVED

Mined historical rules have an explicit lifecycle:

    CANDIDATE -> VALIDATED -> ACTIVE -> DEPRECATED

Historical frequency must not silently become policy.

---

# 5. Specialist agents

The CR is inspected by specialist agents including:

- Field
- Context
- Technical
- Business Impact
- Testing
- Risk
- Evidence
- Clone/Similarity
- Cross-Agent Consistency

Agents publish compact findings, requirements and observations to a shared blackboard.

The blackboard is advisory. Deterministic policy remains authoritative.

Conceptually:

    Context
       |
       v
    Technical
       |
       v
    Testing
       |
       v
    Business
       |
       v
    Risk
       |
       v
    Consistency

---

# 6. Evidence workspace

After Level 1 is finalized, the pipeline resolves the CR evidence workspace.

Preferred structure:

    evidence/
      CHG001234/
        approval.pdf
        test-results.xlsx
        implementation.docx
        rollback.pptx
        screenshot.png

The system also supports the older root-level CR-prefixed layout for compatibility.

The validator never guesses evidence from another CR.

---

# 7. Every file is inventoried

A major hardening change is that inventory is separate from extraction.

Every file in the CR workspace is visible in the evidence manifest.

Possible statuses include:

    supported
    unsupported
    too_large
    stat_error

This distinction matters:

    "not parsed"
       !=
    "does not exist"

Unsupported files therefore cannot disappear silently.

---

# 8. Supported formats

Current extraction supports:

Documents:
- PDF
- DOCX
- PPTX
- TXT
- Markdown
- CSV
- EML
- JSON

Spreadsheets:
- XLSX
- XLSM

Images:
- PNG
- JPG/JPEG
- WEBP
- BMP
- TIFF

Images are retained as evidence and marked for visual review because text extraction cannot establish their contents.

---

# 9. Attachment safety

Default per-file extraction limit:

    75 MiB

Configure:

Windows PowerShell:

    $env:PRE_CAB_MAX_ATTACHMENT_BYTES="157286400"

Linux/WSL:

    export PRE_CAB_MAX_ATTACHMENT_BYTES=157286400

Files above the limit remain visible in the inventory and receive an extraction error rather than being silently discarded.

PDF processing uses pypdf. Current pypdf releases expose resource limits intended to reduce excessive resource consumption from malformed documents.

---

# 10. Level 2 — evidence verification

Evidence is not accepted merely because a keyword exists.

The verifier asks:

    1. Does a relevant document exist?
    2. Does it identify the current CR?
    3. Does it support the claimed purpose?
    4. Is the status positive, negative or unclear?
    5. Does it contradict another current signal?
    6. Was the document actually readable?

Evidence categories:

- customer approval
- UAT
- test execution/results
- QA signoff
- lower-environment validation
- rollback/recovery
- implementation/runbook evidence

---

# 11. Evidence identity

A strong evidence document should identify the current CR in its filename or content.

Example:

    CR:
    CHG001234

    File:
    CHG001234_UAT_Report.pdf

    Text:
    CHG001234
    UAT completed
    Expected Result: PASS
    Actual Result: PASS

A generic document saying only "UAT passed" is not equivalent to CR-specific proof.

---

# 12. Testing model

The system separates:

    Test Plan
       !=
    Test Execution
       !=
    Formal Test Results Evidence

A plan such as "test in UAT" does not prove execution.

The execution parser was hardened to prefer the latest explicit status. This reduces false failures when a document contains historical text such as:

    Previous test failed.
    Remediation completed.
    Current test passed.

---

# 13. Approval model

Negative approval states are checked before positive terms.

Examples:

    not approved
    rejected
    declined
    approval pending
    awaiting approval

This avoids accepting a document simply because it contains the word "approved".

---

# 14. Environment reasoning

The workflow treats an unset Environment according to the existing PROD workflow rules.

DEV, SIT, staging, UAT and Pre-PROD references are validation context.

A document that proves only DEV testing cannot automatically prove production readiness.

Example:

    CR target = PROD
    Evidence = DEV testing completed

This is surfaced as a contradiction.

---

# 15. Rollback

Rollback is evaluated independently from implementation.

A credible recovery mechanism may include:

    restore previous artifact
    restore backup
    revert deployment
    restore previous release
    revert configuration

"Rollback if needed" is not equivalent to a concrete recovery mechanism.

Attachment evidence may corroborate the CR rollback plan.

Historical documents do not prove the current CR rollback.

---

# 16. OCR / vision execution

Image and scanned-document processing is now executable locally.

For raster images:
    
    image
      |
      +--> Tesseract OCR
      |
      +--> optional Ollama vision model
      |
      v
    derived text / visual description
      |
      v
    evidence reasoning context

For textless or mixed PDFs:

    PDF
      |
      v
    native PDF text extraction
      |
      +--> textless pages rendered with PyMuPDF
                   |
                   v
              Tesseract OCR
                   |
                   v
             combined evidence text

OCR metadata is recorded in the evidence manifest, including whether OCR ran,
the engine, confidence where available, page count, and OCR errors.

Install the Python OCR dependencies through the normal documentation/full
installation. The Tesseract executable itself is a local system dependency.

Windows can use the standard Tesseract installation path or:

    $env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"

Optional Ollama vision:

    $env:PRE_CAB_VISION_MODEL="your-local-vision-capable-model"

Vision output is explicitly marked as derived analysis and is not treated as
direct evidence. This prevents a model-generated visual description from being
silently promoted to a verified fact.


---

# 16. Unreadable evidence

The system explicitly surfaces:

    EVIDENCE_UNREADABLE
    EVIDENCE_VISION_REVIEW_REQUIRED

Examples:

- scanned PDF
- screenshot
- image-only document
- encrypted/unreadable document
- parser failure
- oversized file

The system does not silently interpret unreadable evidence as proof.

---

# 17. Level 3 — neural reasoning

The neural engine receives:

- current CR
- Level-1 findings
- specialist observations
- Level-2 evidence state
- ranked evidence excerpts
- semantic historical memory
- contradictions
- uncertainties
- applicable requirements

Reasoning is structured as:

    FACTS
      |
      v
    INFERENCES
      |
      v
    UNCERTAINTIES
      |
      v
    CONTRADICTIONS
      |
      v
    RECOMMENDATIONS
      |
      v
    DECISION

The model is explicitly constrained not to invent current evidence.

---

# 18. Self-critique

The reasoning loop challenges itself with questions such as:

- What evidence would make the conclusion wrong?
- Which claims are unverified?
- Did one keyword determine impact?
- Was UAT incorrectly treated as universal?
- Was test planning confused with execution?
- Is there a current evidence contradiction?
- Was historical memory mistaken for current proof?
- Is rollback actually credible?

The critique can make the result more conservative.

It cannot erase deterministic blockers.

---

# 19. NLG architecture

Reasoning and language realization are separate:

    semantic reasoning
           |
           v
    locked decision/facts
           |
           v
    NLG realization
           |
           +--> CAB narrative
           |
           +--> technical narrative

NLG cannot change:

- decision
- confidence
- facts
- evidence status
- uncertainty

It only improves presentation.

The NLG contract also suppresses repetitive AI filler and separates CAB language from engineering language.

---

# 20. Final decision gate

Decision order is conservative:

    PASS
      |
      v
    CONDITIONAL
      |
      v
    NOT_READY

Examples:

    Stage 1 = NOT_READY
    Stage 2 = PASS
    Final   = NOT_READY

    Stage 1 = PASS
    Stage 2 = NOT_READY
    Final   = NOT_READY

The neural layer may discover additional risk and downgrade readiness.

It cannot convert a deterministic or evidence blocker into PASS.

---

# 21. Persistent semantic memory

The previous memory system relied primarily on SQLite/FTS retrieval.

The upgraded architecture is:

    memory record
        |
        +------------------+
        |                  |
        v                  v
    SQLite row        local embedding
                           |
                           v
                    persistent vector
                           |
                           v
                    semantic retrieval
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
       semantic         lexical         identifiers
       similarity        overlap           / CI
          |                |                |
          +----------------+----------------+
                           |
                     authority/recency
                           |
                           v
                         MMR
                           |
                           v
                    reasoning context

Semantic retrieval is primary while identifiers, authority, confidence and recency remain important governance signals.

---

# 22. Memory consolidation

Repeated episodes can become compact non-authoritative patterns:

    episode
    episode
    episode
       |
       v
    semantic clustering
       |
       v
    consolidated pattern

Consolidated memories are marked:

    authoritative = false
    current_cr_evidence = false

Therefore historical learning cannot silently become current approval evidence.

---

# 23. Feedback learning

Reviewer feedback stores:

- predicted decision
- actual decision
- reviewer notes
- corrected requirements
- lessons

Memory utility receives positive or negative reinforcement.

This changes retrieval/learning metadata, not the original evidence.

---

# 24. Audit

Each run records:

- CR number
- strictness
- Stage-1 decision
- Stage-2 decision
- final decision
- model
- evidence manifest
- reasoning
- findings
- audit run ID

---


# 25. Implementation status / boundaries

| Capability | Status | Current implementation boundary |
|---|---|---|
| OCR / vision execution | ✅ **Implemented (local)** | Raster images are OCR-processed with Tesseract; textless/mixed PDFs are rendered with PyMuPDF and OCR-processed. An optional Ollama vision model can additionally inspect raster evidence and its output is marked as derived, not direct evidence. |
| Automatic rule lifecycle promotion | ✅ **Implemented with validation gates** | Candidate rules can be automatically evaluated against a separate Normal-CR holdout set and promoted to VALIDATED or ACTIVE only when objective stability thresholds pass. |
| README installer description | ✅ **Updated** | The documented default `python install.py` path now matches the actual lightweight manager-demo installer; `python install.py --full` installs the complete AI/API/development stack and runs tests. |

These boundaries are intentional. They prevent the documentation from implying capabilities that are not currently executed by the code.

# 26. Automatic rule lifecycle promotion

Rule mining remains data-driven, but candidate rules can now be automatically
validated and promoted against a separate holdout dataset.

Recommended flow:

    historical training CRs
             |
             v
    mine_field_requirements.py
             |
             v
          CANDIDATE
             |
             v
       holdout validation
             |
       +-----+------+
       |            |
       v            v
    VALIDATED      ACTIVE
                 (only when the stricter gate passes)

The validation checks requirement-class stability between the mined candidate
table and a separate Normal-CR holdout set. It also prevents ACTIVE promotion
when candidate REQUIRED rules fall below the configured holdout population floor.

One-step mine + automatic promotion:

    python scripts/mine_field_requirements.py historical_train.json ^
      --holdout historical_holdout.json ^
      --out config/field_requirements.generated.json

Or validate an existing candidate:

    python scripts/promote_field_requirements.py ^
      config/field_requirements.generated.json ^
      historical_holdout.json ^
      --out config/field_requirements.generated.json

The promotion output records the validation method, thresholds, stability rate,
required-rule regressions, and lifecycle history in the rule-table JSON.

This is automatic promotion **after explicit data validation**; it is not an
unbounded self-modifying policy mechanism.

# 26. Installation — two options

There are two supported installation paths.

### Option A — automatic installer (recommended)

The repository includes a single cross-platform installer:

    install.py

It automatically:

1. checks that Python 3.11+ is available
2. creates or reuses `.venv`
3. upgrades pip/setuptools/wheel
4. installs the dependencies for the selected mode
5. compiles the source and scripts
6. runs the manager demo in default mode, or the full pytest suite with `--full`
7. prints the exact command to activate/run the environment

#### Windows PowerShell

    git clone https://github.com/AryanGupta-234/Demo.git
    cd Demo
    python install.py

Default mode is intentionally lightweight and prepares the manager demo.

For the complete AI/API/development installation and full tests:

    python install.py --full

If `python` is not available but `py` is:

    py install.py --full

#### Linux / WSL

    git clone https://github.com/AryanGupta-234/Demo.git
    cd Demo
    python3 install.py

After installation:

Windows:

    .\\.venv\\Scripts\\Activate.ps1

Linux / WSL:

    source .venv/bin/activate

The installer is intentionally non-destructive: an existing `.venv` is reused instead of deleted.

### Option B — manual installation

Use this when you want full control over every installation step.

#### Windows

Install Python 3.11+ and Git.

Verify:

    python --version
    git --version

Clone:

    git clone https://github.com/AryanGupta-234/Demo.git
    cd Demo

Create and activate the environment:

    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1

If PowerShell blocks activation:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Upgrade tooling:

    python -m pip install --upgrade pip setuptools wheel

Install all runtime/document/API/embedding/test dependencies:

    pip install -e ".[all,dev]"

Equivalent repository requirements file:

    pip install -r requirements.txt

#### Linux / WSL

Ubuntu:

    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip git

Clone:

    git clone https://github.com/AryanGupta-234/Demo.git
    cd Demo

Create and activate:

    python3 -m venv .venv
    source .venv/bin/activate

Upgrade tooling:

    python -m pip install --upgrade pip setuptools wheel

Install:

    pip install -e ".[all,dev]"

Or:

    pip install -r requirements.txt

### What the installer does not install

Ollama is a separate system application, not a Python dependency. Install it separately if you want local neural reasoning, then verify:

    ollama list

Example model:

    ollama pull qwen2.5:7b-instruct

The installer also does not create ServiceNow credentials or API keys. Those remain environment/deployment configuration.

# 29. Selective installation

Core only:

    pip install -e .

Documents:

    pip install -e ".[docs]"

Embeddings:

    pip install -e ".[embeddings]"

API:

    pip install -e ".[api]"

Everything:

    pip install -e ".[all]"

The docs extra covers PDF, XLSX/XLSM, DOCX, PPTX extraction plus the Python OCR bindings. The local Tesseract executable is a separate system dependency.

---

# 30. Installation verification

Run:

    python -m compileall src scripts main.py
    python -m pytest

Then:

    python scripts/diagnose.py ./data/cr_export.json --evidence-root ./data/evidence

---

# 31. Offline diagnostic

The diagnostic tool checks:

- Python/runtime
- document dependencies
- embedding dependency
- CR normalization
- Level-1 decision
- blocking findings
- field requirements
- evidence workspace
- complete file inventory
- supported/unsupported files
- extraction failures

Run:

    python scripts/diagnose.py ./data/cr_export.json \
      --evidence-root ./data/evidence

Machine-readable:

    python scripts/diagnose.py ./data/cr_export.json \
      --evidence-root ./data/evidence \
      --json

---

# 32. Local Ollama

The local provider defaults to:

    http://localhost:11434

Verify Ollama:

    ollama list

Pull/configure the model used by the environment.

Example:

    ollama pull qwen2.5:7b-instruct

Run server if needed:

    ollama serve

Run the validator:

    python main.py ./data/one_cr.json --provider ollama

Optional:

    OLLAMA_MODEL
    OLLAMA_BASE_URL
    OLLAMA_NUM_CTX

Windows PowerShell example:

    $env:OLLAMA_MODEL="qwen2.5:7b-instruct"
    $env:OLLAMA_BASE_URL="http://localhost:11434"
    $env:OLLAMA_NUM_CTX="32768"

---

# 33. Ollama model compatibility

Ollama is treated as a runtime/provider boundary, not as a Qwen-only implementation.

The provider accepts the model name from `OLLAMA_MODEL`, so the same Pre-CAB pipeline can use different Ollama models without changing the validation, evidence, memory, or final-gate code.

Example:

    $env:OLLAMA_MODEL="qwen2.5:7b-instruct"
    python main.py ./data/one_cr.json --provider ollama

Switching models:

    $env:OLLAMA_MODEL="llama3.1:8b"
    python main.py ./data/one_cr.json --provider ollama

A fine-tuned/custom Ollama model can be selected the same way:

    $env:OLLAMA_MODEL="pre-cab-qwen"
    python main.py ./data/one_cr.json --provider ollama

Compatibility depends on the capabilities of the selected model. The model should be suitable for the provider's chat/structured-output contract and have enough context for the configured workload. Context length, tool/function calling, JSON reliability and reasoning quality can vary by model.

Important separation:

    Ollama model
        = neural reasoning / language realization

    Deterministic validators + evidence gate
        = authoritative safety/decision controls

    Embedding model
        = separate semantic-memory retrieval component

Changing the Ollama generation model therefore does not change the deterministic Pre-CAB policy engine. A weaker or incompatible model may produce lower-quality reasoning, but it must not be able to override deterministic or evidence blockers.

The repository's Qwen/QLoRA training path is an example model-specific training workflow; it is not a requirement of the Ollama provider itself.

---

# 34. Cloud provider

Groq:

    $env:GROQ_API_KEY="..."
    python main.py .\data\one_cr.json --provider groq

Linux:

    export GROQ_API_KEY="..."
    python main.py ./data/one_cr.json --provider groq

Hugging Face:

    export HF_TOKEN="..."
    python main.py ./data/one_cr.json --provider huggingface

Automatic provider routing:

    python main.py ./data/one_cr.json --provider auto

Never commit API keys.

---

# 35. Semantic memory configuration

Defaults:

    PRE_CAB_SEMANTIC_MEMORY=true
    PRE_CAB_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    PRE_CAB_SEMANTIC_WEIGHT=0.65

Disable:

    PRE_CAB_SEMANTIC_MEMORY=false

Evidence size:

    PRE_CAB_MAX_ATTACHMENT_BYTES=78643200

NLG:

    PRE_CAB_NLG_MODE=refine

Cache:

    PRE_CAB_MODEL_CACHE=1
    PRE_CAB_MODEL_CACHE_PATH=.pre_cab/model_cache.sqlite3

Budget guard:

    PRE_CAB_BUDGET_GUARD=1

---

# 36. Single CR

Basic:

    python main.py ./data/one_cr.json --provider ollama

Strict:

    python main.py ./data/one_cr.json --strictness strict --provider ollama

Explicit evidence root:

    python main.py ./data/one_cr.json \
      --attachment-root ./data/evidence \
      --provider ollama

---

# 37. Batch

    python scripts/run_batch.py ./data/cr_export.json \
      --output artifacts/cr_results.jsonl

Evidence-aware:

    python scripts/run_batch.py ./data/cr_export.json \
      --attachments ./data/evidence \
      --output artifacts/cr_results.jsonl

---

# 38. Recommended directory

    project/
      data/
        cr_export.json
        evidence/
          CHG001234/
            approval.pdf
            test-results.xlsx
            implementation.docx
            rollback.pptx
            screenshot.png

      .pre_cab/
        memory.sqlite3
        model_cache.sqlite3
        audit.sqlite3

      artifacts/
        pre_cab/

Keep production data outside Git.

Recommended ignore entries:

    .env
    .pre_cab/
    artifacts/
    data/
    real_data/
    evidence/
    attachments/
    *.sqlite3

---

# 39. Current outputs

For a CR:

    artifacts/pre_cab/
      CHG001234_pre_cab.json
      CHG001234_pre_cab.txt
      batch_results.json

The JSON includes:

- final decision
- confidence
- findings
- evidence manifest
- workspace inventory
- extraction errors
- agent state
- reasoning
- model information
- audit ID

---

# 40. Demo vs real production integration

The project deliberately keeps the **same core validator** behind both the demonstration path and the real ServiceNow path.

| Area | Current demonstration | Real / production work |
|---|---|---|
| CR input | Local JSON fixture | ServiceNow REST CR retrieval |
| Authentication | None required | ServiceNow credentials / approved auth mechanism |
| Attachments | Local `evidence/<CR>/` folder | ServiceNow attachment API |
| Document extraction | Local bounded extraction | Same extraction after API retrieval |
| Level 1 validation | Real deterministic engine | Same |
| Specialist agents | Real | Same |
| Level 2 evidence verification | Real | Same |
| Neural reasoning | Configurable local/cloud model | Same, subject to deployment/model policy |
| Semantic memory | Local persistent store | Replaceable production persistence if required |
| Audit | Local SQLite | Production audit store/integration if required |
| ServiceNow writes | Not performed by read-only client | Explicitly implement/authorize separately if required |

### Real ServiceNow path already present

The read-only client performs bounded GET operations for:

    ServiceNow change_request
            |
            +--> CR fields
            |
            +--> attachment metadata
            |
            +--> attachment bytes
            |
            v
    EvidenceDocument records
            |
            v
    run_pre_cab(...)

It includes retry handling for transient transport/429/5xx failures, authentication failure handling, attachment size limits, and in-memory attachment extraction. Credentials are read from environment variables and are not written into reports.

Run a real ServiceNow CR validation with:

    $env:SERVICENOW_BASE_URL="https://your-instance.service-now.com"
    $env:SERVICENOW_USERNAME="..."
    $env:SERVICENOW_PASSWORD="..."
    python scripts/validate_servicenow_change.py CHG001234 --provider ollama

The script is intentionally **read-only**. It does not approve the CR or write status/work notes back to ServiceNow.

### What still needs organization-specific production work

- approved authentication method (Basic is implemented as a controlled reference client; OAuth/service account may be required by the organization)
- exact ServiceNow instance/table/query configuration
- attachment retention/download policy
- production secret management
- production logging/observability
- rate limits and enterprise retry policy
- production audit persistence
- authorization and network controls
- optional, separately reviewed ServiceNow write-back workflow
- enterprise scheduling/orchestration

The clean boundary is:

    ServiceNow adapter
          |
          v
    normalized CR + evidence
          |
          v
    Pre-CAB core

The core must not contain ServiceNow credentials or depend on ServiceNow-specific network behavior.

---

# 41. Failure containment

Missing embedding model:

    semantic retrieval unavailable
          |
          v
    lexical compatibility fallback

Missing LLM:

    deterministic validation continues

Broken document:

    document remains inventoried
          |
          v
    extraction error is surfaced

Malformed model JSON:

    bounded repair
          |
          v
    if still invalid -> deterministic result retained

Evidence blocker + model PASS:

    evidence blocker wins
          |
          v
    NOT_READY

---

# 42. Security rules

Never commit:

- production CR exports
- customer approvals
- screenshots
- ServiceNow attachments
- credentials
- API keys
- .env
- .pre_cab databases
- production artifacts

Historical memory is context, not current evidence.

---

# 43. Test strategy

Tests cover:

- field validation
- requirement resolution
- evidence verification
- CR identity matching
- unreadable evidence
- environment contradictions
- clone/delta analysis
- pipeline decision merging
- neural downgrade protection
- model caching
- audit persistence
- semantic memory persistence
- feedback
- blackboard consistency
- evidence workspace behavior

Run:

    python -m pytest

Also:

    python -m compileall src scripts main.py

---

# 44. Recommended evaluation

Do not measure only language quality.

Measure:

    false PASS rate
    false NOT_READY rate
    requirement precision/recall/F1
    evidence verification accuracy
    contradiction detection
    clone quality
    retrieval relevance
    malformed model-response rate
    reasoning calls
    latency
    memory quality

Run layer ablations:

    deterministic only
    deterministic + agents
    + evidence
    + memory
    + neural reasoning
    + NLG

This shows exactly which layer improves the system.

---

# 45. Next major upgrades

1. Connect the existing upstream ServiceNow downloader to the CR workspace contract.
3. Add page/cell/slide-level evidence provenance.
4. Persist useful agent patterns without persisting current approval as policy.
4. Add a fixed historical benchmark with leakage controls.
5. Add automatic regression checks for false-PASS cases.
6. Add workspace-level parallel document extraction with bounded concurrency.
7. Add stronger document status parsing using structured tables where available.
8. Add explicit evidence freshness/provenance metadata.
9. Add a persistent cognitive blackboard layered above semantic memory.

---

# 46. Core engineering principle

The goal is not:

    "make the LLM decide everything"

The goal is:

    deterministic policy
           +
    specialist reasoning
           +
    current evidence
           +
    semantic historical memory
           +
    adversarial neural reasoning
           +
    controlled language realization
           =
    more intelligent but still auditable Pre-CAB validation

The neural engine should become more semantic, contextual and adaptive without becoming less deterministic, traceable or reproducible.
