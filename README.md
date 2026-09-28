
# Pre-CAB Validator — Full System Guide

Local-first, evidence-aware validation and reasoning engine for Normal ServiceNow Change Requests.

## Current integration boundary

This repository does NOT fetch CRs or attachments from ServiceNow.

The current contract is:

    ServiceNow / upstream fetcher
             |
             v
    private CR workspace
             |
             v
    this validator

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
      +-- record parser failures
      +-- flag image/scanned evidence
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

# 25. Scratch installation — Windows

Install Python 3.11+ and Git.

Verify:

    python --version
    git --version

Clone:

    git clone https://github.com/AryanGupta-234/Demo.git
    cd Demo

Create environment:

    python -m venv .venv

Activate:

    .\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Upgrade tooling:

    python -m pip install --upgrade pip setuptools wheel

Install everything:

    pip install -r requirements.txt

Equivalent:

    pip install -e ".[all]"

---

# 26. Scratch installation — Linux / WSL

Ubuntu:

    sudo apt update
    sudo apt install -y python3 python3-venv python3-pip git

Clone:

    git clone https://github.com/AryanGupta-234/Demo.git
    cd Demo

Environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install:

    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt

---

# 27. Selective installation

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

The docs extra currently covers PDF, XLSX/XLSM, DOCX and PPTX extraction.

---

# 28. Installation verification

Run:

    python -m compileall src scripts main.py
    python -m pytest

Then:

    python scripts/diagnose.py ./data/cr_export.json --evidence-root ./data/evidence

---

# 29. Offline diagnostic

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

# 30. Local Ollama

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

# 31. Cloud provider

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

# 32. Semantic memory configuration

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

# 33. Single CR

Basic:

    python main.py ./data/one_cr.json --provider ollama

Strict:

    python main.py ./data/one_cr.json --strictness strict --provider ollama

Explicit evidence root:

    python main.py ./data/one_cr.json \
      --attachment-root ./data/evidence \
      --provider ollama

---

# 34. Batch

    python scripts/run_batch.py ./data/cr_export.json \
      --output artifacts/cr_results.jsonl

Evidence-aware:

    python scripts/run_batch.py ./data/cr_export.json \
      --attachments ./data/evidence \
      --output artifacts/cr_results.jsonl

---

# 35. Recommended directory

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

# 36. Current outputs

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

# 37. What this repository does NOT do yet

It does not own:

- ServiceNow authentication
- ServiceNow API fetching
- attachment downloading
- ServiceNow scheduling
- upstream file lifecycle management

The intended integration is:

    ServiceNow API
         |
         v
    CR fetcher
         |
         v
    attachment downloader
         |
         v
    evidence/<CR number>/
         |
         v
    Pre-CAB Validator

That boundary should remain clean.

---

# 38. Failure containment

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

# 39. Security rules

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

# 40. Test strategy

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

# 41. Recommended evaluation

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

# 42. Next major upgrades

1. Connect the existing upstream ServiceNow downloader to the CR workspace contract.
2. Add OCR/vision for scanned PDFs and image attachments.
3. Add page/cell/slide-level evidence provenance.
4. Persist useful agent patterns without persisting current approval as policy.
5. Add a fixed historical benchmark with leakage controls.
6. Add automatic regression checks for false-PASS cases.
7. Add workspace-level parallel document extraction with bounded concurrency.
8. Add stronger document status parsing using structured tables where available.
9. Add explicit evidence freshness/provenance metadata.
10. Add a persistent cognitive blackboard layered above semantic memory.

---

# 43. Core engineering principle

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
