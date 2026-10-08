# Architecture Decision Memo

**Maximum length: 500 words**

## Decision

Ship **Architecture A (single agent)**.

## Evidence

Both architectures were evaluated against the 21 benchmark cases (10 starter, 11 synthetic edge cases) using `gemini-3.1-flash-lite` and the mock vendor-risk service.

| Metric | Single agent | Staged / 2-agent |
|---|---:|---:|
| Cases passing your quality criteria | 19/21 strict (90.5%), 21/21 category (100%) | 19/21 strict (90.5%), 21/21 category (100%) |
| Avg latency | 20.4 s (20,375 ms) | 30.0 s (29,988 ms) |
| Avg LLM calls | 3.0 (63 total) | 4.0 (84 total) |
| Avg tool calls | 4.95 (104 total) | 4.86 (102 total) |
| Notable policy/grounding failures | 0 missed rules; 2 over-escalations (BASE-01 extra Procurement; BASE-06 extra security flag) | 0 missed rules; 2 over-escalations (BASE-06 extra privacy/security flags; SYN-02 extra Privacy approval/flag) |

**Key findings:**
- Both architectures achieved identical strict pass rates (19/21) and category accuracy (21/21, 100%), with zero LLM fallbacks across 42 evaluation runs.
- All 4 failures across both architectures were conservative over-escalations; neither agent ever omitted a mandatory approval or policy flag.
- Architecture A over-escalated on BASE-01 (adding Procurement) while Architecture B's reviewer matched there; conversely, Architecture B added an unneeded Privacy approval and flag on SYN-02. Thus, neither architecture was more precise.
- The deterministic policy engine alone achieved 21/21 strict pass in under 2 ms.

## Trade-offs

- **Cost & Latency:** Architecture A uses 25% fewer LLM calls (3.0 vs 4.0) and is ~32% faster (20.4 s vs 30.0 s mean latency; 24.7 s vs 40.9 s p95). Architecture B adds orchestration latency and token cost with zero accuracy gain.
- **Guardrail Philosophy:** Our post-LLM guardrail unions LLM-identified risks with deterministic policy floors. This preserves valid LLM caution for enterprise compliance at the expense of strict synthetic match precision. Clamping outputs strictly to the deterministic floor would score 21/21 on benchmarks but make LLM reasoning moot.

## Risks / limitations

Before enterprise production deployment, we must validate:
1. **Broader Evaluation:** Benchmark covers 21 synthetic cases on one model (`gemini-3.1-flash-lite`). Real distributions require hundreds of historical corporate requests across multiple models.
2. **Catalog Telemetry:** Tool overlap detection lacks live SaaS seat utilization data to determine if existing licenses can be reclaimed.
3. **Adversarial Hardening:** Regex injection detection must be augmented with semantic classifiers.
4. **Rate Limits & Mock Dependencies:** Production requires production API quotas and resilient integration with live vendor risk platforms.

## Why this is the right MVP

Architecture A solves the client problem with minimal orchestration complexity:
1. **Clear Division of Responsibility:** Deterministic Python code rigidly enforces dollar thresholds and mandatory compliance floors, while the LLM provides contextual synthesis, ambiguity resolution, and `use_existing_tool` judgment.
2. **Human-in-the-Loop:** All outputs are strictly advisory; human approval remains mandatory.
3. **Operational Simplicity:** A single agent loop with bound tools is easier to debug, trace, and maintain than multi-agent pipelines, while delivering equal accuracy at lower cost and latency.
