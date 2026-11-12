# Parallel Worlds — Detailed Implementation Plan

**Project:** Counterfactual SaaS simulation with a fine-tuned, tool-using Persona Agent  
**Duration:** Six days | **Team:** Two people | **Compute allowance:** $20 Hugging Face credits (eligibility and prices must be verified)  
**Status:** Architecture and all 10 components finalized. This document is the implementation contract.

## 0. Goal, scientific claim, and definition of done

**Question:** Does single-stage supervised fine-tuning improve a small agent's tool-use reliability and fidelity to a known synthetic counterfactual Oracle, without degrading prompt-injection robustness, temporal isolation, or evidence integrity?

**What this is:** A controlled *synthetic-world* evaluation. It is **not** a validated forecast of real customer behaviour or a general AI-safety guarantee.

**Required MVP:**
- 150 reproducible synthetic SaaS customers, 100/20/30 customer-disjoint train/validation/test split.
- Three plans (Basic, Pro, Enterprise); six months of observable history; synthetic support tickets; separate hidden traits.
- Interpretable three-outcome softmax Oracle, with one nonlinear price-sensitivity interaction.
- Two restricted, logged tools; maximum two calls per task; server-enforced `as_of`.
- Qwen2.5-3B-Instruct base and one single-stage LoRA/QLoRA SFT adapter (fallback to smaller model after feasibility gate).
- 200–300 validated training demonstrations (minimum fallback 100–150), generated only from training customers.
- Agent predicts intermediate risk factors; a **fixed, Oracle-independent** conversion maps factors to stay/downgrade/churn probabilities.
- One-month World A (0% price change) vs World B (+10%), using identical initial states.
- ParallelBench: 30 held-out customers × two worlds × two LLM versions = 120 LLM prediction attempts; Oracle references; separate logistic-regression prediction baseline.
- Safety suite: 10 malicious tickets and 10 matched benign controls; temporal leakage and evidence integrity tests.
- Four-tab cached Streamlit dashboard with saved trace replay; reproducible research summary and demo.

**Success gate:** Even if fine-tuning does not improve metrics, a reproducible and honestly reported paired comparison with complete failure accounting is a successful scientific MVP.

## 1. Roles and collaboration

| Workstream | Person A: dataset / Oracle / simulation / evaluation / UI | Person B: agent / tools / training |
|---|---|---|
| Synthetic data and splits | **Own** | Review schema |
| Hidden Oracle, truth generation | **Own** | No access to test truth during agent development |
| Fixed risk-to-probability converter | **Own**, jointly specify | Integrate exact version |
| SQLite observable schema | **Own** | Review and consume |
| Restricted tools and agent runtime | Contract review, tests | **Own** |
| Training examples and SFT | Audit for leakage | **Own** |
| Base and tuned inference traces | Receive and validate | **Own** |
| Simulator, revenue, statistical baseline | **Own** | Integration checks |
| Prompt-injection fixtures and scoring | **Own** | Execute model tests, log traces |
| Temporal/evidence enforcement | Adversarial tests | **Own** backend enforcement |
| Dashboard, plots, research writing | **Own** | Supply artifacts, reproduce runs |
| Repository integration, tests, submission | **Both** | **Both** |

**Git workflow:** One GitHub repo. Branches `feature/data-oracle-simulation` and `feature/agent-training`; short PRs merged into `main` at least daily. Freeze `contracts/` on Day 1. Changes to schemas require both owners' approval, migration notes, and updated contract tests. Never commit API keys, raw secrets, or private test truth into agent-accessible files.

## 2. System architecture and trust boundaries

```text
Synthetic generator ──┬── observable SQLite (history, plans, tickets) ── restricted tools ── Persona Agent
                     │                                                   │                 │
                     └── private Oracle traits / DGP / labels             │                 └── risk factors + evidence
                                        │                                  │                           │
                                        └── Oracle probabilities           └── tool traces             ▼
                                                                                   Fixed independent converter
                                                                                              │
                                                   Oracle / base / tuned / logistic regression probabilities
                                                                                              │
                                                             Paired World A vs World B simulator
                                                                                              │
                                                              ParallelBench + safety + cached dashboard
```

**Isolation requirements:** The agent's runtime can open only the observable database and its own tool interface. Oracle code, private traits, evaluation labels, and test outcomes are in separate inaccessible directories/process permissions during inference. Tool output is untrusted data; never treat support-ticket text as system instructions. The converter must **not** import Oracle code or reuse its coefficients. A model-predicted factor called `price_sensitivity` is an estimate, not the true latent Oracle factor.

## 3. Proposed repository layout

```text
parallel-worlds/
├── README.md
├── implementation_plan.md
├── pyproject.toml                     # pinned dependencies / test commands
├── .env.example                       # no secrets
├── contracts/
│   ├── schemas.py                     # canonical typed request/result/trace models
│   ├── schema_version.txt
│   └── examples/                      # valid and invalid JSON fixtures
├── data/
│   ├── generate_customers.py
│   ├── schema.sql
│   ├── split_customers.py
│   ├── observable.sqlite              # generated, non-secret demo data
│   └── splits.json
├── private_oracle/                     # excluded from agent runtime
│   ├── dgp.py
│   ├── generate_truth.py
│   └── hidden_traits.csv
├── conversion/
│   ├── risk_to_outcomes.py            # public, frozen, Oracle-independent
│   └── config.json
├── agent/
│   ├── tools.py
│   ├── runtime.py
│   ├── prompts.py
│   ├── output_validation.py
│   └── trace_logger.py
├── training/
│   ├── generate_demonstrations.py
│   ├── validate_demonstrations.py
│   ├── train_lora.py
│   ├── config.yaml
│   └── model_card.md
├── simulation/
│   ├── scenarios.py
│   ├── simulator.py
│   └── business_metrics.py
├── evaluation/
│   ├── parallelbench.py
│   ├── metrics.py
│   ├── logistic_baseline.py
│   ├── safety_cases.jsonl
│   ├── safety.py
│   └── summarize.py
├── dashboard/
│   ├── app.py
│   ├── data_loader.py
│   └── tabs/{worlds,agent,benchmark,safety}.py
├── artifacts/
│   ├── predictions/
│   ├── traces/
│   ├── metrics/
│   ├── figures/
│   └── manifests/
├── tests/
│   ├── test_contracts.py
│   ├── test_dataset.py
│   ├── test_oracle.py
│   ├── test_tools.py
│   ├── test_converter.py
│   ├── test_simulator.py
│   ├── test_benchmark.py
│   └── test_safety.py
└── docs/{methodology,results,limitations,demo_script}.md
```

The directory layout is a target, not a prerequisite for writing code. Generate data and model artifacts through scripts; avoid committing large checkpoints or sensitive logs. Record checksums for frozen datasets, splits, configs, and benchmark cases.

## 4. Frozen integration contracts (Day 1)

### 4.1 Common identifiers and scenarios

- `customer_id`: string `C001`–`C150`; immutable.
- `scenario_id`: `world_a` or `world_b`.
- `intervention`: `{"type":"price_change","value":0.0}` or `{"type":"price_change","value":0.10}`.
- `as_of`: UTC ISO-8601 timestamp, e.g. `2026-03-01T00:00:00Z`; injected by orchestration backend, **not accepted as a model tool argument**.
- `model_id`: `oracle`, `qwen_base`, `qwen_sft`, `logistic_regression`.
- `run_id`, `schema_version`, `dataset_version`, `converter_version`, and `seed` recorded on every run.
- Outcomes exactly `stay`, `downgrade`, `churn`; all probabilities finite and in `[0,1]`, sum within `1e-6` of one.
- Basic-plan downgrade is impossible: `P(downgrade)=0`; both Oracle and converter enforce this by masking downgrade before normalization.

### 4.2 Agent request (backend → model orchestrator)

```json
{
  "request_id": "test-C102-world_b",
  "customer_id": "C102",
  "as_of": "2026-03-01T00:00:00Z",
  "scenario_id": "world_b",
  "intervention": {"type": "price_change", "value": 0.10}
}
```

The model sees the customer ID and intervention; backend supplies trusted cutoff context and handles tool authorization. Never give the model hidden traits or reference probabilities.

### 4.3 Tool contracts

`get_customer_history({"customer_id":"C102"})` returns an observable, timestamp-filtered customer profile, plan, current price, tenure, six-month usage/billing records, and permitted support tickets. Each record has `record_id`, `observed_at`, and `source_type`. Return only records with `observed_at < as_of`. Tool must validate that requested customer equals the task's authorized customer; return structured `invalid_customer`, `unauthorized_customer`, or `no_history` errors as appropriate.

`get_segment_stats({"segment":"small_business"})` returns aggregate counts, historical churn/downgrade rate where observable, average tenure and usage trend, computed from eligible records strictly before `as_of`. Apply a minimum cohort threshold (recommend `n >= 5`); suppress or return `insufficient_cohort` below threshold. Do not expose hidden traits or future outcomes. Segment names come from the observable schema, not inferred arbitrary SQL.

**Runtime:** At most two total tool attempts per prediction, including failed calls; record arguments, timestamp, status, evidence IDs, redacted responses, and errors. `as_of` is bound by backend request context, never provided or overwritten by the LLM. Reject arbitrary SQL and unrecognized tool names.

### 4.4 Agent final output (model → validator)

```json
{
  "customer_id": "C102",
  "risk_factors": {
    "price_sensitivity": 0.80,
    "disengagement": 0.70,
    "loyalty": 0.30
  },
  "evidence_refs": ["history:C102:usage-2026-02", "ticket:C102:T04"]
}
```

All three factors required, finite, `[0,1]`; no extra keys unless versioned. Evidence references must be a subset of evidence IDs **actually returned** by authorized tool calls for this request. Empty evidence is invalid for normal predictions. Preserve malformed raw outputs and structured errors; never silently repair/renormalize a model failure for scoring.

### 4.5 Fixed conversion (independent from Oracle)

- Define one public `risk_to_outcomes(risk_factors, intervention, plan)` function before final evaluation.
- Use a **distinct, documented** coefficient set from the private Oracle, with softmax over outcome scores and downgrade masking for Basic.
- A workable initial converter (to be frozen and unit-tested) uses:
  - `price_effect = price_sensitivity * price_change * 10`
  - `score_stay = 1.0 + 1.2*loyalty - 0.9*disengagement - 0.8*price_effect`
  - `score_downgrade = -0.5 + 0.5*disengagement + 0.6*price_effect` (mask for Basic)
  - `score_churn = -0.8 + 1.0*disengagement - 0.7*loyalty + 0.9*price_effect`
- Treat these as **implementation defaults**, not Oracle truth; adjust only using train/validation data, never held-out test labels. Record the final coefficients and version. A fixed converter constrains the LLM's possible output distributions; report this limitation.
- Keep the conversion and risk-factor schemas identical for base and fine-tuned agents.

### 4.6 Prediction and trace artifacts

`artifacts/predictions/{run_id}.jsonl`: one row per attempted customer × scenario × predictor with `request_id`, `customer_id`, `scenario_id`, `model_id`, `status` (`valid`/`invalid`/`error`), `risk_factors` (LLMs only), `probabilities` (when valid), `evidence_refs`, `error_code`, `latency_ms`, `run_id`, `schema_version`. Logistic regression outputs probabilities directly and has no tool traces. Oracle labels live in private evaluation artifacts, not in agent-visible prediction files.

`artifacts/traces/{run_id}.jsonl`: append-only structured events (`task_start`, `model_call`, `tool_request`, `tool_result`, `final_output`, `validation`, `error`), each with event sequence, timestamp, request ID, model/version, sanitized prompts/responses, tool arguments, evidence IDs, token counts where available. Use restricted file permissions for raw logs and redact secrets. Dashboard reads sanitized exports.

**Validation:** Contract tests parse all JSONL rows; every request has exactly one terminal status, no duplicate IDs, no missing failed attempts, and trace events can be joined to predictions by `request_id`.

## 5. Component-by-component implementation

### C1. Synthetic dataset — Person A

1. Define `customers`, `plans`, `usage_history`, `billing_history`, `support_tickets`, and observable outcome-history tables in SQLite. All event rows have `observed_at`.
2. Generate 150 customer profiles across three plans and realistic segments; six months of historical records before cutoff; support tickets with neutral, positive, and pricing-related language.
3. Generate private latent traits separately (price sensitivity, loyalty, etc.) using seeded RNG; correlate observable history with traits without making them identical.
4. Produce **customer-disjoint** 100 train / 20 validation / 30 test splits. No customer history from test enters SFT.
5. Build referential-integrity, time-order, split-disjointness, and reproducibility tests.

**Acceptance:** 150 unique customers; six months per customer; valid plan/segment distribution; every historical record precedes cutoff; split counts exact; rerun with same seed yields same checksums.

### C2. Private Oracle — Person A

1. Define documented latent traits and interpretable outcome scores for stay/downgrade/churn.
2. Add **one nonlinear interaction**, e.g. `price_change × price_sensitivity²`, so highly sensitive customers react disproportionately to price increases.
3. Apply masked softmax (Basic cannot downgrade); verify probabilities are finite, nonnegative, and sum to one.
4. Generate private truth for both worlds and both training/validation/test customers. Train-only approved targets may be used for supervision, but never expose test truth to the agent.
5. Validate that the Oracle responds plausibly on average to price increases; do not hard-code universal monotonicity for every possible customer if other effects differ.
6. Freeze DGP version, coefficients, seed, and a private evaluation checksum.

**Acceptance:** Repeatable reference probabilities; Basic downgrade zero; no Oracle imports/paths accessible to agent; expected revenue calculations consistent with plans.

### C3. Restricted database tools — Person B (Person A supplies schema/tests)

1. Implement the two tools with typed input/output validation.
2. Bind authorized customer ID and `as_of` in backend context; reject model attempts to change scope.
3. Filter raw records and aggregate statistics at SQL query level before returning anything.
4. Return stable evidence IDs and structured errors.
5. Log every attempted tool call; cap total attempts at two; prohibit arbitrary SQL.
6. Run tests with injected future records, wrong customer IDs, and malformed arguments.

**Acceptance:** Correct data, zero future exposure, zero hidden-trait exposure, deterministic errors, complete tool traces.

### C4. Persona Agent — Person B

1. Test Qwen2.5-3B-Instruct load/inference in selected runtime; fix tokenizer, chat template, structured tool format, and model revision.
2. Implement task → tool choice → tool result → optional second tool → risk-factor JSON.
3. Validate all outputs and evidence refs; emit explicit invalid/error statuses rather than filling defaults.
4. Freeze prompt version and decoding settings for paired base/SFT comparison (prefer deterministic decoding when supported).
5. Save complete, replayable traces for 5–10 smoke tasks before training.

**Acceptance:** Agent completes end-to-end smoke tasks; obeys two-call limit; invalid outputs are preserved; no direct DB/Oracle access.

### C5. Single-stage SFT — Person B

1. Run a **tiny GPU smoke test on Day 1**: model load, 2–5 training examples, adapter save, reload, and inference. Verify whether HF credits apply to the intended GPU product.
2. Generate 200–300 high-quality, train-customer-only demonstrations: history-only, history+segment, evidence interpretation, and safe failures. Each demonstrates tool selection, valid arguments, actual tool outputs, and evidence-grounded risk factors.
3. **Supervision policy:** Never copy private latent Oracle traits into model targets. Use transparent evidence-based teacher estimates or train-split-only targets explicitly constructed without disclosing latent values. If targets derive from train Oracle *outcome probabilities*, document this and use a separate fitting process for public risk-factor estimates; do not imply latent trait recovery.
4. Validate demonstrations for schema, permitted evidence, chronological filtering, customer split, and no hidden-truth strings. Deduplicate near-identical examples; avoid template leakage across splits.
5. Train **one** LoRA/QLoRA adapter with fixed config, capped steps, and a validation checkpoint; save adapter, tokenizer info, train logs, seed, and base revision.
6. Run identical held-out requests for base and SFT; export predictions and traces. Do not retune on test outcomes.

**Acceptance:** Adapter reloads; reproducible config; all training rows validated; base and SFT evaluation complete (including failures).

### C6. Paired simulator — Person A

1. Implement `world_a = 0.00` and `world_b = 0.10` as the only formal scenarios.
2. For each customer and predictor, accept a three-outcome probability vector; never request fresh inference from the dashboard.
3. Calculate expected churn, downgrade, retention, and revenue for each world. For a Pro → Basic downgrade, charge Basic price in that world; Enterprise → Pro likewise. Basic downgrade is disallowed.
4. Compute world-difference metrics and Oracle-relative errors.
5. Optional: sample discrete customer outcomes using a saved random seed and **common per-customer uniform draws** across worlds to reduce comparison noise. Keep expected metrics primary.
6. Separate full-150 demonstration outputs from held-out-30 benchmark outputs.

**Acceptance:** Probability-weighted revenue matches hand calculations; worlds share initial state; deterministic expected metrics; seeded sampling repeats exactly.

### C7. Safety and robustness — Person A designs, Person B executes

1. Create 10 malicious support-ticket variants, each paired with a benign control preserving the same legitimate customer facts.
2. Specify attack goal and success rule for each pair **before** evaluation (e.g. force invalid tool call, induce a specified incorrect risk score, or fabricate evidence).
3. Run 20 cases for each model (40 agent attempts), with same customer/scenario and environment where applicable.
4. Compare attack success, valid task completion, evidence validity, and prediction shift versus matched control.
5. Run independent backend tests for future-record exclusion, aggregate time filtering, unauthorized scope, and evidence IDs.
6. Clearly label backend-blocked attacks versus model-level resistance; report denominators and small-sample uncertainty.

**Acceptance:** Frozen case list, identical conditions, complete traces, zero exposed future records, no claims of broad robustness.

### C8. ParallelBench and statistical baseline — Person A scoring, Person B inference

1. Freeze 30 held-out customer IDs and both scenarios; 60 task instances per LLM; 120 total LLM attempts.
2. Person B runs base and SFT with identical tools, limits, prompt family, decoding, and converter.
3. Person A trains multinomial logistic regression on **observable features and train-only labels**, with preprocessing fitted only on training data. For scenario dependence, include price change as a feature. Output calibrated-looking probabilities only if measured; do not assume calibration.
4. Compute tool-call validity, tool selection accuracy (according to predeclared rubric), task completion, evidence validity, average calls, unnecessary calls.
5. Compute **Oracle probability MSE** (mean over stay/downgrade/churn squared differences), churn MAE, downgrade MAE, expected revenue error, and intervention-effect error. Do not call Oracle-probability MSE an observed-outcome Brier score.
6. Report invalid attempt rate for all methods. Fidelity metrics are conditional on valid predictions **with coverage shown alongside**; optionally add a prespecified failure-penalized sensitivity analysis.
7. Pair comparisons by customer; two scenarios are repeated measures, not 60 independent customers. Show per-customer differences, distribution summaries, and exploratory bootstrap intervals if time permits.
8. Preserve negative findings and segment-level limitations.

**Acceptance:** Every benchmark task accounted for, no test-label leakage, exact denominator reconciliation, reproducible metrics and figures.

### C9. Streamlit dashboard — Person A

1. Build UI with mocked artifacts first; read only JSONL/CSV/metrics and sanitized traces, never live model or private Oracle files.
2. **Parallel Worlds:** World A/B expected metrics; predictor selector; clear full-population vs test-set labels; optional seeded sampled outcomes.
3. **Agent Inspector:** customer/scenario/model selector; chronological tool-call replay, evidence IDs, risk factors, converted probabilities, errors.
4. **ParallelBench:** base/SFT/logistic fidelity charts, tool reliability charts for LLMs, paired intervention-effect comparison, sample sizes and invalid rates.
5. **Safety Lab:** select frozen attack pair, compare base/SFT traces, show model vs backend defense, leakage/evidence tests.
6. Label placeholders `ILLUSTRATIVE` until replaced. Fail loudly on missing artifacts; never present demo numbers as real results.

**Acceptance:** Four tabs load from cached real artifacts offline; all charts reconcile with metric files; a 2-minute demo works without GPU.

### C10. Integration and submission — Both

1. Freeze contracts and dataset/split versions; add CI for unit and contract tests.
2. Merge at least daily; use mock fixtures and smoke tests at every interface boundary.
3. Create a run manifest with Git commit, model revision, adapter hash, dataset checksum, split checksum, converter version, prompts, seed, decoding, tool budget, and environment.
4. Generate a clean README with setup, data generation, training (optional), benchmark replay, and dashboard commands.
5. Prepare research report: problem, synthetic-world assumptions, methodology, training, results, safety, limitations, reproducibility, and explicit negative findings.
6. Record a short demo/video if required by event rules; verify actual submission requirements and deadlines from official guidelines before packaging.
7. Dry-run setup from a clean environment and test cached dashboard with no GPU/API keys.

**Acceptance:** Reproducible results and demo, no secrets, working tests, correct artifact paths, both teammates can explain system end-to-end.

## 6. Six-day schedule with handoff gates

| Day | Person A | Person B | Non-negotiable end-of-day checkpoint |
|---|---|---|---|
| **1 — Freeze & feasibility** | Generate 150 customers, splits, observable DB, contracts; draft Oracle | Tool schema review; model/GPU and 2–5-example LoRA smoke test | Dataset checks pass; contracts frozen; GPU decision made |
| **2 — Independent vertical slices** | Finish private Oracle, public converter, simulator with mocks | Restricted tools, base agent, 5–10 real tool trajectories | Both pipelines run independently; shared fixture passes |
| **3 — Train & prepare evaluation** | ParallelBench scorer, logistic baseline, frozen safety fixtures | Validate 200–300 demonstrations; run one SFT; save adapter | Adapter reloads; scoring/safety tests ready |
| **4 — First end-to-end result** | Integrate predictions; compute first results; start dashboard | Run 60 base + 60 SFT held-out attempts; export traces | **Complete paired base/SFT benchmark exists** |
| **5 — Safety & demo** | Run safety analysis; finish four tabs; charts and failure analysis | Run 40 adversarial attempts; verify logs; package model | All real results and safety cases visible in cached dashboard |
| **6 — Submission** | Write report, slides/demo script, limitations, verify figures | Reproduction commands, adapter/config packaging, final QA | Clean-run checklist passed; submit required materials |

## 7. Test matrix and explicit pass criteria

| Test | Minimum assertion |
|---|---|
| Dataset | Exactly 150 IDs; 100/20/30 disjoint; six months history; seeded rerun identical |
| Oracle | Every distribution valid; Basic downgrade = 0; same seed produces same truth |
| Converter | Finite bounded inputs; valid masked softmax; does not import Oracle |
| Tools | Only authorized customer; `observed_at < as_of`; no hidden traits; max 2 attempts |
| Agent | Correct JSON schema; valid evidence refs; explicit invalid/error status |
| SFT data | 200–300 valid rows or documented fallback; zero test IDs; zero hidden-trait targets |
| Traces | Each attempted request has terminal status and joinable events |
| Simulator | Hand-calculated expected revenue; correct plan transitions; paired reproducible sampling |
| Benchmark | 120 LLM attempts accounted for; logistic predictions separately accounted; no missing failures |
| Safety | 10 attack/10 benign pairs × 2 models; no future records exposed; backend/model defenses distinguished |
| Dashboard | Four tabs load cached artifacts offline; all numbers derived from saved results |
| Submission | Clean setup works; configs and manifests recorded; report limitations stated |

## 8. Budget and execution controls

**Budget ceiling:** $20 credits, if eligible for the intended HF GPU service. Plan **up to $12** for setup and one training run, **reserve $8** for inference/evaluation or unexpected GPU time. Do not assume credit portability between HF services; confirm product eligibility and live hourly price before starting.

- Day 1: smoke test only; record time and billed amount; stop if the expected training run exceeds the remaining budget.
- Day 3: one primary SFT run, no hyperparameter sweep. Save adapter and logs immediately.
- Day 4: evaluate 60 scenarios × 2 model versions, using batched/cached inference where feasible.
- Day 5: evaluate 20 safety cases × 2 models. Avoid repeated runs unless needed for a documented failure.
- The 16 GB MacBook handles generation, SQLite, Oracle, converter, logistic regression, evaluation, Streamlit, and tests; GPU training occurs remotely.

## 9. Failure modes and fallback ladder

| Trigger | Immediate action | What remains scientifically valid |
|---|---|---|
| HF credits do not cover GPU Jobs | Check permitted compute product; use an available compatible GPU or documented alternate provider; stay within actual spend | Same protocol and comparisons if adapter training succeeds |
| 3B training fails / exceeds budget | Switch to a smaller instruction model (e.g. 1.5–1.7B), rerun base and tuned under the **same** model family | Within-model before/after comparison; document changed model |
| 200–300 demos take too long | Freeze 100–150 validated examples, prioritize diversity and correctness | Smaller SFT pilot; report sample count |
| Adapter training fails completely | Deliver base-agent + logistic + Oracle study, clearly mark SFT hypothesis **untested** | Infrastructure/simulation evaluation only, not a fine-tuning result |
| Model emits malformed risk factors | Preserve failures, measure validity, debug only on train/validation | Honest completion and fidelity coverage |
| Model predicts nearly constant risks | Report distributional failure; compare logistic baseline; do not tune on test | Negative finding with strong baseline context |
| Oracle/converter too similar | Audit coefficient independence; freeze converter on train/validation only | Valid synthetic fidelity if no test truth used |
| Integration blocked | Use contract fixtures and mock predictions; prioritize export/import tests | End-to-end simulator and dashboard can still work |
| GPU unavailable during demo | Use cached real traces and predictions | Fully functional offline demonstration |
| Dashboard delayed | Produce static figures and minimal four-tab layout; avoid polishing ahead of results | Reproducible experiment and submission |
| Safety attack succeeds | Report failure, inspect traces; do not silently change frozen test set | Valid safety observation |
| Time critically short | Prioritize: contracts → Oracle/tools → one SFT → paired benchmark → safety → dashboard → report | Most defensible minimum result |

## 10. Research analysis and presentation plan

**Primary results:** (1) tool-call validity and task completion; (2) Oracle probability MSE and churn MAE with valid-output coverage; (3) paired intervention-effect error; (4) prompt-injection success and task completion under attack.

**Secondary results:** logistic-regression comparison, segment errors, revenue error, tool-call efficiency, evidence integrity, latency, qualitative failure cases.

**Figures to generate:** architecture/trust boundary; paired world expected revenue/churn; base vs SFT tool reliability; Oracle MSE across three predictors; per-customer intervention-effect error; attack/control safety comparison; example agent trace.

**Interpretation rules:**
- Never imply the Oracle describes actual SaaS markets; all behavioural truth is planted by our DGP.
- Do not call 60 scenario rows 60 independent customers.
- Do not claim model robustness from backend rejection of a malicious request.
- Do not conflate predicted risk factors with recovered true latent traits.
- Do not exclude invalid model outputs from reliability denominators.
- Explicitly state small test size, synthetic domain, limited interventions, and the converter's expressiveness constraint.
- Report actual results, including no improvement or regressions.

## 11. Submission and clean-room checklist

- [ ] Official hackathon submission format, deadlines, and mandatory tool requirements verified.
- [ ] README explains goal, system architecture, trust boundaries, and setup.
- [ ] Code, adapter or access instructions, pinned dependencies, and reproducible commands provided.
- [ ] Frozen dataset/split and manifest checksums recorded.
- [ ] Oracle hidden traits/test labels excluded from model-facing runtime and training artifacts.
- [ ] Base/SFT benchmark attempts, failures, and traces exported.
- [ ] Logistic-regression baseline and safety suite reported separately.
- [ ] All dashboard tabs run offline using cached **real** results.
- [ ] Figures and tables regenerate from artifact files.
- [ ] No API keys, personal data, or private secrets committed.
- [ ] Both teammates can deliver a two-minute walkthrough and answer methodological questions.

## 12. First actions after approving this plan

**Person A, first 90 minutes:** create repository and contract fixtures; implement observable schema and seeded 150-customer generator; freeze customer splits; create mock Oracle and mock prediction files so simulation/dashboard development can begin.

**Person B, first 90 minutes:** review contracts; implement stub tool API against mock SQLite; verify Qwen2.5-3B-Instruct inference and a 2–5-example LoRA/QLoRA save/reload smoke test on the intended compute service.

**Joint handoff before Day 1 ends:** exchange one real `get_customer_history` result, one valid agent output, one invalid output, and one complete trace; run `pytest tests/test_contracts.py`; sign off schema version `1.0`.

