# Real-Agent Trace Verification with GRPO (OA + Post-Training Lab)

**Goal:** Train a small LLM with TRL/GRPO to audit *real* coding-agent trajectories: decide whether the bug was resolved, and point to grounded evidence of failure.

This is a practice project, **not a verified Mercor OA**. It trains a *trajectory-verification agent*, not a code-fixing agent. GPU training has **not** been run in this package.

## Data provenance
- Task dataset: `SWE-Gym/SWE-Gym-Lite` or `SWE-Gym/SWE-Gym` — real GitHub software engineering issues.
- Recorded rollout dataset: `SWE-Gym/OpenHands-Sampled-Trajectories` — real generated agent messages, tools, outcomes.
- Source: https://huggingface.co/SWE-Gym
- Source code: https://github.com/SWE-Gym/SWE-Gym

The `resolved` and `test_result` fields are *private grading labels*, never model-visible prompt text. Do not treat arbitrary agent self-reports of success as ground truth.

## Quick start

```bash
python -m unittest discover -s tests -v   # red tests first, finish trace_lab.py
pip install -r requirements.txt
python prepare_data.py --max-rows 1200 --train-issues 160 --eval-issues 40
python train_grpo.py --max-steps 60
python evaluate.py --predictions predictions.jsonl  # when you have recorded inference predictions
```

`prepare_data.py` requires Internet access to fetch the official dataset; GRPO training requires a compatible GPU. Pin/check TRL and Transformers versions before a paid run; the example uses the current GRPO `reward_funcs` API. Avoid `use_vllm` initially to keep the run simpler.

## Timing

### Phase A: OA-style Python (90 minutes; no GPU)
1. **0–20 min**: implement `normalize_events(trace)` in `trace_lab.py`: preserve message sequence, parse assistant tool calls, match `tool_call_id` results, parse stringified JSON arguments.
2. **20–45 min**: implement `audit_events(events)` with earliest observed failure, missing/orphan result, invalid JSON, repeated failing call with unchanged state.
3. **45–60 min**: implement `select_safe_redundancies(events)`; propose *candidates* for deletion and explain why deleting events requires replay.
4. **60–75 min**: implement `group_advantages(reward_groups)`; use population standard deviation and return zeros for constant groups.
5. **75–90 min**: write `diagnosis.md`: failure taxonomy, suspected reward hack, counterexample, dataset split and 2 ablations.

Run unit tests after each phase. Fixtures in unit tests are intentionally tiny; the training and audit dataset is real.

### Phase B: Real data audit (1–2 h; CPU)
- Load ~1,200 real trajectories with `prepare_data.py` (change size according to RAM).
- Keep all runs of the same `instance_id` together in **one split**.
- Report resolved-rate, empty-patch rate, error-eval rate, run length distribution and tool error examples.
- Create an evaluation table including at least 15 manually inspected traces and ambiguous cases.
- Inspect both successful and failed samples. Handle missing `test_result` and unclear failure attribution.

### Phase C: TRL GRPO training (GPU)
- Model: `Qwen/Qwen3-0.6B`.
- Task: predict strict JSON `{ "solved": boolean, "failure_type": string, "evidence_step": integer | null }` from a capped, model-visible view of real trajectory messages.
- Group size `num_generations=4`; temperature 0.8–1.0; ~60-150 steps as a learning experiment.
- Baseline reward: outcome agreement + schema reward.
- Proposed improved reward: add evidence-grounding reward based on explicit observable trace faults, **never** arbitrary private reasoning.
- Try reward ablation (outcome only vs outcome + evidence), keeping same model, data split, budget and seed.
- Capture raw generations and group reward variance. Do not assert that training improves without held-out comparison.

### Phase D: Stateful tool-use extension (optional)
- Replace the static prompt with a stateful `TraceReplayEnv` exposing `inspect_chunk(start, count)` and `submit_verdict(...)`.
- `reset()` selects one trace; only sanitized event chunks go to the LLM; `get_reward()` uses *private* grader labels.
- Use TRL `GRPOTrainer(environment_factory=TraceReplayEnv)` after verifying the installed TRL docs. Set a tool-turn limit and record tool costs. This is the agentic environment portion, not required for the baseline.

## Main scientific question
**Does adding verifiable, observation-grounded evidence rewards reduce false-positive success judgments on unseen *issues*, compared with outcome-only GRPO?**

Primary metrics:
- Balanced accuracy and false-positive success rate on held-out `instance_id`s.
- Evidence validity: referenced event index exists and actually supports the predicted failure category.
- Group reward standard deviation / fraction of zero-advantage groups.
- Model output validity, completion length, tool steps and wall-clock time.

Ablations:
1. Outcome reward only vs outcome + evidence reward.
2. Full (capped) visible trajectory vs last N messages.

Controls:
- Keep splits **grouped by `instance_id`**; NEVER randomize by rollout. Avoid label leakage from `resolved`, `test_result`, and verifier patches.
- Evaluate positive and negative labels separately; 'always failed' can look accurate in imbalanced data.
- `test_result.report` gives explicit failure signals but does not reveal a trustworthy first divergence point for every trace.
- A failed test identifies an outcome, not the cause of failure. Reasoning text alone is not evidence of cheating.
- Reward hacking remains possible: track independent outcome metrics and inspect trajectories.

## Why this is well aligned to the OA
The same data exercises nested JSON parsing, event pairing, failure taxonomy, filtering, deterministic reward design, GRPO advantages, dataset leakage prevention and research-style evaluation. Unlike isolated MBPP examples, the tasks originate in real Python repositories. Training a full SWE-bench code-fixing agent would require substantially more environment engineering and compute.

## Files
- `trace_lab.py` — TODOs for OA-style coding.
- `tests/test_trace_lab.py` — offline tests against short fixtures.
- `prepare_data.py` — dataset loading, sanitized prompts and issue-wise split.
- `interfaces.py` — public trace types, adapter and ID parsing functions.
- `trace_schema.py` — shared trace, message and tool-call types and adapter protocol.
- `trace_analysis.py` — ID summaries, task joins, message counts, and exploration plots.
- `swe_gym_trace.py` — SWE-Gym adapter and ID parsing.
- `../01_agent_trace_exp.ipynb` — dataset overview and step-by-step inspection of one trace.
- `train_grpo.py` — basic `GRPOTrainer` run with an executable outcome/schema reward; adapt the reward to your research question.
- `evaluate.py` — evaluation metrics from prediction JSONL.

For inspection, import `SWEGymTraceAdapter` from `interfaces` and call
`SWEGymTraceAdapter().parse(record)`. The result has `messages`, `metadata`,
and separate `labels`. To support another dataset, implement `TraceAdapter.parse`
and return an `AgentTrace`; its source fields do not need to match SWE-Gym.

**Security:** Do not execute commands from recorded agent traces locally. Replay-based auditing only reads data. For real code-execution rewards, use isolated containers with no credentials, restricted network and resource/time limits.

## Conversational SFT preparation

`../02_SFT_prepare.ipynb` loads the raw SWE-Gym trajectories and calls the public
`normalize_trace` and `write_sft_dataset` interfaces. The existing `TraceAdapter`
protocol allows another dataset to supply the same shared `AgentTrace` format.
Every attempt is written to JSONL; a separate issue file records malformed calls,
unmatched tool responses, missing results, and incomplete trajectories by source
row. The summary counts valid and invalid records and message roles. Output paths
must not already exist. The redaction patterns cover common sensitive strings but
are not exhaustive; review generated data before use. Chat-template support for
tool IDs and `datasets` JSON types depends on the selected versions and model.
