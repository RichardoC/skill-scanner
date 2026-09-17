# Plan: parallel constrained decoding ("RLCD") for the scanner's categorical LLM stages

Status: **proposal, not implemented.** No code in this change.

Reference: [`harshatheg/Qwen-2.5-1B-RLCD`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
(Apache-2.0, created 2026-09-16).

---

## 1. What the reference actually is

The Hugging Face repo is worth reading carefully before planning against it,
because three things about it differ from what the name suggests.

**It contains no weights.** The repo holds source code published under a model
repo: `core/engine.py`, `core/engine_mlx.py`, `core/engine_torch.py`,
`core/schema.py`, `core/prompt_builder.py`, a FastAPI demo server, and four
JSON presets. The engine loads `Qwen/Qwen2.5-1.5B-Instruct` (torch) or
`mlx-community/Qwen2.5-1.5B-Instruct-4bit` (MLX) from elsewhere. There is no
fine-tune here — "Qwen-2.5-1B-RLCD" names the decoding engine, not a model.

**"RLCD" is never expanded in the card**, and it collides with the established
meaning in the literature (RL from Contrast Distillation). Internally we should
not use the acronym; this plan calls the technique **parallel constrained
choice** and proposes `parallel_choice` as the module/flag name.

**It is not Apple-only.** The card foregrounds MLX and Apple Silicon, but
`core/engine.py` is a router: MLX on Darwin, otherwise `engine_torch.py`
(CUDA/CPU via `transformers`). That makes it viable for the Linux CI paths that
matter most to this project.

### The mechanism, in one paragraph

For a schema whose every field is a boolean or a bounded enum (≤255 choices),
the engine prefills the context prompt **once**, broadcasts the resulting KV
cache to batch size *M* (one row per field), runs **one** batched forward pass
over per-field suffixes, and then, per field, slices the logits down to the
first token of each candidate choice and takes the argmax plus a softmax over
that slice. The JSON is assembled programmatically from the winning values, so
syntax validity is structural rather than learned. Decode cost collapses from
*O*(output tokens) to *O*(1) forward passes; field-level calibrated
probabilities fall out for free.

The card's headline numbers — 5.6x–7.0x latency, 100% schema validity — are the
author's, on an M4 Max with a 4-bit 1.5B model, with short contexts. "100%
schema validity" is a property of programmatic assembly, not evidence that the
answers are right; the card reports no accuracy comparison against the
autoregressive baseline beyond schema match.

---

## 2. What this can and cannot accelerate here

The technique only applies where **the entire output is a fixed set of
categorical fields**. It cannot emit a title, a description, an evidence
snippet, a remediation paragraph, or an evidence ID copied from the prompt.

That rules out the scanner's largest LLM consumer outright. The main LLM
analyzer (`llm_analyzer.py` → `llm_request_handler.py`) produces findings
against `data/prompts/llm_response_schema.json`, which is mostly free text plus
`evidence_ids` matched against `^(SRC|DET):[a-f0-9]{16}$`. Parallel constrained
choice cannot generate any of that, and no amount of schema work changes it.

It also does nothing for the deterministic half of the scanner. Static
analysis, YARA-X, the dataflow/taint engine, `correlation_analyzer.py`, and the
CEL decision layer issue no LLM calls; they are not "suitable parts" under any
reading.

One more framing point that must not get lost: the scanner's default judge
today is a **hosted** provider (Bedrock / OpenAI / Gemini / Vertex / Azure, via
LiteLLM), with Ollama as the local option. Parallel constrained choice requires
direct logit access, so it is inherently a **local-model** technique. Adopting
it is therefore not "the same judge, decoded faster" — it is "replace a hosted
frontier judge with a local 1.5B classifier on this stage". That is an accuracy
change as much as a latency change, and the plan is built around proving the
accuracy side before taking the latency win.

---

## 3. Candidate call sites, ranked

There are exactly four LLM call sites in the tree:

| Call site | Output shape | Suitability |
|---|---|---|
| `analyzers/adjudicator.py` | `{verdict, confidence, reason}` | **High** |
| `analyzers/behavioral/alignment/threat_vulnerability_classifier.py` | `{classification, confidence, reasoning, key_indicators}` | **High** |
| `analyzers/behavioral/alignment/alignment_llm_client.py` | 4 categorical + 5 prose fields | **Medium (gate only)** |
| `analyzers/meta_analyzer.py` | 7 top-level keys, mostly prose | **Low (partial)** |
| `analyzers/llm_request_handler.py` | full finding generation | **None (pre-gate only)** |

### 3.1 Adjudicator — start here

`adjudicator.py` asks, per deterministic HIGH/CRITICAL finding, whether the
rule's regex fired on the real threat or on benign context, and returns
`{"verdict": "real" | "false_positive", "confidence": 1-5, "reason": "<one
sentence>"}`. Strip `reason` and that is a two-field categorical schema — the
exact shape the engine is built for.

It is also the safest place to start, for reasons already true of the code:

- The pass is **off by default** (`--adjudicate`, `AdjudicatorPolicy.enabled`).
- It is **demote-only** by construction, and that property is load-bearing and
  documented. A weaker model can therefore not raise severity.
- Every error path (unavailable, malformed, timeout, out-of-range confidence)
  already leaves the finding untouched, so an engine failure degrades to
  today's un-adjudicated behaviour.
- `metadata['adjudication']` already preserves an audit record.

Two structural wins beyond raw decode speed:

- **The module lock disappears.** `_LLM_LOCK` exists only because concurrent
  scanner + meta + adjudicator traffic hit Bedrock throttle limits. A local
  engine has no shared quota, so adjudication can run concurrently with the
  scanner's own LLM stage instead of serialising behind it.
- **One prefill, many findings.** Several findings routinely land in the same
  file. Prefill that file's context once and score every finding's fields in a
  single batched pass. This is a bigger win than the card's 5.6x, and it is
  only available because the technique exposes the prefill/suffix split.

`reason` becomes a template rendered from field telemetry (`"demoted:
false_positive @ confidence 4/5 (p=0.97)"`) rather than generated prose. The
audit record gains calibrated probabilities it does not have today, which is
arguably better evidence than a model-written sentence.

### 3.2 Threat/vulnerability classifier

`threat_vulnerability_classifier.py` returns `classification ∈ {THREAT,
VULNERABILITY, UNCLEAR}` and a confidence, plus `reasoning` and
`key_indicators` prose. The decision is purely categorical and it runs
**per alignment finding**, so it fans out. Same treatment: decode the two enums,
template the prose, keep the existing validation (`valid_classifications`) as a
belt-and-braces check that now cannot fail.

### 3.3 Alignment check — as a gate, not a replacement

`alignment_prompt_builder.py` asks for `mismatch_detected` (bool),
`threat_name` (enum), `severity` (enum), `confidence` (enum) — all decodable —
alongside `summary`, `description_claims`, `actual_behavior`,
`security_implications`, `dataflow_evidence`, which are not.

This is where the **largest end-to-end win** sits, and it is architectural
rather than decode-level. Alignment runs one LLM call per candidate function
(`alignment_orchestrator.check_alignment`), and on real packages the
overwhelming majority of functions are aligned. Split it in two:

1. **Gate** (parallel constrained choice, ~1 forward pass): decode
   `mismatch_detected` + `confidence`.
2. **Tail** (existing provider, unchanged): only when the gate says mismatch
   above a configured probability floor, issue today's full call to produce the
   prose fields and the finding.

A scan whose functions are 95% aligned then pays the expensive call 5% of the
time. The gate must be tuned for **recall, not accuracy** — the cost of a false
"aligned" is a missed finding, so the threshold should be set so the gate
escalates on anything ambiguous, and the shadow evaluation in §6 should measure
gate recall against the full-LLM answer specifically.

### 3.4 Meta-analyzer — partial, later

The meta-analyzer's seven keys are mostly prose, but the load-bearing part —
classifying each supplied `_index` exactly once as validated or false positive —
is a vector of one enum per finding index. That maps precisely onto the card's
28-field "support triage" case: *M* fields, one forward pass. The prose keys
(`correlations`, `recommendations`, `missed_threats`, `verdict_reasoning`)
still need the provider. Worth doing only after §3.1–3.3 land and the
evaluation harness is trusted; the hybrid response assembly is fiddly and the
meta-analyzer's own accuracy validation is already noted as pending.

### 3.5 LLM analyzer pre-gate — stretch, recall-critical

A per-file triage ("does this file warrant a full LLM pass?") would cut the
scanner's dominant cost, but a wrong "no" is a silent miss in a security tool.
If attempted at all, it must be **escalate-only relative to a conservative
floor**: allowed to *add* files to the queue freely, allowed to skip only files
below a probability threshold proven on the source-disjoint split, and shipped
in shadow for at least one release. Lower priority than everything above.

---

## 4. Architecture: one seam, two implementations

Add `skill_scanner/core/analyzers/parallel_choice/` with a deliberately narrow
interface so no security-critical caller learns about torch, MLX, or logits:

```
ChoiceSchema      # fields: name -> (bool | enum[choices], description)
ChoiceResult      # field -> (value, probability); plus prefill/eval timings
ChoiceEngine      # classify(context: str, schema: ChoiceSchema) -> ChoiceResult
                  # classify_batch(contexts, schema) for the shared-prefill case
```

Two implementations behind it:

- `LocalParallelChoiceEngine` — the technique above. Backend router mirroring
  `core/engine.py`: MLX on darwin/arm64 when available, torch otherwise, and a
  clean `unavailable` state when neither is installed.
- `ProviderChoiceEngine` — today's LiteLLM path, asked for the same schema as
  JSON and parsed. Needed for three things: parity reference during shadow
  evaluation, fallback when the local engine is unavailable, and keeping the
  call sites honest about not depending on local-only behaviour.

Every caller takes a `ChoiceEngine` and keeps its existing failure semantics.
`LocalParallelChoiceEngine` returning `None` must be indistinguishable, to the
caller, from today's "LLM unavailable" path.

### Two constraints the seam must enforce

**Enum labels must be first-token distinct.** `schema.py` maps each choice to
`tokenizer.encode(choice)[0]` — the *first* token only. The MLX engine claims
token-tree disambiguation for shared prefixes; `engine_torch.py` does not
implement it. Two choices sharing a first token therefore collide silently and
degrade accuracy with no error. This bites immediately on the existing
taxonomy: the `aitech` enum (`AITech-1.1`, `AITech-1.2`, `AITech-4.3`, …) shares
a first token across *every* value and is undecodable as written. Mitigations:

- Derive `aitech` deterministically from `category` via a lookup table rather
  than decoding it. The schema's own description already defines that mapping.
- Add a `ChoiceSchema` validator that compiles candidate tokens at construction
  and **raises** on a first-token collision, plus a unit test over every enum we
  ship. Fail loudly at startup, never silently at inference.

**Prefill, not decode, will dominate our prompts.** The card's contexts are
short incident reports; ours are skill files and finding context running to many
kilobytes. The technique removes decode cost only. Expected speedup therefore
scales with *output* length: large for the adjudicator (~250 output tokens
today), negligible anywhere the model writes prose. `engine_torch.py` also does
`copy.deepcopy(base_cache)` per call, whose cost grows with prefix length — for
our prompt sizes this needs measuring and probably replacing with an
expand-in-place broadcast.

---

## 5. Phases

**Phase 0 — measure before optimising.** Extend the existing `LLMTokenUsage` /
`_extract_token_usage` plumbing to record per-stage wall-clock and output-token
counts (analyzer, meta, adjudicator, alignment, classifier), and emit them in
the scan report under a debug key. Run it over `evals/skills` and the official
bundled-skills set. **Gate:** if the adjudicator and alignment stages turn out
not to be material, most of this plan is not worth doing, and Phase 0 is the
cheap way to find that out.

**Phase 1 — the seam.** `parallel_choice/` package, `ChoiceSchema` with
first-token collision validation, `ProviderChoiceEngine`, and
`LocalParallelChoiceEngine` with both backends. No call site changes. Unit
tests with a stub engine; integration tests marked and skipped when torch is
absent.

**Phase 2 — adjudicator.** Convert `_call_llm` to go through a `ChoiceEngine`.
Add shared-prefill batching for findings in the same file. Remove `_LLM_LOCK`
on the local-engine path only (it must stay for the hosted path). Preserve the
demote-only property and every error path exactly.

**Phase 3 — classifier and alignment gate.** Convert
`threat_vulnerability_classifier.py`. Split `check_alignment` into gate + tail,
with the escalation threshold in `ScanPolicy`.

**Phase 4 — evaluation and rollout** (§6). This is the long pole, not Phases
2–3.

**Phase 5 (optional) — meta-analyzer per-index vector, LLM-analyzer pre-gate.**
Only on the evidence from Phase 4.

---

## 6. Correctness gating — reuse the CEL discipline

This repo already has the right pattern for shipping a risky detection change,
and this work should not invent a second one. Mirror
`docs/architecture/cel-decision-layer.md` and
`docs/development/detection-evaluation-rollout.md`:

Add `--parallel-choice-mode off | shadow | enforce`, matching `--cel-mode`, with
`ScanPolicy.parallel_choice.mode` and `off` as the default.

| Mode | Behaviour |
|---|---|
| `off` | Provider path only. Today's behaviour, byte-for-byte. |
| `shadow` | Run **both** engines. Act on the provider's answer; record the local engine's answer, its per-field probabilities, and both latencies as telemetry. |
| `enforce` | Act on the local engine's answer; fall back to the provider on unavailability or malformed state. |

Promotion from `shadow` to `enforce` requires, per converted call site:

- **Agreement**: local-vs-provider agreement rate and a confusion matrix over
  the shadow corpus, with the disagreement set reviewed by hand.
- **Detection gates**: F1, recall, and benign FPR on the MaliciousSkillBench
  development benchmark *and* the locked source-disjoint split, via
  `evals/runners/benchmark_comparison.py`. No regression in recall or FPR
  against the `off` baseline.
- **Hard negatives**: no new actionable matches on the NotInject set (currently
  0/339) and no regression on the 111-package official-bundled-skills
  compatibility set.
- **Determinism**: repeated runs identical, matching the five-run standard the
  CEL work held itself to. Greedy argmax over a sliced vocabulary should make
  this easier than the hosted baseline, not harder — verify it.
- **Latency**: the measured win, from Phase 0's instrumentation, on the same
  corpora. If agreement is high but the win is small, do not promote.

For the alignment gate specifically, add a **gate-recall** metric: of the
functions the full LLM flags as mismatched, what fraction does the gate
escalate? A gate that is 99% accurate but drops 1 in 20 real mismatches is not
shippable at any latency.

---

## 7. Packaging and supply chain

This is a security scanner. Pulling runtime code from a Hugging Face repo
created the day before, with 0 downloads and no published upstream git remote
(the card's clone URL is the placeholder
`github.com/your-org/parallel-constrained-decoding`), is not something we should
ship — and it is precisely the pattern this tool exists to flag.

**Recommendation: reimplement, do not vendor.** The load-bearing logic is small
— prefill, KV broadcast, per-field logit slice, softmax over candidates — and
reimplementing it in-tree against `transformers` gives us the first-token
collision handling, the batched-prefill path, and the deepcopy fix we need
anyway. Credit the technique and the reference implementation in the module
docstring and in this document. If vendoring is preferred instead, copy the
Apache-2.0 sources under `third_party/` with license headers, a pinned commit
SHA, and an entry in the existing supply-chain review — not a runtime download.

The model is a separate question from the code:

- Pin by **revision SHA**, never by tag or branch.
- Never auto-download during a scan. Require an explicit fetch step; respect
  `HF_HUB_OFFLINE`; fail to `off` with a clear message when weights are absent.
- Optional extra: `parallel-choice = ["torch", "transformers>=4.40",
  "accelerate"]`, with `mlx`/`mlx-lm` carrying
  `sys_platform == 'darwin' and platform_machine == 'arm64'` markers. Keep it
  out of `all`; these are heavyweight dependencies and `all` is currently
  provider SDKs only.
- Size and warm-up: 1.5B at fp16 is ~3 GB resident, 4-bit ~1 GB, with seconds
  of load time. For a one-shot CLI scan of a single skill that load cost can
  exceed everything it saves. Document it as suited to **bulk scans, the API
  server, and CI batch jobs**, where the process is warm and amortises across
  packages — not as a default for `skill-scanner scan ./one-skill`.

---

## 8. Security notes

**A genuine hardening win.** The scanner feeds untrusted skill content to the
judge, and the prompts go to some length to state that package content is inert
evidence. With constrained decoding, injected content *cannot* steer the output
beyond the candidate set: the value space is the enum, and the JSON is assembled
by us. For the categorical stages this is a stronger guarantee than any prompt
instruction, and it is worth stating as a benefit in its own right rather than
as a side effect of the speed work.

**The countervailing risk.** A 4-bit 1.5B model is a materially weaker judge
than the hosted models these stages use today. Constrained decoding makes it
*confidently* wrong rather than malformed — calibrated probabilities over a
candidate set say nothing about whether the right answer was in the set or
whether the model understood the evidence. Hence: demote-only stays demote-only,
the alignment gate escalates on doubt, nothing is promoted without §6, and no
converted stage may be the sole signal behind a verdict.

---

## 9. Honest expectations

- **Adjudicator**: the real win versus a hosted provider is removing the network
  round trip and `_LLM_LOCK` serialisation, plus shared prefill across findings
  in a file — not the card's 5.6x. Versus a local Ollama judge, the decode-side
  speedup should be broadly in the card's range.
- **Alignment**: the largest win, and it comes from skipping expensive calls on
  aligned functions, not from decoding faster.
- **Main analyzer / meta prose**: unaffected. If the Phase 0 numbers show these
  dominate total scan time — which is likely — then the honest summary is that
  this technique addresses a minority of the scanner's LLM cost, and the
  decision to proceed should be made with that number in hand.
- **Accuracy**: unknown until Phase 4. Treat every latency number in this plan
  as provisional on the detection gates holding.
