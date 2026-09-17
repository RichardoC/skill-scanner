# Plan: parallel constrained decoding ("RLCD") for the scanner's categorical LLM stages

Status: **proposal, not implemented.** No code in this change.

References:

- TypeSafe AI, [Introducing System One Models &
  Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  (2026-09-15) — the origin of the term and the hosted implementation.
- [`harshatheg/Qwen-2.5-1B-RLCD`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
  (Apache-2.0, created 2026-09-16) — a third-party local implementation of the
  sampling half.

---

## 1. What "RLCD" refers to, and the two things that share the name

**RLCD is Reinforcement Learning for Calibrated Decisions**, TypeSafe AI's
training method for what they call **System One Models** — a model class that
takes structured program state in and returns typed, probabilistic decisions
out, rather than text. Where RLHF optimises for human-preferred prose and RLVR
for programmatically verifiable outputs, RLCD optimises for *epistemically
honest probabilities on decision tasks*. Their stack has three parts: a new
architecture, a parallel sampler, and RLCD itself. The first public model is
**Jev**, in early access since 2026-09-15.

The two halves matter separately for us, because the two available artefacts
give us different halves:

| | Parallel sampling | RLCD-trained calibration |
|---|---|---|
| **Jev** (hosted, early access) | yes | yes |
| **`Qwen-2.5-1B-RLCD`** (local, HF) | yes | **no** |

### 1.1 The hosted path: Jev

Claimed: 70ms–500ms end to end against 3–329s for frontier models; input at
$0.042/MTok with output tokens free; type errors structurally impossible;
cardinality up to 255 per field, with a two-stage score-then-choose fallback
above that; and calibrated, consistent confidences on every field.

Two details are directly useful to this plan. First, TypeSafe ship a **"System
One LLM wrapper"** that constrains ordinary LLMs to emit decisions against the
same API — which is exactly the parity harness §6 needs, and means the shadow
comparison does not have to be built from scratch. Second, their framing of the
confidence problem is precisely the adjudicator's: *"If a model can do a task
95% of the time but doesn't say when it's in the 5%, it can't automate that
task."*

### 1.2 The local path: the Hugging Face repo

Read carefully, because three things differ from what the name suggests.

**It contains no weights.** The repo holds source code published under a model
repo: `core/engine.py`, `core/engine_mlx.py`, `core/engine_torch.py`,
`core/schema.py`, `core/prompt_builder.py`, a FastAPI demo server, and four
JSON presets. The engine loads `Qwen/Qwen2.5-1.5B-Instruct` (torch) or
`mlx-community/Qwen2.5-1.5B-Instruct-4bit` (MLX) from elsewhere.

**It is not RLCD-trained**, despite the name. It reimplements the parallel
sampling mechanism on a stock Qwen. Its "calibrated probabilities" are a plain
softmax over a sliced vocabulary — which is the *overconfidence* failure mode
RLCD exists to fix, not a solution to it. Confidence numbers from this path must
not be treated as interchangeable with Jev's, and any threshold tuned against
one is invalid for the other.

**It is not Apple-only.** The card foregrounds MLX and Apple Silicon, but
`core/engine.py` is a router: MLX on Darwin, otherwise `engine_torch.py`
(CUDA/CPU via `transformers`). That makes it viable for the Linux CI paths that
matter most to this project.

Internally we should avoid the bare acronym, since it names a training method we
are not doing on the local path. This plan calls the shared mechanism
**parallel constrained choice** and proposes `parallel_choice` as the
module/flag name.

### 1.3 The mechanism, in one paragraph

Taking the local implementation, which is readable: for a schema whose every
field is a boolean or a bounded enum (≤255 choices), the engine prefills the
context prompt **once**, broadcasts the resulting KV cache to batch size *M*
(one row per field), runs **one** batched forward pass over per-field suffixes,
and then, per field, slices the logits down to the first token of each candidate
choice and takes the argmax plus a softmax over that slice. The JSON is
assembled programmatically from the winning values, so syntax validity is
structural rather than learned. Decode cost collapses from *O*(output tokens) to
*O*(1) forward passes. Jev's sampler is not public, but the shared 255-choice
ceiling and the two-stage fallback above it suggest the same family.

Both sets of headline numbers need the same discount. The HF card's 5.6x–7.0x
is the author's, on an M4 Max with a 4-bit 1.5B model and short contexts.
TypeSafe's 193.6x/444.6x come from their own workflow evals, measured from
their laptops, built by their own capabilities team, and scored against the
*average of frontier models' probabilities as a reference* rather than against
ground truth — they say so themselves in the post's nuance sections, which is to
their credit and is also exactly why the numbers do not transfer. "No type
errors" and "100% schema validity" are properties of programmatic assembly, not
evidence that the answers are right.

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

One framing point must not get lost. Whichever path we take, adopting this is
**not "the same judge, decoded faster"** — it replaces the frontier model
currently judging a stage with a different model of a different class. On the
local path that is a 1.5B Qwen. On the hosted path it is Jev, which claims
comparable intelligence *on System One tasks*, but has no published result on
adversarial security classification and none on our corpora. Either way it is an
accuracy change at least as much as a latency change, and this plan is built
around proving the accuracy side before taking the latency win.

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
  scanner + meta + adjudicator traffic hit Bedrock throttle limits. Neither
  choice engine shares that quota — the local one has none, and Jev's is
  separate — so adjudication can run concurrently with the scanner's own LLM
  stage instead of serialising behind it. On the hosted path, confirm the
  vendor's own rate limits before removing it.
- **One prefill, many findings.** Several findings routinely land in the same
  file. Prefill that file's context once and score every finding's fields in a
  single batched pass. This is a bigger win than the card's 5.6x, and it is
  only available because the technique exposes the prefill/suffix split. On the
  hosted path this is only available if the API exposes prefix reuse; if it does
  not, ask, because it is worth asking for.

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

## 4. Architecture: one seam, three implementations

The seam matters more now than it did when this looked like a single-vendor
question. There are two viable backends with very different dependency, cost,
and data-handling profiles, and the sensible move is to commit to neither in the
call sites. Add `skill_scanner/core/analyzers/parallel_choice/` with a
deliberately narrow interface, so no security-critical caller learns about
torch, MLX, logits, or a vendor SDK:

```
ChoiceSchema      # fields: name -> (bool | enum[choices], description)
ChoiceResult      # field -> (value, probability); plus prefill/eval timings
ChoiceEngine      # classify(context: str, schema: ChoiceSchema) -> ChoiceResult
                  # classify_batch(contexts, schema) for the shared-prefill case
```

Three implementations behind it:

- `JevChoiceEngine` — the hosted System One API. No heavyweight dependencies,
  no model load, no platform constraint, calibrated confidences, and output
  tokens billed at zero. Blocked on early-access approval (§7).
- `LocalParallelChoiceEngine` — the mechanism in §1.3. Backend router mirroring
  `core/engine.py`: MLX on darwin/arm64 when available, torch otherwise, and a
  clean `unavailable` state when neither is installed. No vendor dependency and
  nothing leaves the host, at the cost of a much weaker model and uncalibrated
  probabilities.
- `ProviderChoiceEngine` — today's LiteLLM path, asked for the same schema as
  JSON and parsed. Needed for three things: parity reference during shadow
  evaluation, fallback when the chosen engine is unavailable, and keeping the
  call sites honest about not depending on any one backend's behaviour.
  TypeSafe's "System One LLM wrapper" does the same job on their side and is
  worth evaluating before we write our own.

Every caller takes a `ChoiceEngine` and keeps its existing failure semantics. An
engine returning `None` must be indistinguishable, to the caller, from today's
"LLM unavailable" path.

Confidence thresholds are **per engine, not per call site**. The adjudicator's
`min_fp_confidence` tuned against Jev's calibrated probabilities is meaningless
against a Qwen softmax, and vice versa. Store thresholds keyed by engine and
refuse to start when a configured engine has no calibrated threshold on record.

### Two constraints the seam must enforce

**Enum labels must be first-token distinct (local path only).** Jev states no
such restriction and reports a two-stage score-then-choose path for high
cardinality, so this is a property of the HF reimplementation, not of the
technique. On that path: `schema.py` maps each choice to
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

**Prefill, not decode, will dominate our prompts.** Both vendors' demo contexts
are short — TypeSafe note their side-by-side uses "a short, dense" paragraph and
say plainly that it "paints our model in an advantageous light". Ours are skill
files and finding context running to many kilobytes. The technique removes
decode cost only, so expected speedup scales with *output* length: large for the
adjudicator (~250 output tokens today), negligible anywhere the model writes
prose. This also shapes the hosted economics: free output tokens are worth
little to us when we are paying for a large prefill, though $0.042/MTok input is
still one to two orders of magnitude under what the current judge costs.
`engine_torch.py` additionally does `copy.deepcopy(base_cache)` per call, whose
cost grows with prefix length — for our prompt sizes this needs measuring and
probably replacing with an expand-in-place broadcast.

---

## 5. Phases

**Phase 0 — measure before optimising.** Extend the existing `LLMTokenUsage` /
`_extract_token_usage` plumbing to record per-stage wall-clock and output-token
counts (analyzer, meta, adjudicator, alignment, classifier), and emit them in
the scan report under a debug key. Run it over `evals/skills` and the official
bundled-skills set. **Gate:** if the adjudicator and alignment stages turn out
not to be material, most of this plan is not worth doing, and Phase 0 is the
cheap way to find that out.

Do in parallel with Phase 0, because it has a lead time we do not control: **join
the Jev waitlist** and start the vendor review in §7. If early access does not
arrive, the plan still runs on the local path, just with a worse model and
uncalibrated confidences.

**Phase 1 — the seam.** `parallel_choice/` package, `ChoiceSchema` with
first-token collision validation, `ProviderChoiceEngine`,
`LocalParallelChoiceEngine` with both backends, and `JevChoiceEngine` if access
has landed. No call site changes. Unit tests with a stub engine; integration
tests marked and skipped when torch or vendor credentials are absent.

**Phase 1b — backend bake-off.** Before converting anything, run all available
engines against one converted schema (the adjudicator's) over a sample of the
shadow corpus, and compare agreement, calibration, and latency. Decide the
default backend on that evidence rather than on either vendor's numbers. A
plausible outcome is Jev for hosted/CI use and the local engine for
air-gapped deployments, in which case both need §6 gates separately.

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
`ScanPolicy.parallel_choice.mode` and `off` as the default, plus
`--parallel-choice-engine jev | local` selecting the backend.

| Mode | Behaviour |
|---|---|
| `off` | Provider path only. Today's behaviour, byte-for-byte. |
| `shadow` | Run **both** engines. Act on the provider's answer; record the choice engine's answer, its per-field probabilities, and both latencies as telemetry. |
| `enforce` | Act on the choice engine's answer; fall back to the provider on unavailability or malformed state. |

Promotion from `shadow` to `enforce` requires the following **per converted call
site and per engine**. Engines are not interchangeable here: passing the gates
with Jev says nothing about the local path, and a deployment that switches
backends is making a fresh detection change.

- **Agreement**: engine-vs-provider agreement rate and a confusion matrix over
  the shadow corpus, with the disagreement set reviewed by hand.
- **Calibration**: a reliability diagram and expected calibration error over the
  shadow corpus, per engine. This is the gate that decides whether a confidence
  threshold means anything, and it is the one place we can test TypeSafe's
  central claim on *our* data rather than take it on trust. It is also the gate
  the local path is most likely to fail, since a sliced softmax from a 1.5B
  model is not a calibrated probability. Every threshold we ship —
  `min_fp_confidence`, the alignment escalation floor — must be read off this
  curve rather than guessed.
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

## 7. Packaging, supply chain, and data handling

The two backends have almost disjoint risk profiles, so review them separately.

### 7.1 Hosted (Jev)

Packaging is nearly free: an HTTP client, an API key handled the same way
`SKILL_SCANNER_LLM_API_KEY` already is, and no change to the dependency tree.
That alone removes most of §7.2's problems. The costs move elsewhere:

- **Availability.** Early access since 2026-09-15, waitlist-gated. Nothing here
  can be scheduled until access lands, which is why §5 starts that clock early.
- **Maturity.** A company two years out of stealth, a model days old, no
  published SLA, no stated retention or deletion policy in the announcement, and
  a pricing model the post itself declines to claim is unsubsidised. Depending
  on it for a release-gating detection stage is a bet on a single young vendor.
  It belongs behind the same optional, off-by-default switch as the adjudicator,
  never on the default scan path, until that changes.
- **Data handling — the gate that actually matters.** Adopting this sends
  **customer skill source code** to a new third party. Much of what this scanner
  reads is proprietary, and some of it is run in CI on private repositories.
  Requirements before any conversion ships on this path: a data processing
  agreement, a documented retention and training-use position in writing, and
  a clear statement in our own docs naming exactly which stages transmit what.
  Users on `off` must transmit nothing, and that must be testable.
- **Egress.** CI runners are frequently network-restricted. A new required
  endpoint needs documenting alongside the existing provider endpoints, and the
  fallback to `ProviderChoiceEngine` must be clean when it is unreachable.

### 7.2 Local (the Hugging Face engine)

This is a security scanner. Pulling runtime code from a Hugging Face repo
created the day before, with 0 downloads and no published upstream git remote
(the card's clone URL is the placeholder
`github.com/your-org/parallel-constrained-decoding`), is not something we should
ship — and it is precisely the pattern this tool exists to flag.

**Recommendation: reimplement, do not vendor.** The load-bearing logic is small
— prefill, KV broadcast, per-field logit slice, softmax over candidates — and
reimplementing it in-tree against `transformers` gives us the first-token
collision handling, the batched-prefill path, and the deepcopy fix we need
anyway. Credit the technique, TypeSafe's published description of it, and the
reference implementation in the module docstring and in this document. If
vendoring is preferred instead, copy the Apache-2.0 sources under `third_party/`
with license headers, a pinned commit SHA, and an entry in the existing
supply-chain review — not a runtime download.

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

Its compensating advantage is real and worth stating: nothing leaves the host.
For air-gapped and privacy-sensitive deployments this is the only viable path,
whatever the bake-off says about accuracy.

---

## 8. Security notes

**A genuine hardening win.** The scanner feeds untrusted skill content to the
judge, and the prompts go to some length to state that package content is inert
evidence. With constrained decoding, injected content *cannot* steer the output
beyond the candidate set: the value space is the enum, and the JSON is assembled
by us. For the categorical stages this is a stronger guarantee than any prompt
instruction, and it is worth stating as a benefit in its own right rather than
as a side effect of the speed work.

**A second, quieter win: knowing when it does not know.** The reason the
adjudicator is demote-only and off by default is that we cannot tell when its
judgement is wrong. If RLCD's calibration claim holds on our data — the §6 gate —
that is a direct answer to this stage's actual problem, and arguably more
valuable to us than the latency. It is what would let a threshold mean
"demote only where the model is right 97% of the time" instead of "demote where
the model said 4 out of 5".

**The countervailing risk.** Constrained decoding makes a wrong answer
*confident* rather than malformed. A probability over a candidate set says
nothing about whether the right answer was in the set, whether the model
understood the evidence, or whether the evidence was adversarial — and this
scanner's inputs are adversarial by definition, which is not the distribution
either vendor benchmarked on. The risk is sharper on the local path, where a
4-bit 1.5B Qwen is a materially weaker judge than the models these stages use
today and its softmax is not calibrated at all. So regardless of engine:
demote-only stays demote-only, the alignment gate escalates on doubt, nothing is
promoted without §6, and no converted stage may be the sole signal behind a
verdict.

---

## 9. Honest expectations

- **Adjudicator**: on the local path the win is removing the network round trip
  and `_LLM_LOCK` serialisation, plus shared prefill across findings in a file —
  not the card's 5.6x. On the hosted path a 70–500ms call replacing a
  multi-second one is a genuine order of magnitude, and the lock can go either
  way, since Jev's quota is not shared with the scanner's main judge.
- **Alignment**: the largest win, and it comes from skipping expensive calls on
  aligned functions, not from decoding faster. This holds on both paths.
- **Cost**: plausibly the most defensible benefit on the hosted path. These
  stages are prefill-heavy and output-light, which is the shape $0.042/MTok-in,
  free-out prices best. Worth computing properly from Phase 0's token counts —
  it may carry the proposal even if the latency win disappoints.
- **Main analyzer / meta prose**: unaffected. If the Phase 0 numbers show these
  dominate total scan time — which is likely — then the honest summary is that
  this technique addresses a minority of the scanner's LLM cost, and the
  decision to proceed should be made with that number in hand.
- **Accuracy and calibration**: unknown until Phase 4. Both vendors' published
  numbers are self-run, on non-adversarial tasks, and in TypeSafe's case scored
  against frontier-model consensus rather than ground truth. Treat every figure
  in this plan as provisional on our own detection gates holding.

---

## 10. Questions to put to TypeSafe on early access

They asked for exactly this feedback, so ask. In rough priority order:

1. **Data handling**: retention period, training use, deletion, sub-processors,
   and whether a DPA is available. This gates everything else for us.
2. **Adversarial robustness**: the inputs we classify are attacker-authored and
   include prompt injection aimed at the judge. Is there any evaluation of Jev
   under adversarial input, and does the parallel sampler change the attack
   surface relative to autoregressive decoding?
3. **Calibration under distribution shift**: calibration is the headline claim,
   and our distribution (malicious skill packages) is nothing like the training
   or eval distribution. What degradation should we expect, and is there a
   recommended recalibration procedure on a labelled set like ours?
4. **Prefix reuse**: our prompts are prefill-dominated with many small decisions
   over the same file context. Can a prefix be prefilled once and reused across
   queries, and is it billed once?
5. **Long contexts**: maximum input length, and how pricing and latency behave at
   the tens-of-kilobytes end rather than the short-paragraph end.
6. **Determinism**: are repeated identical queries bit-identical? Our promotion
   gate requires five exact runs.
7. **Rate limits and availability**: concurrency ceilings, and any SLA or status
   commitment we can plan a CI-blocking stage around.
