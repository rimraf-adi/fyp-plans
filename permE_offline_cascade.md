# Permutation E: Fully-Offline Cascade Architecture

**Working Title:** "A Fully-Offline Cascade Architecture for Connectivity-Constrained Marathi ASR"

---

## What This Paper Does

Designs a two-model cascade that runs entirely on-device with zero cloud dependency. A tiny first-pass model handles easy utterances cheaply; only when it's uncertain does a second, larger local model get activated. The key design question is the *escalation policy* — when to call the big model, and how big that second model should be.

## Research Question

> Given unreliable connectivity (a real constraint in rural Maharashtra), what local two-stage cascade policy (escalation trigger + second-model size) best balances accuracy against average-case compute, with no cloud fallback assumed at all?

---

## Claimed Contributions

### Contribution 1: On-Device Two-Stage Cascade

- **Stage 1 (Tiny model):** A heavily compressed/quantized ASR model (~10–30 MB) that runs on every utterance. Fast, cheap, but lower accuracy.
- **Stage 2 (Full model):** A larger, more accurate ASR model (~100–200 MB) that is only loaded and run when Stage 1 signals low confidence.
- **Key:** Both models live on-device. There is no cloud fallback at any point. This is a fully offline system.
- **The claim:** the cascade achieves near-full-model accuracy at a fraction of the average-case compute, because most utterances are "easy" and handled by Stage 1 alone.

### Contribution 2: Confidence-Gated Escalation Policy

- Stage 1 produces a confidence score (e.g., CTC posterior entropy, or a learned "difficulty" head)
- If confidence < threshold → escalate to Stage 2
- The threshold is tunable: higher threshold = more escalations = better accuracy but more compute; lower threshold = fewer escalations = faster but riskier
- **The claim:** there exists a sweet spot where 60–80% of utterances are handled by Stage 1, achieving ~95% of Stage 2's accuracy at ~30% of Stage 2's average compute

---

## Novelty Assessment

### Contribution 1 (Two-Stage On-Device Cascade) — LOW-MEDIUM NOVELTY

| Prior Work | What It Does | How You Differ |
|---|---|---|
| Cloud cascades (e.g., on-device → cloud ASR) | Many production systems do this (Google, Apple) | You're fully offline — no cloud at all |
| Model cascades in NLP | Early-exit / multi-model cascades for text | Different modality, but same concept |
| Speculative decoding in LLMs | Small model drafts, large model verifies | Similar concept, different task |

**Novelty verdict:** The cascade *concept* is well-known. The novelty would need to come from the *specific escalation policy design* for ASR under offline constraints, or from showing non-obvious results (e.g., "the optimal second model is surprisingly small").

### Contribution 2 (Confidence-Gated Escalation) — OVERLAPS WITH PRIOR WORK ⚠️

This reuses the same CTC-confidence-gating idea from Paper 1's Contribution 2. The prior art problem carries over:

| Prior Work | What It Does |
|---|---|
| "Accelerating RNN-T Training and Inference Using CTC Guidance" (2022) | CTC confidence gates frame-level compute decisions |
| "Blank-regularized CTC for Frame Skipping" (Interspeech 2023) | Same mechanism |

**However:** There is a genuine distinction to make:
- **Prior work:** frame-level skipping *within one model* (skip frames where CTC is confident → fewer decoder steps)
- **This paper:** utterance-level escalation *between two models* (if whole-utterance CTC confidence is low → run a second, separate model)

These are different problems — one is intra-model frame gating, the other is inter-model cascade triggering. But you **must** make this distinction explicit and prominent, or reviewers will conflate them.

---

## Issues & Risks

### Issue 1: Confidence-Gating Overlap With Published Work

Same CTC-confidence mechanism as Paper 1's Contribution 2. Even though the *application* is different (cascade escalation vs. frame skipping), the *mechanism* is the same. Reviewers familiar with the prior work will see the similarity.

**Severity:** MEDIUM — fixable with careful framing, but requires a dedicated related-work paragraph distinguishing intra-model gating from inter-model escalation.

### Issue 2: Cascade Design Space Is Large

The paper needs to justify specific design choices:
- Why CTC confidence and not a learned difficulty predictor?
- Why two stages and not three?
- How big should Stage 2 be? (This is an expensive hyperparameter search)
- Should Stage 2 re-process from raw audio or continue from Stage 1's encoder states?

Each of these is a design question with multiple valid answers. Without systematic ablation, the paper reads as "we tried one cascade configuration and it worked."

### Issue 3: What Baseline Are You Beating?

The obvious baselines are:
- Always run Stage 1 only (fast, less accurate)
- Always run Stage 2 only (slow, most accurate)
- The cascade should Pareto-dominate both

But a reviewer might ask: "Why not just compress one model to an intermediate size between Stage 1 and Stage 2?" If a single medium-sized model matches the cascade's accuracy at similar average compute, the cascade is unnecessary complexity.

### Issue 4: "Fully Offline" Framing Needs Real Motivation

Claiming "connectivity-constrained" as the motivation is valid for rural India, but you need data or citations to make it concrete. Without grounding in real connectivity measurements, it reads as a hypothetical.

---

## Suggested Fixes

1. **Distinguish escalation-for-cascade from frame-skipping explicitly.** Write a dedicated paragraph in Related Work:
   > *"CTC-guided frame skipping [cite] reduces compute within a single model by skipping blank frames. Our cascade escalation operates at the utterance level between two separate models — a fundamentally different decision granularity..."*

2. **Ablate the cascade design space systematically:**
   - Stage 1 sizes: {10MB, 20MB, 30MB}
   - Stage 2 sizes: {50MB, 100MB, 200MB}
   - Threshold values: {0.3, 0.5, 0.7, 0.9}
   - Plot the Pareto frontier of (average compute, WER) across all configurations

3. **Include the "single medium model" baseline.** If a 50MB single model matches your cascade's accuracy at similar average compute, report that honestly. The cascade wins if there's a *gap* — i.e., no single model can match cascade accuracy at cascade compute.

4. **Ground the offline motivation with real data.** Cite TRAI mobile connectivity reports, Ookla rural India data, or any published measurements of rural Maharashtra network reliability.

5. **Combine with Permutation F's benchmark idea.** Build the "realistic low-end benchmark suite" as part of evaluating the cascade. A benchmark with a technique attached to it is a much easier sell than a standalone benchmark paper.

---

## Relationship to Other Papers

- **Reuses CTC-confidence mechanism from Paper 1** — same overlap risk, but at different granularity (utterance vs. frame)
- **Natural home for the "realistic low-end benchmark suite" idea** (Permutation F) — build the benchmark as part of evaluating the cascade
- **Benefits most from real hardware** — a cascade on a real phone with real latency measurements is much more convincing than simulation
- **Can use Paper 1's compressed model as Stage 1** — builds on that work

## Target Venues

| Venue | Fit |
|---|---|
| **IEEE Access** | Broad scope, accepts simulation-based work |
| **Interspeech** | If focused on the ASR aspects, not the systems aspects |
| **MobiSys / MobiCom workshop** | If paired with real device numbers later |

## Feasibility

- **Timeline:** ~12–16 weeks (design cascade → train Stage 1 & Stage 2 → threshold sweep → ablation → writing)
- **Hardware needed:** GPU for training two models; CPU for cascade evaluation; physical device would strengthen paper significantly
- **Data needed:** Vaani Marathi + RESPIN for evaluation
- **Risk level:** MEDIUM — the cascade will work, but novelty is thin unless the design-space analysis reveals surprising results
- **Recommendation:** Don't do this before Paper 1 and Permutation C. It's a natural third paper once you have the compressed model (Stage 1) and real hardware.

---

## Novelty Upgrades (How to Make This Paper Stronger)

### Upgrade 5A: Shared-Encoder Cascade (Decoder-Only Escalation)

Instead of running a completely separate Stage 2 model, **share the encoder** between stages. Stage 1: shared encoder → CTC greedy (fast). Stage 2 escalation: same encoder hidden states → full TDT decoder (expensive). Encoder runs once — escalation only adds decoder compute.

**Why novel:** Standard cascades re-run the entire pipeline. Shared-encoder cascading with decoder-only escalation avoids redundant encoder computation and is unexplored for CTC→Transducer on-device escalation.

### Upgrade 5B: Error-Correction Escalation (Not Re-Recognition)

Stage 2 is a **text error-correction model**, not a second ASR model. Stage 1 produces a possibly erroneous transcript. Stage 2 takes the transcript + per-token confidence scores and *corrects* it (small seq2seq, ~5–20 MB). Cost is proportional to transcript length (short), not audio length (long).

**Why novel:** ASR error correction as post-processing exists. Using it as the *escalation stage in a fully-offline cascade* — where the correction model is tiny and local — is new. Key insight: correction is cheaper than re-recognition.

**Pick either 5A or 5B as the core mechanism — don't do both.** 5B is bolder and more surprising.

### Upgrade 5C: Learned Multi-Signal Escalation Policy

Instead of "escalate if CTC confidence < threshold" (trivial), train a tiny **policy network** that decides based on: CTC entropy, utterance length, domain keywords detected, and LM perplexity of Stage 1's output.

**Why novel:** Fixed-threshold escalation is trivial. A learned multi-signal policy jointly considering acoustic + linguistic features is more sophisticated.

### Revised Contributions After Upgrades

| # | Contribution | Novelty |
|---|---|---|
| 1 | Shared-encoder cascade OR error-correction escalation | **NOVEL** |
| 2 | Learned multi-signal escalation policy | **PARTIALLY NOVEL** |
| 3 | Fully-offline evaluation with no cloud assumption | **Framing** (not technical novelty) |

**Revised risk:** MEDIUM → LOW-MEDIUM. The shared-encoder or error-correction mechanism is clean, testable, and clearly differentiated from frame-level CTC skipping.
