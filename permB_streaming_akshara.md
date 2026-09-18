# Permutation B: Content-Adaptive Streaming ASR

**Working Title:** "Content-Adaptive Streaming ASR for a Morphologically Rich, Code-Mixed Language: Akshara-Level Emission and Ambiguity-Driven Look-ahead for Marathi"

---

## What This Paper Does

Instead of chopping audio into fixed-size chunks (the standard streaming approach), this paper makes the streaming encoder *content-aware*: it emits whole Indian syllables (aksharas) as output units, waits longer when the audio is ambiguous, and detects when the speaker is about to switch between Marathi and English mid-sentence.

## Research Question

> Do Marathi's specific linguistic properties (suffix-final agreement, conjunct orthography, mid-utterance code-switching) justify departing from fixed-chunk cache-aware streaming, and does a content-adaptive approach beat it at equal latency?

---

## Claimed Contributions

### Contribution 1: Akshara-Level Output Units

Standard streaming ASR emits subword tokens or raw unicode codepoints. This paper uses *aksharas* — the fundamental orthographic syllable unit in Devanagari (e.g., "क्ष" is one akshara, not three codepoints). This matters because:
- An akshara is the smallest unit a Marathi reader considers meaningful
- Partial akshara emission (emitting "क" when the model should wait for "क्ष") creates confusing partial hypotheses on screen
- Akshara boundaries align with phonological boundaries better than arbitrary subword boundaries

### Contribution 2: Ambiguity-Adaptive Look-Ahead

Instead of a fixed chunk size (e.g., always wait 160ms), the model dynamically adjusts how much future audio it waits for:
- Clear vowel in open syllable → emit immediately (low look-ahead)
- Ambiguous consonant cluster or conjunct → wait longer (high look-ahead)
- The look-ahead length is predicted by a small auxiliary head on the encoder

### Contribution 3: Causal Code-Switch Gating

A gate in the joint network predicts whether the speaker is about to switch from Marathi to English (or vice versa) within the current utterance. When a switch is predicted:
- Language model constraints shift
- The decoder's output vocabulary weighting adjusts
- This is causal (streaming-compatible) — it uses only past and current context, no future

---

## Novelty Assessment

### Contribution 1 (Akshara Units) — LIKELY NOVEL but needs verification

No direct prior art found for akshara-level emission in streaming ASR. However:
- The search wasn't exhaustive, particularly for Kannada/Tamil conjunct-heavy streaming work
- Character-level and subword-level streaming ASR is well-studied
- The novelty is in the *unit choice* being linguistically motivated, not in the streaming mechanism itself

**Novelty verdict:** Probably open, but **check Kannada/Tamil streaming ASR literature specifically** before committing. If someone has done "linguistically-motivated emission units for Indic streaming ASR," this contribution weakens significantly.

### Contribution 2 (Adaptive Look-Ahead) — LIKELY NOVEL

Existing streaming ASR uses:
- Fixed chunk sizes (cache-aware streaming conformer)
- Triggered attention (but the trigger is frame-level, not content-semantic)
- No one has made look-ahead length a *function of acoustic/linguistic ambiguity* in the current frame

**Novelty verdict:** Appears open. The risk is not prior art but rather *proving it works* — adaptive latency is harder to tune than fixed latency.

### Contribution 3 (Code-Switch Gating) — PARTIALLY NOVEL

Code-switching detection exists (including for Indic languages). But:
- Existing work is mostly offline, post-hoc classification
- Causal, streaming-compatible code-switch gating *within the decoder* is less explored
- CodeFed (federated Indic code-switch detection) exists but is a different architecture/task

**Novelty verdict:** Defensible novelty in the *causal streaming integration*, not in code-switch detection itself.

---

## Issues & Risks

### Issue 1: Confounding Variables (MAJOR)

If you combine akshara units AND adaptive look-ahead AND code-switch gating, and WER improves, a reviewer will immediately ask:

> *"Which of these three mechanisms caused the improvement? How do I know it wasn't just the look-ahead?"*

You need a **2×2×2 ablation** (unit type × look-ahead policy × gating) = 8 configurations. That's 8 full training runs minimum. This is expensive and time-consuming.

### Issue 2: Latency Measurement Is Tricky

Adaptive look-ahead means *variable* latency per utterance. Comparing against fixed-chunk streaming (constant latency) is apples-to-oranges. You need a latency distribution metric, not just mean latency — and reviewers will scrutinize this.

### Issue 3: Strong Linguistic Claim Needs Strong Evidence

Claiming "akshara units are better because of Marathi's morphology" is a linguistic argument. Reviewers may ask:
- Does it also help for Hindi? Telugu? (If yes, it's not Marathi-specific. If no, why?)
- Is the gain from akshara units or just from larger output granularity?

### Issue 4: No Direct Prior Art = Double-Edged Sword

No prior art means novelty, but also means no established baselines or evaluation protocols. You're defining the problem *and* solving it — that's harder to get right.

---

## Suggested Fixes

1. **Run the full 2×2 ablation** (at minimum unit type × look-ahead policy). Without this, the paper is unpublishable in a serious venue. Budget for 4–8 training runs.

2. **Define a latency distribution metric** — e.g., 50th/90th/99th percentile emission latency, not just mean RTF. Compare at matched P90 latency, not matched mean.

3. **Check Kannada/Tamil streaming ASR literature exhaustively** before committing to this paper. Search specifically for: "streaming Indic ASR conjunct," "akshar output unit speech," "syllable-level streaming recognition Devanagari."

4. **Test akshara units on at least 2 languages** (Marathi + Hindi) to show the benefit is from the linguistic unit choice, not from overfitting to one language's eval set.

5. **Separate code-switch gating into its own ablation line** — it's the most independently testable of the three contributions.

---

## Relationship to Other Papers

- **Shares no baselines with Paper 1** (different axis entirely — interaction quality vs. compute/memory)
- **Akshara tokenization work is reusable** — build it once, cite it in Paper 1 too if you do vocab right-sizing with akshara-informed subwords
- **Does not depend on SraVaani compression** — can use any streaming encoder

## Target Venues

| Venue | Fit |
|---|---|
| **ICASSP** | Streaming ASR tracks, short paper for fast feedback |
| **Interspeech** | Good fit for linguistically-motivated ASR work |
| **IEEE/ACM TASLP** | Journal version with full ablation |

## Feasibility

- **Timeline:** ~16–20 weeks (akshara tokenizer → streaming encoder mods → ablation runs → latency analysis → writing)
- **Hardware needed:** GPU for multiple training runs (the ablation is expensive)
- **Data needed:** Vaani Marathi + at least one other Indic language for cross-lingual check
- **Risk level:** HIGH — three novel mechanisms, each needing isolated validation, plus expensive ablation matrix

---

## Novelty Upgrades (How to Make This Paper Stronger)

### Upgrade 2A: FST-Constrained Akshara Decoding

Formalize valid akshara sequences as a **finite-state transducer** encoding Devanagari orthographic rules (which conjuncts are valid, which matras follow which consonants, where virama is mandatory). Integrate the FST into beam search as a hard constraint. This isn't just "different output units" — it's **linguistically-constrained decoding** that provably eliminates invalid Devanagari sequences. Measure "invalid akshara rate" as a new metric.

**Why novel:** No streaming ASR enforces orthographic well-formedness through a formal grammar.

### Upgrade 2B: Temporal ACT (Learned Halting for Emission Timing)

Frame adaptive look-ahead as **Adaptive Computation Time on the time axis**: the encoder produces a halt probability at each step. Below threshold → wait for more audio. Above → emit. Trained jointly with ASR loss + ponder cost penalty. This is ACT applied to *when to emit* (temporal), not *how many layers to run* (depth) — a genuine reframing of ACT.

**Why novel:** ACT exists for depth-wise pondering (Graves 2016, Universal Transformers). Temporal ACT for streaming emission timing is unexplored.

### Upgrade 2C: Hypothesis Stability Metric

Define **streaming hypothesis stability**: fraction of emitted tokens never revised (stability score) + number of on-screen text changes per second (flicker rate). Show akshara+FST emission improves stability vs. subword emission. This is a user-experience metric missing from streaming ASR evaluation.

**Why novel:** WER and latency are standard. Hypothesis stability is under-studied but directly affects UX.

### Revised Contributions After Upgrades

| # | Contribution | Novelty |
|---|---|---|
| 1 | FST-constrained akshara decoding | **NOVEL** |
| 2 | Temporal ACT for emission timing | **NOVEL** |
| 3 | Hypothesis stability metric + code-switch gating | **NOVEL** (metric) + **PARTIAL** (gating) |

**Revised risk:** HIGH → MEDIUM. FST and temporal ACT are independently testable.
