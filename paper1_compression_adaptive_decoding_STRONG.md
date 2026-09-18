# Paper 1: Language-Specialized Compression + Adaptive-Compute Decoding [Novelty: STRONG]

**Working Title:** "Language-Specialized Compression and Adaptive-Compute Decoding for Low-Resource Marathi ASR on Resource-Constrained Hardware"

---

## What This Paper Does

Takes a large multilingual Indic ASR model (SraVaani, 430M params, 65 languages), surgically removes the capacity wasted on languages other than Marathi, and pairs the compressed model with a runtime that spends compute adaptively (cheap decoder for easy frames, expensive decoder for hard frames). Evaluates everything under honest constrained-CPU conditions.

## Research Question

> Given a multilingual Indic ASR model, how much of its capacity is language-general vs. wasted on languages you don't need, and can removing the waste — combined with runtime compute that adapts to utterance difficulty — get Marathi ASR to run well within a low-end device's budget without retraining from scratch or losing cross-lingual transfer?

---

## Claimed Contributions

### Contribution 1: Language-Specialized Compression

- **Vocab right-sizing:** Retokenize on Marathi-only text, shrink SentencePiece vocab from ~5000 → ~500–1000 pieces. Shrinks embedding table and joint network output projection.
- **Usage-informed structured pruning:** Run Marathi audio through the full multilingual encoder, collect per-head/per-channel activation statistics, prune heads/channels disproportionately used by other languages while keeping shared + Marathi-relevant capacity.
- **Baselines:** Uniform/magnitude pruning (same budget), from-scratch small Marathi-only model (same budget), generic output-only knowledge distillation (same budget).

### Contribution 2: Adaptive-Compute Decoding

- CTC-greedy as default fast path.
- Only invoke the heavier autoregressive TDT predictor+joint on frames where CTC confidence is low.
- Metric: average-case FLOPs/wall-clock reduced at equal WER.

### Contribution 3: Constrained-CPU Evaluation Protocol

- 1–2 CPU threads (taskset/cgroups), no GPU.
- Quantized runtime (ONNX Runtime INT8).
- Capped RAM via cgroups; report peak RSS.
- Report all four: model size (MB), peak RAM (MB), RTF, WER.

---

## Novelty Assessment

### Contribution 1 — PARTIALLY NOVEL (strongest part, but has close prior art)

| Prior Work | What It Does | How You Differ |
|---|---|---|
| **ASR Pathways** (Google, 2022) | Learns per-language sub-networks in a multilingual RNN-T during training | You do **post-hoc** activation-based pruning on an **existing frozen model** — no retraining of the teacher |
| **DAMA** (2026) | Layer-wise language-specificity analysis + adaptation | Very close in spirit; you'd need to show your activation-based importance scoring differs meaningfully |
| **Multilingual encoder compression** (2026) | Vocab trim + structured pruning + distillation combined for low-resource languages | Same recipe, but for **text** (NLP), not speech. Modality difference is your differentiator |

**Novelty verdict:** Defensible but not wide-open. The specific combo of (vocab right-sizing + activation-based structured pruning + Indic phonetic transfer preservation + speech modality) hasn't been published together, but each piece exists independently. You need strong experimental evidence that the combo matters more than its parts.

### Contribution 2 — NOT NOVEL ⚠️

| Prior Work | What It Does |
|---|---|
| "Accelerating RNN-T Training and Inference Using CTC Guidance" (2022) | Blank-frame skipping via co-trained CTC — exactly this mechanism |
| "Blank-regularized CTC for Frame Skipping in Neural Transducer" (Interspeech 2023) | Same: CTC confidence gates which frames go to the transducer |
| Hybrid CTC-Transducer models generally | Use exactly this blank/confidence-gated skip mechanism |

**Novelty verdict:** This is already published multiple times. A reviewer familiar with transducer literature will flag this immediately. Cannot be claimed as a novel contribution.

### Contribution 3 — METHODOLOGY, NOT RESEARCH

**Novelty verdict:** Useful and honest, but "we ran inference under cgroups constraints and reported 4 metrics" is engineering methodology, not a research contribution on its own. Strengthens the paper's credibility but won't excite reviewers as a standalone claim.

---

## Issues & Risks

1. **The paper's weight rests on Contribution 1, which has close prior art.** If reviewers see ASR Pathways / DAMA as covering the same ground, the paper's core novelty collapses. You must explicitly differentiate (post-hoc vs. during-training, speech vs. text, Indic phonetic transfer specifically).

2. **Contribution 2 is a liability if claimed as novel.** It will actively hurt the paper — a reviewer who spots it will question your literature awareness and become skeptical of other claims too.

3. **If usage-informed pruning barely beats uniform pruning**, the headline result is weak. The "does it even help?" question looms.

4. **No physical device validation.** The constrained-CPU protocol is an honest proxy, but reviewers may still push back with "show me a real phone."

---

## Suggested Fixes

1. **Demote Contribution 2:** Do NOT claim CTC-gating as novel. Use it as a known optimization, cite the prior work explicitly, and frame it as: "We show that known CTC-gating composes with our compression pipeline." It becomes part of the experimental setup, not a contribution.

2. **Restructure the 3 contributions as:**
   - C1: Usage-informed monolingual specialization pipeline (vocab + pruning as one unified method)
   - C2: Cross-lingual phonetic transfer analysis under compression (which languages' capacity survives? which is reusable for Marathi?)
   - C3: Multi-dimensional constrained-resource evaluation (the protocol + CTC-gating as deployment technique, not novelty)

3. **Lean hard into the transfer analysis.** Even if pruning only ties with uniform pruning on WER, showing *which* language capacity transfers to Marathi is scientifically interesting and independently publishable.

4. **If pruning doesn't beat baselines:** Reframe as an honest negative/mixed result — "when does language-usage information help vs. not" is still publishable. Keep all 4 metrics so a loss on WER can still show a win on size/speed.

---

## Target Venues

| Venue | Fit |
|---|---|
| **ACM TALLIP** | Best thematic fit (low-resource Indic language focus) |
| **IEEE Access** | Broad scope, faster review, good for modeling+systems combo |
| **Interspeech / ICASSP** | Conference version for faster feedback before journal |

## Feasibility

- **Timeline:** ~15–18 weeks (baseline reproduction → pruning → adaptive decoding → evaluation → writing)
- **Hardware needed:** GPU for training/profiling, CPU-only for evaluation (no physical device required)
- **Data needed:** Vaani/SraVaani Marathi subset
- **Risk level:** MEDIUM — techniques will produce results, question is whether results are novel enough

---

## Novelty Upgrades (How to Make This Paper Stronger)

### Upgrade 1A: Phonetic Transfer Topology (replaces vanilla activation pruning)

Instead of just measuring "which heads activate for Marathi," build a **cross-lingual phonetic transfer graph**: for each attention head, measure activation correlation across all language pairs. Pruning becomes graph-informed — keep heads on the transfer path from related languages (Hindi, Konkani, Sanskrit) to Marathi. Prune heads only connected to unrelated clusters (Dravidian, etc.).

**Why novel:** Nobody has mapped the internal transfer topology of a multilingual ASR encoder. ASR Pathways learns sub-networks *per language independently* — it doesn't map cross-language relationships inside the model.

### Upgrade 1B: Phoneme-Aware Vocabulary Construction

Instead of training SentencePiece on Marathi text (frequency-based), build vocab using **acoustic-phonetic clustering**: run audio through encoder, cluster into phonetic units, build tokens that respect phonetic boundaries and share cognate phonemes across Hindi/Konkani.

**Why novel:** Standard vocab right-sizing ignores acoustic structure. This ties the tokenizer to the speech signal.

### Upgrade 1C: Distillation Through a Phonetic Bottleneck

Add a shared phonetic bottleneck (articulatory features: place, manner, voicing, aspiration, nasality) between teacher and student. Student matches teacher at this linguistically grounded intermediate representation, not at output logits.

**Why novel:** Standard KD matches output distributions. Phonetic-bottleneck distillation preserves meaningful intermediate representations.

### Upgrade 1D: Entropy-Adaptive Encoder Compute (replaces CTC-gating)

Instead of CTC-gated frame skipping (published), do **encoder-level** entropy-adaptive layer allocation: high-entropy frames get more encoder layers, low-entropy frames exit early. This operates *inside the encoder* (layer depth), not at the decoder boundary (frame skip). Fundamentally different mechanism from CTC-guided skipping.

**Why novel:** ACT for speech encoders is limited; entropy-adaptive layer allocation within a *compressed* encoder is unexplored.

### Revised Contributions After Upgrades

| # | Contribution | Novelty |
|---|---|---|
| 1 | Phonetic-transfer-topology-informed pruning | **NOVEL** |
| 2 | Phoneme-aware vocabulary construction | **NOVEL** |
| 3 | Entropy-adaptive encoder compute + constrained-CPU eval | **PARTIALLY NOVEL** |
