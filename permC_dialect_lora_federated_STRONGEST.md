# Permutation C: Dialect LoRA + Federated Updates [Novelty: STRONGEST]

**Working Title:** "Storage-Efficient Dialect Robustness for Marathi ASR via Federated Low-Rank Adapters"

---

## What This Paper Does

Instead of training and storing separate ASR models for each dialect, this paper keeps one small base model frozen and trains tiny LoRA adapter modules (~1–5% of base model size) for each dialect. These adapters are updated on-device through federated learning that is designed for the patchy, low-bandwidth internet connectivity typical of rural India.

## Research Question

> Can a single small base model plus tiny per-dialect LoRA deltas, updated through bandwidth-aware federated averaging, match dedicated per-dialect models — without the storage cost of multiple checkpoints or the connectivity assumptions of standard federated ASR work?

---

## Claimed Contributions

### Contribution 1: Dialect-Specific LoRA Adapters for ASR

- Freeze the base ASR model (the compressed model from Paper 1, or SraVaani directly)
- Train small Low-Rank Adaptation (LoRA) matrices for each dialect present in the data
- Each adapter is ~1–5% of the base model size (a few MB instead of hundreds)
- At inference time, load the base model + the user's dialect adapter
- **The claim:** dialect-specific LoRA matches or beats (a) the unadapted base model on dialect speech, and (b) dedicated per-dialect full models at a fraction of the storage

### Contribution 2: Connectivity-Aware Federated Averaging

Standard federated learning assumes devices can upload/download model updates regularly. In rural India, connectivity is intermittent, low-bandwidth, and expensive. This contribution:
- Designs a federated averaging protocol that tolerates:
  - Devices going offline for days
  - Asymmetric upload/download bandwidth
  - Partial updates (not all adapter layers, just the ones that changed most)
- Uses compression on the gradient updates themselves (quantized gradients, sparsified updates)
- **The claim:** this protocol converges to comparable accuracy as standard FedAvg but under realistic rural connectivity constraints

### Contribution 3: Base Model from Paper 1

The compressed Marathi-specialized model from Paper 1 serves as the shared base. This means:
- Total storage per dialect = base model (one copy, shared) + dialect adapter (tiny, per-dialect)
- Instead of N full models for N dialects, you store 1 base + N tiny adapters
- This paper explicitly builds on Paper 1 rather than re-deriving a base model

---

## Novelty Assessment

### Contribution 1 (Dialect LoRA for ASR) — NOVEL COMBINATION ✓

| Prior Work | What It Does | How You Differ |
|---|---|---|
| LoRA for ASR speaker adaptation (several 2024–25 papers) | Per-speaker LoRA adapters for voice personalization | You do **per-dialect**, not per-speaker. Dialect = linguistic variation (phonology, morphology), speaker = acoustic variation (pitch, timbre). Different problem. |
| Dialect-aware ASR (various) | Usually trains separate models or uses dialect embeddings | You use LoRA adapters — orders of magnitude cheaper in storage |
| Federated ASR for Indic languages (CodeFed) | Federated learning for code-switch detection | Different task (code-switching, not dialect adaptation), different mechanism (not LoRA) |

**Novelty verdict:** The specific combination of (dialect-level LoRA + ASR + Indic languages) is **genuinely under-explored**. The scoping plan's novelty check confirmed this. This is your cleanest novelty claim across all permutations.

### Contribution 2 (Connectivity-Aware Federated Averaging) — PARTIALLY NOVEL

- Federated learning with communication constraints is well-studied in general ML
- But applying it specifically to ASR dialect adapters under realistic rural Indian connectivity is not
- The novelty is in the application domain + the specific protocol design, not in federated learning itself

**Novelty verdict:** Defensible as a systems/application contribution, not as an algorithmic breakthrough.

---

## Issues & Risks

### Issue 1: ~~Dialect Labels May Be Weak~~ RESOLVED ✓

**Original risk:** Vaani's district metadata was only a proxy for dialect. If district ≠ dialect, the entire framing collapses.

**Status: RESOLVED.** You have the RESPIN corpus with explicit, ground-truth dialect labels (`utt2dialect`). This eliminates the proxy problem entirely. Your framing can use "dialect" confidently, not the weaker "regional variation."

### Issue 2: How Many Dialects? How Much Data Per Dialect?

Even with RESPIN's labels, you need enough data per dialect to train a meaningful LoRA adapter. Questions to check:
- How many distinct dialects does RESPIN label?
- What's the minimum hours-per-dialect? (If some dialects have <1 hour, LoRA may not learn anything useful)
- Is there a long-tail distribution? (5 dialects with 50 hours each + 30 dialects with 1 hour each = very different paper)

**Risk:** If the data is too thin for most dialects, you may only be able to show results for 3–5 major dialects, which weakens the "dialect robustness" claim.

### Issue 3: Federated Learning Simulation Credibility

You're likely simulating the federated protocol, not running it on actual phones in actual rural India. This is fine for a first paper, but:
- The connectivity model (how you simulate "patchy rural internet") needs to be justified and realistic
- Reviewers will ask: "Where did your connectivity model come from? Is it based on real measurements?"

### Issue 4: Depends on Paper 1's Base Model

If Paper 1's compression doesn't work well (e.g., the compressed model has bad WER), this paper's foundation is shaky. You can mitigate by also showing results with an uncompressed base (SraVaani directly), but that weakens the storage-efficiency narrative.

---

## Suggested Fixes

1. **Characterize RESPIN's dialect distribution immediately:** Count dialects, hours per dialect, speakers per dialect. This determines whether the paper is viable at all. Do this before any implementation.

2. **Set a minimum viability threshold:** e.g., "we need at least 5 dialects with ≥10 hours each to show meaningful LoRA adaptation." If RESPIN doesn't meet this, the paper scope needs adjustment.

3. **Ground the connectivity model in real data:** Use published measurements of rural Indian mobile connectivity (e.g., from TRAI reports, Ookla speedtest data, or academic surveys of rural network quality). This makes the federated protocol credible.

4. **Show results with BOTH compressed and uncompressed base models:** This decouples Permutation C from Paper 1's success. If compression helps, great — compound story. If not, the dialect LoRA contribution still stands independently.

5. **Compare against the right baselines:**
   - Unadapted base model (lower bound)
   - Per-dialect fine-tuned full models (upper bound, expensive)
   - Speaker-adapted LoRA (existing approach, wrong granularity)
   - Dialect embedding injection (existing approach, different mechanism)

---

## Relationship to Other Papers

- **Directly depends on Paper 1** for the compressed base model (but can also run independently on SraVaani)
- **RESPIN corpus is the key differentiator** — this is why you can do dialect LoRA when others can't (they don't have dialect labels)
- **Does not compete with Permutation B** (different axis: dialect robustness vs. streaming quality)

## Target Venues

| Venue | Fit |
|---|---|
| **Interspeech** | Strong fit for dialect-focused ASR work |
| **ACM TALLIP** | Best for Indic language / low-resource framing |
| **IEEE SLT** | Spoken Language Technology workshop, good for first results |

## Feasibility

- **Timeline:** ~12–16 weeks (RESPIN characterization → LoRA training → federated simulation → baselines → writing)
- **Hardware needed:** GPU for LoRA training (cheap — LoRA is lightweight), CPU for federated simulation
- **Data needed:** RESPIN (dialect labels) + Vaani Marathi (base model training/eval)
- **Risk level:** LOW-MEDIUM — LoRA adaptation almost certainly works to some degree; the question is how well, and whether the federated protocol adds enough over simple centralized training
- **Depends on:** Paper 1 (weakly — can use uncompressed base as fallback)

---

## Novelty Upgrades (How to Make This Paper Stronger)

### Upgrade 3A: Mixture-of-Dialect-LoRAs With Soft Routing

Instead of loading one adapter per dialect manually, train a **mixture of dialect LoRAs** with a learned soft router: a tiny routing network takes encoder hidden states and produces a soft weighting over all dialect LoRAs. At inference, the model dynamically blends adapters — no manual dialect selection needed. Handles dialect continua, speakers who mix dialects, and unknown dialect speakers.

**Why novel:** MoE-LoRA exists in NLP (MoLoRA) but has NOT been applied to dialect adaptation in ASR. The soft routing reveals a dialect similarity topology (interpretable and linguistically interesting).

### Upgrade 3B: Dialect-Aware Contrastive Adapter Training

Train LoRA adapters using a **contrastive objective**: adapted representations should be close to same-dialect utterances and far from different-dialect utterances of the same sentence. Forces each LoRA to capture *dialect-specific* variation, not speaker/noise variation.

**Why novel:** Standard LoRA fine-tuning optimizes ASR loss only. Contrastive training is *discriminative* — each adapter learns what makes its dialect different.

### Upgrade 3C: Differential Privacy for Federated Dialect Updates

Dialect speech patterns reveal geographic origin, caste, and community membership — sensitive in Indian context. Add **differential privacy** to federated averaging (gradient clipping + calibrated noise). Report privacy budget (ε, δ) and accuracy-privacy tradeoff.

**Why novel:** DP-federated learning exists. Federated ASR exists. But **DP-federated dialect adaptation** protecting speakers' dialect identity is new and socially relevant.

### Revised Contributions After Upgrades

| # | Contribution | Novelty |
|---|---|---|
| 1 | Mixture-of-dialect-LoRAs with soft routing | **NOVEL** |
| 2 | Contrastive dialect adapter training | **NOVEL** |
| 3 | DP-federated dialect updates | **NOVEL** |

**Revised risk:** LOW-MEDIUM → LOW. Each upgrade strengthens an already clean story. Mixture routing is the strongest single upgrade.
