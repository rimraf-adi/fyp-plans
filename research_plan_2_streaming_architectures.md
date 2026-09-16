# Streaming Architectures for Indic Dialect ASR: Beating Sarvam

A competitive analysis and architectural exploration plan — not locked to Mamba, but exploring the full design space of causal/streaming architectures for dialect-robust Indic ASR.

---

## What Sarvam is Doing (Saaras v3)

### Architecture
- **Encoder trained from scratch with causal attention** — streaming-native, not retrofitted
- Described as an "audio extension" of their 3B parameter language model (Sarvam-1)
- Unified model trained on varying chunk sizes — handles different latency/accuracy tradeoffs in one model
- Supports 3 operational modes: Fast (<150ms TTFT), Balanced, Accurate

### Training
- **1M+ hours** curated multilingual audio (22 languages + English)
- Adaptive data sampling across high/low-resource languages
- Heavy real-world conditioning: telephony 8kHz, market noise, construction sites, vehicles
- Explicit code-switching optimization (Hinglish, Tanglish, etc.)

### Product Stack
| Product | Function |
|---|---|
| **Saaras v3** | ASR (streaming STT) |
| **Bulbul v3** | TTS (48kHz, 11 languages) |
| **Shuka v1** | Audio understanding (Saaras encoder + Llama-3-8B decoder, open-source) |
| **Sarvam-1** | Foundation LLM (3B params) |

### Publicly Known Weaknesses
- **Low-resource dialect degradation**: Performance gap persists between major languages (Hindi) and low-resource dialects
- **Rural/non-standard accents**: Despite training on diverse data, rural dialectal variation still degrades WER
- **Benchmark vs. real-world gap**: 2.8x-5.7x WER degradation in production vs. clean benchmarks
- **Architecture is closed-source**: No published paper on Saaras v3 internals — they publish only benchmarks (Indic DiarBench) and evaluation methodology

### Key Benchmark Competitor
**SraVaani 1.0** (IISc SPIRE Lab, arXiv:2608.08235, Aug 2026):
- FastConformer ~430M params, Hybrid TDT-CTC decoder
- 65 languages/dialects (vs Sarvam's 22)
- Novel **audio-image representation alignment** pretraining stage
- Open-source (MIT license)
- Beats Saaras v3 on low-resource/tribal languages specifically
- Competitive on high-resource languages

---

## Where Sarvam is Beatable

You're not going to beat Sarvam on **scale** (1M hours, full engineering team). You beat them on **architectural insight** and **specific failure modes**. Here's where:

### 1. Dialect-Specific Performance
Sarvam treats all speech as a flat multilingual problem. They don't model dialect variation explicitly — no dialect conditioning, no dialect-aware state dynamics. Their model sees "Hindi" as one language, not "Braj vs Awadhi vs Bundeli vs Khariboli."

**Your advantage**: RESPIN has explicit `utt2dialect` labels across 38+ dialects. You can build dialect-aware models that Sarvam cannot, because they don't have (or don't use) dialect-level supervision.

### 2. Streaming Architecture Design
Sarvam uses causal attention (transformer-based). This is the obvious choice but not necessarily the best:
- Causal attention still requires KV-cache that grows with context
- Memory footprint scales linearly with sequence length during inference
- No architectural inductive bias for the sequential nature of phonological processes

**Your advantage**: Non-attention architectures (SSMs, gated linear recurrences, RWKV) have constant memory per step. For a true edge-deployment streaming system in rural India (agriculture domain), this matters.

### 3. Efficiency at Inference
Saaras v3 is optimized for cloud deployment. You can target on-device/edge — a genuinely different and underserved setting for Indic ASR.

---

## The Architecture Design Space (Beyond Just Mamba)

Here's what's available in 2026 for streaming-first ASR encoders:

### Tier 1: Pure Recurrent / SSM (Natively Causal)

| Architecture | Key Mechanism | Streaming | Memory/Step | ASR Status |
|---|---|---|---|---|
| **Mamba-2 (SSD)** | Selective state space, hardware-optimized | Native | O(1) | Samba-ASR, ConMamba |
| **RetNet** | Multi-scale retention with exponential decay | Native | O(1) | Speech enhancement; NO ASR paper yet |
| **RWKV-7 ("Goose")** | Linear attention + token shift | Native | O(1) | AudioRWKV, RWKV-ASR |
| **Hawk (RG-LRU)** | Gated linear recurrence | Native | O(1) | No ASR paper yet |

### Tier 2: Hybrid (Recurrent + Local Attention)

| Architecture | Key Mechanism | Streaming | Memory/Step | ASR Status |
|---|---|---|---|---|
| **Griffin** | Gated linear recurrence + local attention | Native | O(window) | No ASR paper yet |
| **Stateformer** | MH-SSM + Transformer blocks | Configurable | O(window) | Interspeech 2023, LibriSpeech SOTA |
| **AS-Split Conformer-Mamba** | Mamba for global + Conv for local | Configurable | O(1) + O(kernel) | Recent, limited evaluation |

### Tier 3: Efficient Attention (Streaming-Adapted)

| Architecture | Key Mechanism | Streaming | Memory/Step | ASR Status |
|---|---|---|---|---|
| **Zipformer** | U-Net multi-scale + dynamic right-context | Chunked | O(chunk) | ICLR 2024, production-grade |
| **Emformer** | Memory-efficient transformer | Block | O(memory bank) | Production (Meta) |
| **Cache-aware Conformer** | KV-cache reuse (Nemotron) | Incremental | O(cache) | NVIDIA production |

---

## Retention vs Attention vs SSM: The Mathematical Core

This is the theoretical backbone of your paper. You're not just comparing architectures — you're comparing **three fundamentally different ways to propagate information across a sequence**. Each has a different inductive bias for how "history" is compressed and used.

### 1. Attention (Transformer / Conformer)

Standard multi-head attention computes:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

**At inference (streaming):**
- Must maintain a KV-cache: stores all past K, V vectors
- Memory per step: O(context_length × d)
- Every new token attends to ALL previous tokens
- **No information compression** — full history is retained, which is powerful but expensive

**Inductive bias**: Every past frame is equally accessible. No built-in notion of "recent frames matter more." The model must learn temporal decay from data.

### 2. State Space Model (Mamba-2)

The selective SSM computes:

```
h_t = A_t * h_{t-1} + B_t * x_t       (state update)
y_t = C_t * h_t                         (output)
```

**At inference (streaming):**
- Maintains fixed-size hidden state h_t
- Memory per step: O(state_dim) — constant, does not grow
- **Lossy compression**: history is compressed into the state; old information is gradually "overwritten"
- A_t is input-dependent (selective) — the model learns WHAT to remember

**Inductive bias**: Information is compressed through a learned linear dynamical system. The state transition A controls how quickly information decays. But A acts uniformly across the state — there's no explicit multi-scale temporal structure.

### 3. Retention (RetNet) — THE MIDDLE GROUND

Retention replaces softmax attention with **exponentially decayed linear attention**:

**Recurrent form (streaming inference):**
```
S_n = γ · S_{n-1} + K_n^T V_n          (state update)
o_n = Q_n · S_n                          (output)
```

**Parallel form (training):**
```
O = (Q K^T ⊙ D) V
```
Where D is a causal mask weighted by exponential decay: `D[i,j] = γ^(i-j)` for i >= j, 0 otherwise.

**At inference (streaming):**
- Maintains fixed-size state S_n (a d_k × d_v matrix)
- Memory per step: O(d²) — constant, like SSM
- **Structured lossy compression**: γ controls the decay rate explicitly

**The critical difference — Multi-Scale Retention:**
RetNet uses **different γ values per head**:
- Head 1: γ₁ = 0.99 → long memory (slow decay, remembers distant frames)
- Head 2: γ₂ = 0.95 → medium memory
- Head 3: γ₃ = 0.80 → short memory (fast decay, focuses on recent frames)
- Head h: γ_h = assigned from a predefined set

This gives RetNet an **explicit multi-scale temporal hierarchy** — different heads capture different temporal resolutions, hardcoded through the decay structure.

**Chunkwise recurrent form (best of both worlds):**
```
For chunk c:
  O_c = (Q_c K_c^T ⊙ D_intra) V_c + (Q_c · S_{c-1}) ⊙ decay_inter
  S_c = γ^chunk_size · S_{c-1} + (K_c^T ⊙ D_cross) V_c
```
- Within each chunk: parallel computation (GPU-efficient training)
- Between chunks: recurrent state passing (memory-efficient inference)
- Can process in chunks of 64-256 frames during training, then switch to pure recurrent at inference

### Comparison Table: Information Propagation

| Property | Attention | SSM (Mamba) | Retention (RetNet) |
|---|---|---|---|
| **How history is stored** | Full KV-cache (lossless) | Fixed state h (lossy, learned) | Fixed state S (lossy, structured decay) |
| **Decay structure** | None (uniform access) | Learned via A matrix | Explicit exponential γ per head |
| **Multi-scale** | Must be learned | Not built-in | Built-in via multi-scale γ |
| **Inference memory** | O(seq_len × d) — grows | O(state_dim) — constant | O(d²) — constant |
| **Training** | Parallel (native) | Parallel (via conv unrolling) | Parallel (via D matrix) |
| **Streaming** | Requires chunking/cache | Native (recurrent mode) | Native (recurrent mode) |
| **Input-dependent gating** | Yes (Q,K,V from input) | Yes (A_t, B_t selective) | Partial (Q,K,V from input; γ fixed) |

### Why Retention is Interesting for Dialect ASR Specifically

**The multi-scale decay argument for dialects:**

Dialectal variation operates at multiple temporal scales simultaneously:
- **Phoneme level** (5-50ms): Consonant lenition, vowel shifts — captured by SHORT-memory heads (small γ)
- **Syllable level** (100-300ms): Stress patterns, tonal contours — captured by MEDIUM-memory heads
- **Morphological level** (500ms+): Agglutination patterns, case marker chains — captured by LONG-memory heads (large γ)

RetNet's multi-scale γ naturally decomposes these without the model having to learn the temporal hierarchy from scratch. This is a stronger inductive bias for speech than:
- Attention (no temporal structure — must learn everything)
- SSM (single-scale state dynamics — unless you manually design multi-rate processing like Zipformer)

**The dialect conditioning opportunity — γ modulation:**
Instead of using fixed γ values, make them **dialect-dependent**:
```
γ_h^d = γ_h_base + Δγ_h(e_d)
```
Where `Δγ_h(e_d) = tanh(Linear(e_d))` adjusts the decay rate per head per dialect. This means:
- For a dialect with rapid phonological changes (e.g., Telangana Telugu with fast vowel elision): decrease γ for phoneme-level heads → faster decay, more focus on local context
- For a dialect with long agglutinative patterns (e.g., Kannada with case stacking): increase γ for morphological heads → slower decay, longer memory

**This is architecturally novel**: nobody has proposed dialect-conditioned decay rates in RetNet for ASR.

---

## Proposed Paper: Architectural Comparison for Streaming Dialect ASR

### Title (working)
*"Beyond Conformers: A Systematic Evaluation of Causal Streaming Architectures for Dialect-Robust Indic ASR"*

### Core Contribution
A **controlled, apples-to-apples comparison** of streaming-first encoder architectures — SSM (Mamba-2), retention (RetNet), gated linear recurrence (RG-LRU/Hawk-style), RWKV, and hybrid (Griffin-style) — on dialect-stratified Indic ASR using RESPIN. With dialect conditioning applied uniformly across all architectures to isolate the effect of the backbone.

### Why This is Novel and Strong
1. **No such comparison exists** — existing papers compare individual architectures against Conformer baselines, but never against each other on the same task/data
2. **Retention mechanism has NEVER been applied to ASR** — only to speech enhancement. You'd be the first RetNet ASR paper
3. **Dialect-conditioned γ modulation** is a genuinely new idea — it maps linguistic insight (multi-scale dialectal variation) to architectural structure (multi-scale decay rates)
4. **Dialect stratification** as an evaluation dimension is unique — it reveals failure modes invisible in aggregate WER
5. **Streaming-first constraint** levels the playing field — you're comparing architectures at what they're actually designed for
6. **Directly positions against Sarvam**: your baselines include a causal Conformer (what Sarvam uses), and you show where non-Conformer architectures win

### Architecture Matrix

All models share:
- Same causal conv frontend (4x subsampling)
- Same CTC decoder head
- Same auxiliary DID head
- Same dialect conditioning mechanism (adapted per architecture)
- Same parameter budget (~30-50M for fair comparison)
- Same training data (RESPIN splits)

| Model Name | Encoder Backbone | Dialect Conditioning |
|---|---|---|
| **Causal-Conformer** | Causal self-attention + conv (Sarvam-style baseline) | Embedding injection |
| **DiaMamba** | Mamba-2 SSD blocks | State bias + transition modulation |
| **DiaRetNet** | Multi-scale retention blocks | Dialect-conditioned γ decay modulation |
| **DiaRWKV** | RWKV-7 blocks with 2D depthwise conv | Channel-mixing modulation |
| **DiaHawk** | RG-LRU (gated linear recurrence) | Recurrence gate modulation |
| **DiaGriffin** | RG-LRU + local windowed attention | State modulation + attention bias |
| **DiaStateformer** | MH-SSM + transformer blocks | SSM state modulation |

### Dialect Conditioning (Unified Across Architectures)

The key insight: each architecture has a different "state" mechanism. Your dialect conditioning adapts to each:

| Architecture | What gets modulated | How |
|---|---|---|
| Mamba-2 | SSM state transition A | Dialect gating on A matrix |
| RetNet | Multi-scale decay γ per head | Dialect-conditioned γ shift: `γ_h^d = γ_h + Δγ(e_d)` |
| RWKV | Token mixing / channel mixing | Dialect-conditioned decay factors |
| Hawk (RG-LRU) | Recurrence gate | Dialect-specific gate bias |
| Griffin | Recurrence gate + attention | Gate bias + dialect key/value offset |
| Causal Conformer | Attention + conv | Dialect embedding addition to Q/K |

### Experimental Design

**Data**: RESPIN-S1.0 (pick 3-4 languages: Hindi, Telugu, Kannada, Bengali)

**Evaluation axes** (this is where you beat Sarvam's analysis):

| Axis | Metrics |
|---|---|
| **Accuracy** | WER, CER, dialect-stratified WER |
| **Streaming performance** | First-token latency (ms), end-of-utterance latency |
| **Efficiency** | RTF, peak memory (MB), FLOPs per second of audio |
| **Dialect robustness** | WER variance across dialects (lower = more robust), worst-dialect WER |
| **Scaling** | Performance at 30h, 120h, 1000h per language |
| **Conditioning impact** | With vs without dialect conditioning (per architecture) |

**Key experiments**:

1. **Main comparison table**: All 6 architectures × 4 languages × {30h, 120h} splits
2. **Dialect robustness analysis**: WER per dialect heatmap, correlation between dialect distance and WER
3. **Efficiency-accuracy Pareto**: Scatter plot of WER vs RTF vs memory for all architectures
4. **Conditioning ablation**: Each architecture with and without dialect conditioning
5. **Streaming latency analysis**: Latency distribution per architecture under real-time constraint
6. **Low-resource dialect zoom-in**: Performance specifically on least-represented dialects

### What the Results Will Likely Show (Honest Assessment)

Based on the literature:

| Likely outcome | Why |
|---|---|
| Mamba-2 wins on efficiency + long utterances | Linear complexity, hardware-optimized SSD |
| **RetNet/DiaRetNet strong on dialect conditioning gains** | Multi-scale γ gives it the best inductive bias for multi-scale dialectal variation; γ modulation is the most interpretable conditioning |
| Griffin/Stateformer wins on raw WER | Local attention captures fine acoustic alignment better |
| RWKV competitive but may lag on short utterances | Token shift designed for long context |
| Hawk strong baseline but least studied | Simple architecture, fast inference |
| Causal Conformer strong but worst memory/latency | KV-cache grows with sequence |
| Dialect conditioning helps ALL architectures | But magnitude varies — biggest gain likely for recurrent/retention models |

**The publishable finding**: "Which backbone benefits most from dialect conditioning?" If recurrent architectures (Mamba/Hawk/RWKV) show larger gains from dialect conditioning than attention-based ones, you have a strong argument for **architectural affinity between sequential state models and sequential phonological processes**.

---

## How This Beats Sarvam Specifically

| Sarvam's approach | Your paper's counter |
|---|---|
| Causal Conformer (single architecture choice) | You compare 6 architectures and show Conformer isn't always optimal |
| No dialect conditioning | You show explicit dialect conditioning yields consistent gains |
| 22 languages, no dialect stratification | You evaluate at 38+ dialect granularity on RESPIN |
| Closed-source, no ablations | You provide full ablation suite — reproducible research |
| Cloud-optimized | You measure edge-deployment metrics (memory, RTF) |
| 1M hours brute force | You show that architectural choice can compensate for data scale at 30-120h |

---

## Target Journals

| Journal | Why this paper fits |
|---|---|
| **IEEE/ACM TASLP** | Systematic architecture comparison papers are their bread and butter |
| **Speech Communication** (Elsevier) | Streaming + dialect + Indic focus |
| **Computer Speech & Language** | NLP + speech intersection |
| **IEEE SPL** (4-page, fast) | If you focus on one key finding (e.g., "dialect conditioning benefits recurrent > attention") |

---

## Implementation Roadmap (10 weeks)

### Phase 1: Infrastructure (Week 1-2)
- [ ] RESPIN data pipeline, dialect-stratified splits
- [ ] Unified training framework (ESPnet2 or fairseq fork)
- [ ] Implement causal Conformer baseline (this IS the Sarvam-like baseline)

### Phase 2: Implement Backbones (Week 3-4)
- [ ] Mamba-2 SSD encoder (from mamba-ssm library)
- [ ] RWKV-7 encoder (adapt AudioRWKV)
- [ ] RG-LRU / Hawk encoder (implement from paper)
- [ ] Griffin encoder (RG-LRU + local attention)
- [ ] Stateformer (MH-SSM + transformer, from Interspeech 2023 code)

### Phase 3: Dialect Conditioning (Week 5-6)
- [ ] Implement unified dialect conditioning interface
- [ ] Adapt modulation to each backbone's state mechanism
- [ ] Train all models × conditioning variants on RESPIN 30h split

### Phase 4: Full Evaluation (Week 7-8)
- [ ] Scale to 120h split for best-performing architectures
- [ ] Full metric suite: WER/CER/latency/RTF/memory/FLOPs
- [ ] Dialect-stratified analysis, heatmaps, Pareto plots

### Phase 5: Paper (Week 9-10)
- [ ] Write up, figures, tables
- [ ] Error analysis: sample transcriptions per architecture per dialect
- [ ] Submit

---

## Honest Critique of This Plan

### Risks
1. **6 architectures is a lot of engineering** — be prepared to cut to 4 if time is tight (keep Causal Conformer, Mamba-2, Griffin, and one of RWKV/Hawk)
2. **Fair comparison is hard** — parameter count, training hyperparameters, learning rates all matter. Reviewers will attack this. Use the SAME training recipe (optimizer, LR schedule, augmentation) for all.
3. **You may not beat Conformer on WER** — and that's fine! The story is about the efficiency-accuracy tradeoff AND the dialect conditioning interaction, not about raw WER.

### What makes this publishable even if Conformer wins WER
- "Conformer gives best WER but at 3x memory cost and 2x latency"
- "Dialect conditioning yields 15% relative WER gain for Mamba but only 5% for Conformer"
- "On the lowest-resource dialect, Griffin matches Conformer at 1/3 the compute"

Any ONE of these findings is a publishable result.

---

## Key References (Additional to MD #1)

1. **Saaras v3** — Sarvam AI, 2026 (commercial, no paper — cite blog/docs)
2. **SraVaani 1.0** — IISc SPIRE Lab, arXiv:2608.08235, Aug 2026
3. **RetNet (Retentive Network)** — Sun et al., arXiv:2307.08621, 2023 (the retention mechanism paper)
4. **A Survey of Retentive Network** — arXiv:2506.06708, 2025
5. **Griffin/Hawk** — De et al., arXiv:2402.19427, 2024
6. **RWKV** — Peng et al., 2023-2025 (RWKV-7 "Goose")
7. **AudioRWKV (A-RWKV)** — 2025/2026 (2D depthwise conv + Bi-WKV)
8. **Stateformer (MH-SSM)** — arXiv:2305.12498, Interspeech 2023
9. **Nemotron Speech Streaming** — NVIDIA, 2026 (cache-aware)
10. **Moonshine v2** — sliding-window attention for edge ASR
11. **Indic DiarBench** — Sarvam AI, Interspeech 2026
12. **LAHAJA** — AI4Bharat, 2024 (Hindi accent benchmark)
13. **Voice of India** — Large-scale real-world Indic ASR benchmark
14. **Conformer** — Gulati et al., Interspeech 2020 (the baseline to beat)
15. **LRetUNet** — RetNet for speech enhancement (precedent for retention in audio)
