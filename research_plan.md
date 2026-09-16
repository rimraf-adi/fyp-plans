# Causal Streaming Dialect-Conditioned SSM for Indic ASR

A concrete research plan for a journal paper targeting dialect-robust, streaming-first ASR on the RESPIN corpus using a modified SSM architecture.

---

## The Gap (Why This Paper Can Exist)

From the literature search, here's the state of play:

| What exists | What's missing |
|---|---|
| Samba-ASR (Mamba encoder+decoder, general ASR) | No dialect conditioning in any Mamba/SSM ASR |
| ConMamba (Conv + Mamba hybrid) | No streaming-first SSM for Indic languages |
| Joint DID-ASR on RESPIN (Conformer + RoBERTa, Interspeech 2025) | Uses Conformer, not SSM; offline, not streaming |
| MLMA (Multilingual Mamba ASR) | Language-level, not dialect-level conditioning |
| BanglaTalk (Dialect-aware streaming Bengali) | Uses Whisper/LoRA, not novel architecture |
| Dialect conditioning via PRSC/PSC/embeddings | Only applied to attention-based models |

**The clear gap**: No one has built a **streaming-first, dialect-conditioned SSM** for speech recognition, let alone for Indic languages on RESPIN. This is your paper.

---

## Proposed Architecture: DiaMamba (working title)

### Core Idea

A causal, streaming ASR encoder that uses Mamba-2 blocks with dialect-aware state injection, enabling the SSM's recurrent state to be conditioned on dialect identity — making the state dynamics themselves dialect-specific.

### Why This is Principled (Not Just "Bolting Mamba On")

The key insight you need to defend:

1. **Dialectal phonological processes are sequential state transformations.** When a Bhojpuri speaker says a word differently from a Hindi speaker, the difference isn't a single token swap — it's a *chain* of phonological rules (vowel centralization → consonant lenition → nasalization) applied *sequentially* across the utterance. This is literally what an SSM state transition models: `h_t = A h_{t-1} + B x_t`.

2. **Dialect conditioning should happen at the state dynamics level, not just as input features.** Existing approaches (PRSC, embeddings) add dialect info as *input features*. But if two dialects apply *different transformation rules* to the same phonemes, you need the transition matrix A itself to vary — not just the input `B x_t`.

3. **Streaming is native to SSMs.** Conformer streaming requires lookahead hacks, chunking, and careful masking. Mamba is *inherently causal* — you get streaming for free. For a deployment-realistic Indic ASR system (agriculture/finance domains in rural India), this matters.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DiaMamba Architecture                         │
│                                                                 │
│  Audio Input (streaming chunks)                                 │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │ Conv Frontend │  Causal 1D conv subsampling (4x)             │
│  │ (Causal)      │  No future frames needed                     │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────┐                   │
│  │  Dialect-Conditioned Mamba-2 Encoder      │                  │
│  │  ┌────────────────────────────────────┐   │                  │
│  │  │ Block 1: DiaMamba Layer            │   │                  │
│  │  │  ┌──────────┐    ┌──────────────┐  │   │                  │
│  │  │  │ Mamba-2   │◄───│ Dialect State │  │   │                  │
│  │  │  │ SSM Core  │    │ Modulator    │  │   │                  │
│  │  │  └──────────┘    └──────┬───────┘  │   │                  │
│  │  │                         │          │   │                  │
│  │  │              ┌──────────┴────┐     │   │                  │
│  │  │              │ Dialect ID    │     │   │                  │
│  │  │              │ Embedding     │     │   │                  │
│  │  │              └───────────────┘     │   │                  │
│  │  └────────────────────────────────────┘   │                  │
│  │  ┌────────────────────────────────────┐   │                  │
│  │  │ Block 2: DiaMamba Layer            │   │                  │
│  │  └────────────────────────────────────┘   │                  │
│  │  ...                                      │                  │
│  │  ┌────────────────────────────────────┐   │                  │
│  │  │ Block N: DiaMamba Layer            │   │                  │
│  │  └────────────────────────────────────┘   │                  │
│  └──────────────────────────────────────────┘                   │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  CTC Head     │  → Standard text output                     │
│  └──────────────┘                                               │
│         │                                                       │
│  ┌──────────────┐  (auxiliary, multitask)                       │
│  │  DID Head     │  → Dialect classification                   │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### The "Dialect State Modulator" — Your Architectural Novelty

This is where you differentiate from Samba-ASR and ConMamba. The standard Mamba-2 selective SSM computes:

```
h_t = A_t * h_{t-1} + B_t * x_t
y_t = C_t * h_t
```

Where `A_t, B_t, C_t` are input-dependent (selective). Your modification:

#### Option 1 — State Bias Injection (simpler, safer)

```
h_t = A_t * h_{t-1} + B_t * x_t + Δ_d
```

Where `Δ_d = MLP(e_d)` is a dialect-specific state bias derived from dialect embedding `e_d`. This shifts the operating point of the state dynamics per dialect.

#### Option 2 — Transition Modulation (stronger, riskier)

```
h_t = (A_t ⊙ Γ_d) * h_{t-1} + B_t * x_t
```

Where `Γ_d = σ(Linear(e_d))` is a per-dialect gating on the state transition matrix. This makes the *dynamics themselves* dialect-dependent — different dialects literally have different state evolution rules.

#### Option 3 — Hybrid (recommended for the paper)

Use State Bias Injection in early layers (where phonetic features dominate) and Transition Modulation in deeper layers (where linguistic/dialectal patterns emerge). Ablate all three in the paper.

**Why Option 3 is strongest**: It gives you a natural ablation study (3 variants + baseline), and the depth-varying strategy has a linguistic motivation: early layers capture acoustics (universal), deeper layers capture dialect patterns (variable).

### Dialect ID Source

Three modes (ablate all):
1. **Oracle DID**: Use RESPIN's ground-truth `utt2dialect` labels — upper-bound performance
2. **Predicted DID**: Use the auxiliary DID head's output from intermediate CTC embeddings — realistic deployment scenario
3. **No DID (baseline)**: Standard Mamba-2 without dialect conditioning

---

## Experimental Design

### Data

- **RESPIN-S1.0**: 9 languages, 38+ dialects, 10,000+ hours
- Focus on **3-4 languages with strongest dialectal variation**:
  - **Hindi** (Braj, Awadhi, Bundeli, Khariboli) — most speakers, well-studied baseline gap
  - **Telugu** (Telangana, Coastal Andhra, Rayalaseema) — strong dialectal divergence
  - **Kannada** (North Karnataka, Coastal, Old Mysore) — Dravidian family contrast
  - **Bengali** (Rarh, Northern, Eastern) — MADASR 1.0 baselines available
- Use MADASR-defined splits (30h small / 120h medium) + full corpus

### Baselines (must beat or match)

| Model | Type | Reference |
|---|---|---|
| E-Branchformer (ESPnet2) | Offline, no dialect conditioning | RESPIN-S1.0 paper |
| Whisper-Large-v3 + LoRA | Offline, dialect-adapted | MADASR 2.0 submissions |
| Conformer + RoBERTa Joint DID-ASR | Offline, dialect-conditioned | Interspeech 2025 (SPIRE Lab) |
| Samba-ASR (vanilla Mamba) | Offline, no dialect conditioning | arXiv:2501.02832 |
| ConMamba | Offline, no dialect conditioning | ISCSLP 2024 |

### Your Models

| Variant | Description |
|---|---|
| DiaMamba-base | Mamba-2 encoder + CTC, no dialect conditioning (your baseline) |
| DiaMamba-bias | + State Bias Injection |
| DiaMamba-gate | + Transition Modulation |
| DiaMamba-hybrid | + Depth-varying hybrid |
| DiaMamba-hybrid-pred | + Predicted DID (not oracle) |

### Metrics

- **WER / CER** per dialect and aggregated
- **Dialect-stratified WER** (critical — show improvement on low-resource dialects specifically)
- **Streaming latency** (ms) — first-token and utterance-final
- **RTF** (Real-Time Factor) — inference speed
- **AWWER** (Agriculture Weighted WER) if using agriculture domain subset
- **Parameter count / FLOPs** comparison with Conformer baselines

### Key Ablations

1. State Bias vs. Transition Modulation vs. Hybrid (architecture contribution)
2. Oracle DID vs. Predicted DID vs. No DID (dialect conditioning contribution)
3. Layer depth where dialect modulation is applied (early / middle / late / all)
4. Number of Mamba-2 blocks (scaling behavior)
5. Comparison of causal-only vs. lookahead-augmented (streaming trade-off)

---

## RESPIN Corpus Details

### Languages & Dialects Covered

| Language | Language Family | Dialectal Scope |
|---|---|---|
| Bengali | Indo-Aryan | Western Bengali (Rarh), Northern, Eastern/delta border zones |
| Bhojpuri | Indo-Aryan (Bihari) | Standard, Western/Purvanchal, Southern/Shahabad |
| Chhattisgarhi | Indo-Aryan (Eastern Hindi) | Central, Surgujia (Northern), Halbi-influenced Southern |
| Hindi | Indo-Aryan (Central) | Braj, Awadhi, Bundeli, Khariboli zones |
| Kannada | Dravidian | North Karnataka (Dharwad/Gulbarga), Old Mysore/Southern, Coastal |
| Magahi | Indo-Aryan (Bihari) | Central (Patna/Gaya), Eastern |
| Maithili | Indo-Aryan (Bihari) | Madhubani/Darbhanga (Standard), Bajjika-influenced/Western |
| Marathi | Indo-Aryan (Southern) | Deshi/Western, Vidarbha (Varhadi), Marathwada, Konkani-influenced |
| Telugu | Dravidian | Telangana, Coastal Andhra, Rayalaseema, Uttarandhra |

### Data Volume

- **Total**: 10,000+ hours clean, 14,000+ hours total
- **Recordings**: ~7.7 million audio recordings
- **Speakers**: 18,800+ enrolled speakers across 1,500+ postal pincodes
- **Per language**: ~1,100 hours average
- **Standardized splits**: 30h (small), 120-150h (medium), 1000+h (full) per language

### Annotations Available

- `utt2lang`: Utterance → language code
- `utt2dialect`: Utterance → dialect class (Census 2011 district-mapped)
- `lexicon.txt`: Dialect-aware phonetic pronunciation lexicons
- `utt2spk` / `spk2utt`: Speaker mappings
- Speaker demographics: gender, age group, district, pincode
- Recording metadata: acoustic condition (clean/semi-noisy/noisy), duration

### Key Resources

- Dataset portal: `https://spiredatasets.ee.iisc.ac.in/respincorpus`
- Baseline code: `https://github.com/labspire/respin_baselines`
- HuggingFace: `https://huggingface.co/SpireLab` and `https://huggingface.co/RESPIN`

---

## Existing Work on RESPIN (What You're Building On)

### Core Papers

1. **RESPIN-S1.0** — Kumar et al., NeurIPS 2025 (Datasets & Benchmarks Track)
   - Baselines: TDNN-HMM, E-Branchformer, Whisper, wav2vec 2.0
   - OpenReview Forum ID: `qL8M2dOY4L`

2. **Joint DID-ASR** — Kumar, Amartyaveer, Ghosh, Interspeech 2025 (arXiv:2607.02862)
   - Conformer + Bottleneck Encoder + RoBERTa fusion
   - 81.63% DID accuracy, 17.73% WER / 4.65% CER across 33 dialects of 8 languages
   - Repo: `labspire/respin_did_interspeech25`

3. **MADASR 1.0** — ASRU 2023 (arXiv:2307.07948)
   - Bengali + Bhojpuri, ~850 hours
   - Notable submissions: TalTech (wav2vec2 + prefix tuning), Transsion (Squeezeformer + CTC)

4. **MADASR 2.0** — ASRU 2025
   - 8 languages, 33 dialects, ~1,200 hours
   - SPRING Lab IIT Madras: multi-decoder with Common Label Set (CLS)

---

## SSM/Mamba in ASR — Literature Landscape

### Key Papers

| Paper | Year | Key Contribution |
|---|---|---|
| **Samba-ASR** (arXiv:2501.02832) | 2025 | First full Mamba encoder+decoder ASR. Surpasses open-source transformer baselines. |
| **ConMamba** (ISCSLP 2024) | 2024 | Conv-augmented Mamba replacing Conformer attention. Local + global feature capture. |
| **MADEON** (SLT 2024) | 2024 | Mamba-based decoder-only ASR with speech prefixing for bidirectional processing. |
| **MLMA** | 2025 | Multilingual Mamba ASR. Language-level conditioning (not dialect-level). |
| **Speech Slytherin** | 2024 | Comprehensive Mamba evaluation across ASR, TTS, speech enhancement. |
| **SAM (Mamba-2 Audio)** (Interspeech 2026) | 2026 | Mamba-2 backbone for audio-language modeling, matches larger transformers. |
| **Mamba-2 / SSD** — Dao & Gu | 2024 | Structured State Space Duality — bridges SSMs and attention, hardware-optimized. |

### Why Mamba Fits ASR

- **Linear complexity**: O(N) vs O(N²) for transformers
- **Inherently causal**: Natural streaming without lookahead hacks
- **Long-context modeling**: Handles long-form speech efficiently
- **Recurrent inference**: Constant memory per step during streaming

### What's Missing in SSM ASR Literature

- ❌ No dialect conditioning in any SSM ASR work
- ❌ No SSM ASR on Indic languages / RESPIN
- ❌ No streaming-first SSM with auxiliary dialect identification
- ❌ No study of how SSM state dynamics relate to dialectal phonological processes

---

## Streaming ASR Landscape

### Current SOTA Streaming Architectures

| Architecture | Key Idea | Streaming Approach |
|---|---|---|
| **Zipformer** (ICLR 2024) | U-Net multi-scale encoder | Chunked attention with dynamic right-context |
| **Emformer** | Efficient memory transformer | Block processing with memory bank |
| **Streaming Conformer** | Standard conformer with causal masking | Chunk-based with lookahead |
| **Mamba/SSM** | State space models | Natively causal — no chunking needed |

### Streaming + Dialect Gap

- **BanglaTalk** (2025/2026): Dialect-aware streaming Bengali ASR, but uses Whisper+LoRA, no architectural novelty
- **FireRedASR2S** (2026): Hierarchical LID for dialect conditioning, but attention-based
- No streaming SSM with dialect conditioning exists anywhere in the literature

---

## Dialect-Robust Indic ASR — Related Work

### Hindi Dialects
- **LAHAJA** (AI4Bharat, 2024): Multi-accent benchmark, 132 speakers, 83 districts. Exposed severe WER degradation on rural accents.
- **"Dialect Matters"** (2026): Cross-lingual transfer study. Key finding: acoustic-phonetic distance predicts transfer success better than linguistic family.

### Telugu Dialects
- **Swecha Gonthuka** (Viswam AI, 2025): 15M+ audio samples across all Telangana/AP districts.
- **Multi-task DID-ASR** (Yadavalli et al., 2025/2026): Dialect ID heads conditioning decoder → 15-20% relative WER reduction.

### Kannada Dialects
- **FastConformer for Kannada** (2026): 6 dialects, phonemically balanced lexicon, 11.23% WER.

### Tamil Dialects
- **DravidianLangTech** (ACL 2024-2026): Shared tasks on dialect-based speech recognition.
- **Diglossia studies** (arXiv:2408.13739): Spoken vs literary Tamil divergence.

### Methodological Trends
1. **Gated Joint DID-ASR** instead of disjoint pre-filtering
2. **Common Label Sets (CLS)** exploiting Pan-Indian phonemic structures
3. **Parameter-efficient dialect adaptation** (LoRA, MixLoRA, prefix tuning)
4. **Beyond WER**: AWWER, semantic error rates, BRIDGE 7-metric framework

---

## Open Questions (Need Your Input)

### Q1: Which languages to focus on?
Recommend Hindi + Telugu + Kannada + Bengali (covers both language families). Even 2 languages with thorough ablations is sufficient.

### Q2: Mamba-2 vs original Mamba?
Recommend Mamba-2 — SSD framework maps cleanly to hardware, and your modifications sit naturally in it.

### Q3: Scope — ASR only, or ASR + text normalization?
**Strongly recommend ASR only.** Text normalization (mT5/IndicBART) can be a separate paper or downstream application.

### Q4: Compute situation?
Full RESPIN (10K hours) needs serious GPUs. MADASR 30h/120h splits are designed for limited compute — still publishable.

---

## Target Journals (ranked by fit)

| Journal | Impact Factor | Why it fits | Review timeline |
|---|---|---|---|
| **Speech Communication** (Elsevier) | ~3.2 | Speech + low-resource + architecture novelty | 3-4 months |
| **Computer Speech & Language** (Elsevier) | ~4.0 | Speech processing + NLP intersection | 3-5 months |
| **ACM TALLIP** | ~2.0 | Indic language focus, good for B.Tech | 2-4 months |
| **IEEE Signal Processing Letters** | ~3.9 | Short format (4 pages), fast turnaround | 1-2 months |
| **IEEE/ACM TASLP** | ~5.4 | Top-tier, ambitious but possible | 4-6 months |

**Recommendation**: Target Speech Communication or Computer Speech & Language as primary. IEEE SPL as fast-turnaround backup.

---

## Implementation Roadmap (8 weeks)

### Phase 1: Baseline (Week 1-2)
- [ ] Set up RESPIN data pipeline (use `respin_baselines` repo)
- [ ] Train vanilla Mamba-2 CTC model (DiaMamba-base)
- [ ] Reproduce at least one RESPIN baseline (E-Branchformer or Whisper-LoRA)

### Phase 2: Dialect Conditioning (Week 3-4)
- [ ] Implement Dialect State Modulator (all 3 variants)
- [ ] Train DiaMamba-bias, DiaMamba-gate, DiaMamba-hybrid
- [ ] Run full ablation suite

### Phase 3: Streaming Evaluation + DID (Week 5-6)
- [ ] Add auxiliary DID head, train DiaMamba-hybrid-pred
- [ ] Measure streaming latency, RTF
- [ ] Dialect-stratified WER analysis

### Phase 4: Paper Writing (Week 7-8)
- [ ] Write up results, error analysis
- [ ] Generate figures (WER heatmaps per dialect, latency-accuracy Pareto curves)
- [ ] Submit

---

## Key References

1. **RESPIN-S1.0** — Kumar et al., NeurIPS 2025
2. **Joint DID-ASR on RESPIN** — Kumar, Amartyaveer, Ghosh, Interspeech 2025 (arXiv:2607.02862)
3. **Mamba** — Gu & Dao, 2023
4. **Mamba-2 (SSD)** — Dao & Gu, 2024
5. **Samba-ASR** — arXiv:2501.02832 (2025)
6. **ConMamba** — ISCSLP 2024
7. **LAHAJA** — AI4Bharat, 2024
8. **MADASR 1.0** — arXiv:2307.07948, ASRU 2023
9. **MADASR 2.0** — ASRU 2025
10. **Zipformer** — ICLR 2024
11. **Speech Slytherin** — 2024
12. **SAM (Mamba-2 Audio)** — Interspeech 2026
13. **MLMA (Multilingual Mamba ASR)** — 2025
14. **BanglaTalk** — 2025/2026
15. **Swecha Gonthuka** — Viswam AI, 2025
16. **Dialect conditioning PRSC/PSC** — Hakka/Tamil ASR literature
