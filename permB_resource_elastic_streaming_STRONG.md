# Permutation B: Resource-Elastic Streaming ASR for Code-Switched Indic Speech [Novelty: STRONG]

**Working Title:** "Streaming Under Duress: Designing and Scheduling Streaming ASR Models for Real Constrained Hardware, with Code-Switched Indic Speech as the Stress Test"

---

## What Changed and Why

The previous version of this permutation built its novelty on akshara-level output units, FST-constrained decoding, and "temporal ACT" for emission timing. You flagged the linguistic-unit angle as unconvincing, and asked for something that's unambiguously an **ML systems** paper — model design for streaming *on constrained hardware* — rather than another streaming-ASR-with-a-clever-tokenizer paper.

I searched current literature (through mid-September 2026) before rewriting. Short version of what I found: akshara-level emission is genuinely under-explored, but the *systems* space you're asking about is not empty either — it's just empty in a specific, exploitable way. There's a lot of work on (a) making one streaming model small and fast, and separately (b) making streaming ASR multilingual/code-switch-capable. There is very little work that puts those two under the same **live, degrading hardware budget** at once. That gap is where this rewrite lives. Everything akshara-related has been removed, not softened.

---

## Why Constrained Hardware, Concretely

This isn't an abstract "let's optimize things" framing — there's a real, checkable baseline to react against. ARTPARK/IISc's **SraVaani** family is the closest existing thing to what "Paper 1" in this series presumably builds on: a ~430M-parameter FastConformer TDT/CTC model trained on the Vaani corpus, covering 63–65 Indian languages and dialects, including a dedicated streaming variant (`SraVaani-0.5-live`) that is cache-aware and exposes a single exported graph with four latency settings from ~1040ms down to 0ms look-ahead. Community ONNX/int8 ports exist specifically because the stock TorchScript + `transformers` + `sentencepiece` stack is "a heavy stack on resource-constrained platforms" (their words, describing exactly the problem this paper should target) — and even the quantized port is ~650MB on disk with real measured peak RSS in the 1.5GB range on a 9-13s test clip.

That's not a criticism of SraVaani — it's a genuinely strong model. It's the observation that motivates this paper: **the best current streaming multilingual Indic ASR model is sized for a mid-range-or-better Android phone, not for the budget devices, kiosk boards, ASHA-worker tablets, and IVR embedded systems that a large fraction of its actual target speakers use.** A systems paper that takes this model family (or an architecturally similar one) and asks "what does it take to run this well when the hardware is actively working against you — low RAM, no NPU, thermal throttling mid-call, battery-saver mode — while it's also switching between Marathi and English mid-sentence" is a different paper from both Paper 1 (compute/memory-focused compression of a single model) and the original akshara draft (linguistic unit choice).

---

## Research Question

> When a streaming, code-switched, multilingual Indic ASR system has to run on hardware whose *available* compute, memory, and power budget changes during the session (thermal throttling, battery-saver transitions, background load) — not just hardware that's fixed-but-small — what model and runtime design keeps accuracy and latency from collapsing, and how much of that is buyable with the same mechanisms designed for stable constrained hardware versus something genuinely new?

The emphasis on *changing* budget, not just small budget, is the load-bearing part of this question. Most prior work (below) profiles a device once and optimizes for that fixed point. Real phones don't hold still.

---

## Claimed Contributions

### Contribution 1: Telemetry-Driven Elastic Streaming Controller

Instead of a model tuned for one latency/compute point, or even the original draft's audio-ambiguity-driven look-ahead, this is a **runtime controller** that sits above the model and picks an operating point from a small, pre-calibrated ladder (chunk size, active encoder depth via early-exit, decoder precision) based on live device telemetry: thermal headroom, CPU governor / DVFS state, battery mode, memory pressure. The key design constraint, and the actual research contribution, is a **bounded-degradation guarantee**: given a calibration pass on a device class, the controller should be provable (or at least empirically certifiable with confidence bounds) to never let WER exceed some ceiling or emission latency exceed some ceiling, even as it downgrades — as opposed to the best-effort "graceful degradation" seen in current mobile-LLM thermal management writeups, which react to thermal state but don't tie the reaction to an accuracy contract.

This reframes the original doc's "ambiguity-adaptive look-ahead" idea: the *content*-driven signal (is this audio ambiguous?) becomes one input to the same controller alongside the *resource*-driven signal (is the device about to throttle?), rather than the sole driver of adaptivity. That fusion — jointly conditioning emission timing on acoustic ambiguity *and* hardware headroom — is the piece that's still yours to claim; each half exists separately in the literature.

### Contribution 2: Memory-Bounded Multilingual/Code-Switch Caching

Cache-aware streaming conformers (the architecture family SraVaani-0.5-live and NVIDIA's Nemotron Speech Streaming both belong to) avoid recomputation by caching encoder activations across chunks — that's the whole point of "cache-aware" streaming. What's under-characterized is what happens to that cache budget when the model also has to be multilingual and code-switch-ready: per-language conditioning state, an expanded shared vocabulary, and (if you keep any form of the original draft's code-switch gate) language-transition state all add to the *cache*, not just the weights, and cache is exactly the part of memory that's live and growing during a session — the part a fixed on-disk model-size number hides.

This contribution is a systems characterization first (how does peak RSS scale with number of active/loaded languages and with code-switch gating turned on, holding model weights fixed?) and a proposed fix second (a bounded eviction/sharing policy for per-language cache state, so peak RSS is flat or near-flat in the number of supported languages rather than growing with it).

### Contribution 3: Sharing vs. Specialization Under a Hardware Budget

There's a real, currently unresolved production-vs-academia gap here. Production systems (e.g., Gladia's June 2026 writeup on real-time multilingual code-switch ASR) have moved toward **routing between small monolingual streaming specialist models** with a language-ID gate, rather than one large shared multilingual encoder — specifically because it's cheaper on CPU-only, on-device deployments, and because their own numbers show a single large multilingual model (Voxtral-Mini-4B) losing to an ensemble of ~100M-parameter monolingual streaming Zipformers on inter-utterance code-switching. But routing isn't free: when the language-ID gate is late or wrong, you pay a **rollback cost** — buffered audio has to be re-decoded once the correct language is detected, and the user briefly sees wrong-language partials.

Nobody has published a systematic academic study of *when* routing wins over a shared multilingual encoder once you fix an explicit memory + compute + power budget and measure the actual rollback cost distribution, not just steady-state WER. That's contribution 3: take a shared cache-aware multilingual streaming encoder (sharing arm) and a routed ensemble of small monolingual streaming specialists with a bounded-cost rollback policy (specialization arm), and measure both — peak RSS, sustained throughput under thermal load, energy per hour of audio, and WER — under matched hardware budgets, specifically on Marathi-English and Hindi-English code-switched speech. This directly reuses the original draft's code-switch gate, just repositioned as the mechanism under study rather than the headline novelty.

---

## Novelty Assessment & Reranking

| Contribution | Novelty Level | Why |
|---|---|---|
| **C1: Telemetry-Driven Elastic Controller** | **STRONG** | Adaptive inference exists (LLMs, VSR), but fusing live thermal/telemetry signals with *acoustic ambiguity* for ASR emission timing, backed by a bounded-degradation guarantee, is genuinely novel. |
| **C2: Memory-Bounded Caching** | **MEDIUM-STRONG** | Isolating the *cache memory cost* (peak RSS) of multilinguality/code-switching in streaming models is under-characterized. Risk: Might just be a small effect, leading to a negative result. |
| **C3: Sharing vs. Specialization Study** | **STRONG** | The architectures aren't new, but the *systematic study under fixed hardware budgets* + measuring rollback costs during intra-utterance code-switching fills a major academic/industry gap. |

### Prior Art Actually Found (search date: September 2026)

| Work | What it does | Why it doesn't cover this paper |
|---|---|---|
| Microsoft, *Pushing the Limits of On-Device Streaming ASR* (arXiv 2604.14493, Apr 2026) | 50+ config benchmark of CPU-only streaming ASR (Whisper, Nemotron, Parakeet, Canary, Qwen3-ASR), quantization + ONNX Runtime | English only; fixed hardware profile, not live-degrading; not multilingual/code-switch |
| *Breaking Down Power Barriers in On-Device Streaming ASR* (arXiv 2402.13076) | Differentiated compression of RNN-T components (Joiner/Predictor/Encoder) by power sensitivity, 180+ trained models | Not multilingual, not live-adaptive, doesn't touch cache memory specifically |
| Spiralformer (arXiv 2510.00982), Splitformer (arXiv 2506.18035), I3D (arXiv 2303.07624) | Early-exit / layer-skipping for streaming ASR encoders on edge devices | Compute-adaptive but not resource-*telemetry*-adaptive (no live thermal/battery signal), not multilingual/code-switch focused |
| Gladia engineering writeup (Jun 2026) | Production routing between monolingual streaming Zipformers with LID gate + rollback for code-switch | Industry blog, not a controlled academic study; no explicit hardware-budget matching between routing and shared-model arms |
| Google, *A Language Agnostic Multilingual Streaming On-Device ASR System* (arXiv 2208.13916) | Single shared streaming multilingual model natively supporting intersentential code-switching | Argues the opposite architectural choice from the current production trend above; no budget-matched comparison against routing |
| NVIDIA NeMo cache-aware streaming docs / Nemotron Speech Streaming | Cache-aware Conformer, avoids recomputation, prompt-conditioned multilingual variant (40 locales) | Documents the mechanism this paper measures the *cost* of; doesn't characterize multilingual cache growth |
| Thermal-aware adaptive inference for mobile LLMs and on-device VSR (OODIn, "Adaptive Thermal Management for VSR") | Runtime engine/precision switching keyed to measured thermal state | Not ASR; no content/ambiguity fusion; heuristic ladders, no stated accuracy-degradation bound |

---

## Issues & Risks

### Issue 1: "Constrained hardware" needs a hard definition, early

The original draft's risks were about confounded ablations. This draft's biggest risk is scope drift on what "constrained" means. Pick actual target devices (e.g., a specific sub-₹15,000 Android SoC class, a Raspberry Pi 4/5 as an embedded-kiosk stand-in) and commit to them in the first month — otherwise Contribution 1's calibration ladder and Contribution 3's budget-matching have no fixed point to be measured against, and reviewers will ask "constrained relative to what?"

### Issue 2: Device-to-device variance will eat your significance

Thermal behavior, DVFS governors, and even RAM pressure differ meaningfully across phones with the "same" SoC tier. A single-device measurement (common in the papers above — Raspberry Pi, one Jetson Orin Nano) is publishable but weaker. Budget for at least 3-4 physical devices per tier if you want the sharing-vs-specialization numbers in Contribution 3 to survive review.

### Issue 3: Thermal testing is slow and physically annoying

Getting a phone to actually throttle reproducibly (not just once, but repeatably enough to calibrate a controller against) takes sustained-load runs of many minutes per trial, often in a controlled-ambient setup. This is the single biggest schedule risk in the whole proposal — budget real lab time, not just GPU-hours.

### Issue 4: Contribution 2 might be a negative result

As flagged above, the multilingual cache overhead might just be small. Do this measurement *first*, cheaply, before scoping the rest of the paper around it — if it's a few MB, fold it into Contribution 1 or 3 as a footnote rather than carrying it as a headline contribution.

### Issue 5: Code-switch data density determines whether Contribution 3 has teeth

The rollback-cost story is dramatic on intra-utterance switching and much less interesting on clean inter-utterance switching. Audit your Marathi-English / Hindi-English code-switch data for intra-utterance switch density before committing to this as the paper's central empirical result.

---

## Suggested Fixes

1. **Lock target hardware in writing before any training/calibration work starts.** Name specific SoC tiers and at least one non-phone embedded target (Raspberry Pi-class or similar), with device counts per tier.
2. **Run the Contribution 2 cache-overhead measurement as a one-week spike** before deciding whether it's a full contribution or a paragraph.
3. **Build the reproducible-throttling harness early** (sustained synthetic load + real device, controlled ambient temperature if possible) — this is infrastructure the whole paper depends on, not a late-stage detail.
4. **Audit code-switch data for intra- vs. inter-utterance switch density** before finalizing Contribution 3's framing.
5. **Decide the "bounded degradation" claim's actual strength early**: formal bound, statistical bound over a calibration set, or empirical-only. This changes what theory section (if any) the paper needs.
6. **Do a dedicated systems-venue literature pass** (MobiSys/MobiCom/SenSys/EuroMLSys proceedings directly, not just arXiv/Google Scholar) — thermal-aware ML serving is a systems-community topic and may be under-indexed in the speech-community sources this search leaned on.

---

## Relationship to Other Papers

- **Paper 1** (compute/memory axis, SraVaani compression): this permutation doesn't compress SraVaani's weights and doesn't depend on Paper 1's results. It sits one layer up — runtime scheduling and multi-model deployment strategy around a streaming Indic ASR model, whichever one Paper 1 lands on. If Paper 1 produces a compressed SraVaani variant, that variant is a natural candidate for one arm of this paper's Contribution 3 (sharing arm), not a dependency.
- **Akshara-level units, FST-constrained decoding, temporal ACT, hypothesis-stability metric** (all from the previous draft): removed entirely.
- **Code-switch gating**: survives from the previous draft, but repositioned — it's no longer pitched as a novel mechanism, it's the object being measured in Contribution 3's specialization arm.

---

## Target Venues

| Venue | Fit |
|---|---|
| **MLSys** | Best fit for the overall systems framing — model/runtime co-design under resource constraints is squarely in scope |
| **MobiSys / MobiCom / SenSys (ACM)** | Best fit specifically for Contribution 1 (telemetry-driven runtime controller) and the thermal-throttling measurement work |
| **EuroMLSys (workshop, EuroSys)** | Lower-stakes venue for an early version of Contributions 1-2 before a full MLSys submission |
| **Interspeech (Show & Tell / Industry track)** | Good fit for a systems demo of the controller + routed-vs-shared comparison, alongside a research-track submission elsewhere |

---

## Feasibility

- **Timeline:** ~20-26 weeks. Rough breakdown: 2 weeks target-hardware lock-in + Contribution 2 spike measurement, 4-6 weeks building the throttling harness and calibration pipeline, 6-8 weeks training/adapting the shared vs. specialist model arms, 4-6 weeks device-farm data collection, 4 weeks writing.
- **Hardware needed:** a small device farm (3-4 units across 2 SoC tiers, plus a Raspberry Pi 4/5 class board).
- **Data needed:** Marathi-English and Hindi-English code-switched speech with enough intra-utterance switches to make Contribution 3 interesting.
- **Risk level:** MEDIUM-HIGH. Contribution 3 is the strongest and most defensible piece and can stand alone if 1 and 2 don't pan out.
