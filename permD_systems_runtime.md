# Permutation D: Systems/Runtime — Residency, Paging, Thermal Scheduling

**Working Title:** "Resource-Adaptive Runtime Management for On-Device Speech Recognition: Residency, Paging, and Thermal-Aware Scheduling"

---

## What This Paper Does

Instead of treating the ASR model as a fixed blob loaded entirely into memory, this paper designs a *runtime manager* that dynamically controls: which parts of the model live in RAM, which are paged from disk on demand, and which compute tier to use based on the phone's current thermal state and battery level. Marathi ASR is the workload — the contribution is the scheduler, not the speech model.

## Research Question

> How should an ASR runtime manage memory residency, weight paging, and compute-tier selection dynamically, in response to live device state (RAM pressure, thermal throttling, battery), instead of a fixed configuration chosen at build time?

---

## Claimed Contributions

### Contribution 1: Two-Tier Model Residency

- A tiny voice-activity detector (VAD, ~1–5 MB) lives permanently in RAM — always listening
- The full ASR model (~50–200 MB) is only loaded into memory when the VAD detects speech
- When speech ends and a timeout expires, the ASR model is evicted from memory
- **The claim:** this reduces average RAM usage by 60–80% during idle periods while adding only ~200–500ms cold-start latency when speech begins

### Contribution 2: Memory-Mapped / On-Demand Weight Streaming

- Instead of loading the entire model checkpoint into RAM at once, memory-map the weights file and let the OS page in only the layers currently needed
- For a multi-layer encoder, early layers are accessed every frame; later layers may be accessed less frequently
- Optionally: group weights by access frequency and pin hot layers while leaving cold layers on disk
- **The claim:** peak RAM usage drops proportional to the fraction of layers actively needed at any instant

### Contribution 3: Thermal- and Battery-Aware Scheduling

- Monitor device thermal state (CPU temperature) and battery level
- When thermal throttling is imminent, switch to a smaller/cheaper model tier (from an elastic supernet) to reduce heat generation
- When battery is critically low, reduce inference frequency or switch to a "low-power" mode
- **The claim:** inference continues gracefully under thermal/power stress instead of abruptly degrading

---

## Novelty Assessment

### Contribution 1 (Two-Tier Residency) — LOW NOVELTY

This is standard practice in production ASR systems:
- Google's on-device speech recognizer uses exactly this pattern (always-on hotword detector → load full ASR)
- Apple's Siri does the same (always-on "Hey Siri" detector → full model activation)
- The concept is well-known in embedded systems literature

**Novelty verdict:** Engineering best practice, not a research contribution. You'd need to show something beyond "we implemented the obvious pattern."

### Contribution 2 (Memory-Mapped Weights) — LOW-MEDIUM NOVELTY

- `mmap` for neural network weights is used in llama.cpp and other LLM inference engines
- The specific application to ASR model weight streaming is less explored but not fundamentally new
- Layer-level access frequency analysis for pinning decisions could be mildly novel

**Novelty verdict:** Incremental. The technique exists; the application to ASR weight paging under RAM constraints is a minor contribution at best.

### Contribution 3 (Thermal/Battery Scheduling) — UNTESTABLE ⚠️

**Novelty verdict:** Potentially novel, but **completely untestable without physical hardware under thermal probes.** See Issues below.

---

## Issues & Risks

### Issue 1: THERMAL SCHEDULING CANNOT BE SIMULATED CREDIBLY (FATAL)

This is the paper's most serious problem:

- Thermal throttling behavior is device-specific, OS-specific, and highly nonlinear
- You cannot simulate it with cgroups or VM resource limits — thermal throttling involves CPU frequency scaling, GPU clock reduction, and OS-level power management that interact in complex ways
- Systems venues (MobiSys, EMSOFT) will **reject** thermal claims based on simulation
- Even reporting "simulated thermal constraints" will damage your credibility at these venues

**Verdict:** If you don't have physical devices with thermal monitoring, **do not claim thermal-aware scheduling.**

### Issue 2: No Physical Device = No Systems Paper

Systems papers are grounded in physical measurement. The reviewers at MobiSys/EMSOFT/IEEE Pervasive Computing expect:
- Real device measurements (latency, throughput, power draw, thermal traces)
- Multiple device tiers tested (not just one phone)
- Real OS-level profiling (not simulated resource caps)

Without this, the paper reads as a design document, not a research paper.

### Issue 3: Low Novelty of Individual Components

Each component (VAD → full model loading, mmap weight streaming, thermal throttling) exists independently in production systems or adjacent fields. The novelty would have to come from the *combination* and the *ASR-specific scheduling policy*, but that's a thin claim.

### Issue 4: Wrong Audience for Your Strengths

This is a systems/OS paper, not a speech/ML paper. It targets different venues, different reviewers, and different evaluation standards than your other work. You'd be competing against systems researchers who have device farms and kernel-level profiling expertise.

---

## Suggested Fixes

1. **Cut thermal-aware scheduling entirely.** A claim you can't test shouldn't be in a submitted paper, not even as a caveat. Move it to "future work" only.

2. **Scope down to residency + paging only.** Both of these *can* be measured meaningfully via cgroups/RAM caps:
   - Residency: measure cold-start latency vs. RAM savings (real trade-off, measurable in simulation)
   - Paging: measure peak RSS vs. mmap-based weight streaming (measurable via `/usr/bin/time -v`)

3. **Don't make this a standalone paper.** Fold residency + paging into Paper 1 as part of the constrained-CPU evaluation protocol, or into Permutation E (offline cascade) as part of the deployment strategy. It's not strong enough to stand alone.

4. **If you really want a systems paper:** Wait until you have physical hardware (even a Raspberry Pi). Then combine Contributions 1–2 with real device measurements and submit to IEEE Access or a workshop, not a top systems venue.

---

## Relationship to Other Papers

- **Conceptually independent** from Papers 1, B, C (different axis)
- **Could fold into Paper 1** as part of the constrained-CPU evaluation infrastructure
- **Natural companion to Permutation E** (cascade) — residency management is part of a cascade architecture
- **Save for last** in the thesis sequencing — benefits most from having real hardware

## Target Venues

| Venue | Fit | Caveat |
|---|---|---|
| **MobiSys** | Best systems venue | Requires physical device measurements — not feasible now |
| **EMSOFT** | Embedded systems | Same hardware requirement |
| **IEEE Pervasive Computing** | Broader scope | Still expects some real device data |
| **IEEE Access** | Most forgiving | Accepts simulation-based work, but impact is lower |

## Feasibility

- **Timeline:** ~10–14 weeks for the scoped-down version (residency + paging only)
- **Hardware needed:** Physical device (Raspberry Pi minimum) for credibility; GPU-only is insufficient
- **Data needed:** Any ASR model + audio — the speech model is just the workload
- **Risk level:** HIGH — low novelty, wrong venue fit for your skills, hardware dependency
- **Recommendation:** DO NOT PRIORITIZE THIS. It's the weakest permutation for your current situation. Park it as a late thesis chapter for when you have real hardware.

---

## Novelty Upgrades (How to Make This Paper Stronger)

### Upgrade 4A: Predictive Model Pre-Loading (ML-for-Systems)

Instead of reactive loading (load on VAD trigger, standard), **predict when the user will need ASR** before they speak: tiny on-device context model takes time of day, active app, recent app switches, screen state → predicts P(speech input in next 10 seconds) → pre-loads ASR model if P > threshold.

**Why novel:** Reactive model loading is standard. Predictive pre-loading based on user behavioral context is a genuine ML-for-systems contribution — unexplored for ASR.

### Upgrade 4B: Cross-Model Weight Deduplication

If multiple models live on device (VAD, ASR Stage 1, Stage 2, dialect adapter base), identify **shared weight blocks** via cosine similarity analysis. Share identical/near-identical blocks (e.g., early conv layers converge to similar Gabor filters). Report storage + RAM savings.

**Why novel:** Weight deduplication across *different models on the same device* is unexplored for ASR.

### Upgrade 4C: Hardware-in-the-Loop NAS (requires Raspberry Pi minimum)

Define a model search space, deploy candidates on physical device, measure real latency/RAM/power as NAS reward signal. Find the Pareto-optimal architecture for each device tier.

**Why novel:** HW-aware NAS exists (Once-for-All, FBNet) but not for ASR on Indic-language models on real low-end hardware.

### Revised Contributions After Upgrades

| # | Contribution | Novelty |
|---|---|---|
| 1 | Predictive model pre-loading | **NOVEL** |
| 2 | Cross-model weight deduplication | **NOVEL** |
| 3 | Hardware-in-the-loop NAS | **PARTIALLY NOVEL** |

**Revised risk:** HIGH → MEDIUM-HIGH. Predictive pre-loading is testable in simulation (with logged usage patterns). But still hardware-dependent overall. **Still don't prioritize this.**
