# Permutation B: Information-Theoretic Elastic Streaming [Novelty: STRONG]

**Working Title:** "Information-Theoretic Elastic Streaming: Hardware-Aware KV-Cache Eviction and Dynamic Compute for Edge ASR"

---

## What Changed and Why

This permutation has been completely overhauled to remove any reliance on code-switched data or linguistic-unit output (like aksharas). Instead, it represents a pure, mathematically grounded **ML Systems** paper. 

The core observation is that current edge ASR research optimizes for a *static* hardware profile (e.g., "fits in 1GB of RAM"). In the real world (especially rural India), mobile hardware degrades during a session due to thermal throttling, battery-saver states, and OS memory pressure. This paper proposes a mathematical control plane that dynamically scales both the computational depth (layers) and memory footprint (KV-cache) of a streaming ASR model based on a joint fusion of **Acoustic Entropy** and **Live Hardware Telemetry**.

---

## Research Question

> When deploying streaming ASR on low-end hardware prone to thermal throttling and memory paging, how can we mathematically fuse live OS telemetry with acoustic entropy to dynamically evict KV-cache states and modulate compute depth, thereby guaranteeing bounded latency and preventing system crashes?

---

## Claimed Contributions

### Contribution 1: Joint Hardware-Acoustic Markov Decision Process (MDP)
Current dynamic compute models (like *Splitformer*) use "early exit" strategies based solely on acoustic confidence (if the audio is clear, they exit early). They are blind to the hardware state.
*   **The Mechanism:** We formulate streaming inference as a Markov Decision Process (MDP). At every audio chunk $t$, a controller policy $\pi$ decides whether to *Exit Early* (use subset of layers), *Compute Fully* (use all layers), or *Drop Frame*. 
*   **The Math:** The reward function $J$ jointly optimizes across two decoupled signals: $S_{audio}$ (the Shannon entropy of the encoder's internal state) and $S_{hardware}$ (live OS telemetry: CPU governor state, thermal zone temp, battery state).
*   **The Claim:** By making the model *hardware-aware*, it preemptively scales down compute during simple acoustic moments to "save up" thermal headroom for complex moments, completely avoiding OS-forced arbitrary frame drops.

### Contribution 2: Telemetry-Coupled Information-Theoretic KV Cache Eviction
Cache-aware streaming ASR (e.g., *Nvidia Nemotron*) saves past Keys (K) and Values (V) to avoid recomputation, but this cache grows at $O(T)$, eventually crashing low-RAM devices. Recent LLM research (*CapKV*, *StreamingLLM*) uses Information Bottleneck (IB) principles to evict tokens, but they use a *static* memory budget.
*   **The Mechanism:** We propose **Telemetry-Modulated Information-Theoretic Eviction**. We score every KV-cache audio token by its attention entropy. However, the cache eviction threshold $B_t$ is dynamically controlled by the OS memory pressure. 
*   **The Math:** If the OS signals high memory pressure, $B_t$ shrinks drastically, and the algorithm aggressively evicts all but the absolute lowest-entropy "attention sink" acoustic frames (effectively performing feature-level Voice Activity Detection, discarding noise/silences from RAM).
*   **The Claim:** We achieve a strict, dynamically bounded $O(1)$ memory footprint that reacts to the OS, preserving long-range acoustic context without crashing the device.

### Contribution 3: The "Time-to-Throttle" (TTT) Evaluation Protocol
Standard papers measure Word Error Rate (WER) and Real-Time Factor (RTF) on a "cold" device for a 10-second clip. This fails to capture edge realities.
*   **The Mechanism:** We propose the **Time-to-Throttle (TTT)** metric. We run ASR continuously for 15+ minutes on physical constrained hardware (e.g., Raspberry Pi 4, low-end Android) and plot WER/Latency against *Time*.
*   **The Claim:** A baseline model's WER degrades catastrophically after $X$ minutes when the OS throttles the CPU. The proposed elastic model maintains a steady, bounded WER indefinitely. This standardizes how the community should evaluate edge streaming resilience.

---

## Novelty Assessment & Reranking

| Contribution | Novelty Level | Why |
|---|---|---|
| **C1: Joint MDP Controller** | **STRONG** | Fusing OS telemetry with acoustic entropy for dynamic depth in ASR is a genuinely novel control-plane formulation. |
| **C2: Telemetry-Coupled KV Eviction** | **STRONG** | LLM eviction (CapKV, H2O) uses static budgets. ASR uses chronological sliding windows. Information-theoretic eviction modulated by *live hardware states* bridges a massive systems gap. |
| **C3: Time-to-Throttle Metric** | **MEDIUM-STRONG** | Systems venues reward papers that expose flawed standard metrics and propose rigorous real-world evaluation methodologies. |

### Prior Art Actually Found (search date: September 2026)

| Work | What it does | The Research Gap Exploited |
|---|---|---|
| *CapKV (2026)*, *SnapKV (2024)*, *H2O* | Information Bottleneck and semantic KV-cache eviction for LLMs. | Operates on text, uses a static memory budget. Does not dynamically scale via hardware telemetry. |
| *Nemotron Speech (2024/2025)* | Cache-aware Conformer streaming ASR. | Uses standard cache chunking without rigorous entropy-based internal eviction. Susceptible to $O(T)$ memory bloat. |
| *Splitformer (2025)*, *TCUQ (2026)* | Early-exit / uncertainty-quantified ASR. | Modulates compute based strictly on audio confidence, completely blind to underlying thermal/hardware states. |

---

## Issues & Risks

### Issue 1: Thermal Testing is Slow and Physically Annoying
Getting a phone/board to throttle reproducibly (not just once, but repeatably enough to calibrate a controller against) takes sustained-load runs of many minutes per trial in a temperature-controlled ambient setup. This is the biggest schedule risk.

### Issue 2: Device-to-Device Variance
Thermal behavior, DVFS governors, and RAM pressure differ across phones even within the same SoC tier. Budget for at least 2-3 physical devices (e.g., one Raspberry Pi, one budget Android) to prove the telemetry hooks generalize.

### Issue 3: OS Telemetry Access
Accessing raw thermal zones (`/sys/class/thermal/`) or memory pressure states requires specific OS permissions. On Android, this might require ADB access or a custom profiling app wrapper, which is engineering overhead.

---

## Suggested Fixes

1. **Build the reproducible-throttling harness early:** Set up a Raspberry Pi with synthetic background workloads to force CPU/memory stress artificially. This allows you to test the MDP controller rapidly without waiting for actual room-temperature overheating.
2. **Abstract the Telemetry API:** Write a simple wrapper that outputs normalized values (0.0 to 1.0) for Thermal, Compute, and Memory pressure. This decouples the MDP math from the messy OS-specific polling.
3. **Lock down the Information Bottleneck math first:** Before writing any Android code, prove mathematically (in Python/PyTorch) that evicting high-entropy Conformer KV states does not hurt WER on the Vaani test set.

---

## Target Venues

| Venue | Fit |
|---|---|
| **MLSys** | Best fit. Model/runtime co-design, caching algorithms, and hardware-constrained ML are the exact focus. |
| **MobiSys / MobiCom / SenSys** | Excellent fit for the telemetry controller, thermal-throttling measurement (TTT metric), and on-device deployment. |
| **ICASSP / Interspeech** | Good fit if the focus is weighted heavily toward the mathematical derivation of the Information-Theoretic cache eviction on audio. |

---

## Feasibility

- **Timeline:** ~16-20 weeks. (4 weeks mathematical formulation & Python simulation of KV eviction; 4-6 weeks building the telemetry harness/MDP controller; 4 weeks physical device benchmarking; 4 weeks writing).
- **Hardware needed:** Standard GPU for training the baseline ASR. Physical edge devices (Raspberry Pi 4/5, budget Android phone) for the TTT evaluation. 
- **Data needed:** Standard monolingual Marathi speech (Vaani or SraVaani subsets). No code-switched or dialect-specific data required.
- **Risk level:** MEDIUM. The math is sound, and the problem is real. The primary risk is pure software engineering: reliably polling OS telemetry and hooking it into the inference loop in real-time.
