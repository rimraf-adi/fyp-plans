# Permutation F: Foundational Resource / Benchmark Paper [Novelty: MEDIUM]

**Working Title:** "A Regionally-Annotated Marathi ASR Benchmark for Dialect and Resource-Constrained Evaluation"

---

## What This Paper Does

Creates two reusable research artifacts that every other paper in your thesis (and other researchers' work) can build on:

1. A curated, well-characterized Marathi/Indic ASR test set with dialect and regional metadata
2. A documented, reproducible constrained-CPU evaluation harness that tests models under realistic low-end hardware conditions

This is an infrastructure paper — it doesn't propose a new model or algorithm. It builds the evaluation foundation.

## Research Question

> None of the other papers can rigorously claim dialect robustness or low-end viability without a shared, well-characterized evaluation resource. This paper is that resource.

---

## Claimed Contributions

### Contribution 1: Regionally-Annotated Test Set

- Curate a Marathi evaluation set from Vaani (and supplementary sources) with:
  - Regional/district metadata per utterance
  - Speaker demographics (age, gender where available)
  - Recording conditions (noise level, microphone type)
  - Duration and transcript quality annotations
- Add RESPIN dialect labels where available (or map RESPIN's labels onto Vaani's regional structure)
- Publish train/dev/test splits with stratification by region/dialect
- **The claim:** this is the first Marathi ASR evaluation set with systematic regional/dialect metadata attached, enabling dialect-stratified evaluation that wasn't possible before

### Contribution 2: Constrained-CPU Evaluation Harness

- A reproducible, open-source tool/script that:
  - Caps inference to N CPU threads (via `taskset`/cgroups)
  - Caps available RAM (via cgroups)
  - Runs inference through a quantized runtime (ONNX Runtime INT8)
  - Automatically measures and reports: model size (MB), peak RAM (MB), RTF, WER
- Configurable "device profiles" (e.g., "low-end Android" = 2 threads, 1GB RAM; "mid-range" = 4 threads, 2GB RAM)
- **The claim:** a standardized, reproducible way to evaluate ASR models under constrained conditions, enabling fair comparison across papers

### Contribution 3 (Optional): Baseline Numbers

- Run at least one baseline model (e.g., quantized SraVaani, or a simple pruned model) through the harness
- Report the full 4-metric table (size, RAM, RTF, WER) stratified by dialect/region
- **Purpose:** demonstrates the harness works, and gives future papers a baseline to compare against

---

## Novelty Assessment

### Contribution 1 (Annotated Test Set) — LOW TECHNICAL NOVELTY, HIGH PRACTICAL VALUE

- Dataset/benchmark papers are common in the field (LibriSpeech, Common Voice, FLEURS, etc.)
- The novelty is in the *specific language* (Marathi), *specific metadata* (dialect/region), and *specific use case* (low-resource + constrained evaluation)
- This type of paper is valuable infrastructure but rarely cited as "novel research"

**Novelty verdict:** No technical novelty. Value comes from utility and adoption.

### Contribution 2 (Evaluation Harness) — LOW NOVELTY

- Constrained evaluation via cgroups/taskset is straightforward engineering
- The value is in packaging, documenting, and standardizing it — not in the technique itself
- Similar harnesses exist for TinyML (MLPerf Tiny) but not specifically for Indic ASR

**Novelty verdict:** Useful tooling, not a research contribution.

---

## Issues & Risks

### Issue 1: "Why Should We Care?" Pushback

This is the fundamental challenge of pure resource papers. Reviewers will ask:
- "What question does this dataset answer that existing datasets don't?"
- "Who will use this harness? Is there a community waiting for it?"
- "What did you learn from creating this that I couldn't learn from reading the README?"

Without a compelling answer, the paper gets desk-rejected or receives "weak accept" at best.

### Issue 2: No Technical Contribution

Resource papers in top venues (LREC, Interspeech data tracks) are accepted based on:
- Scale (how much data)
- Quality (how carefully curated)
- Community need (is there a gap)
- Adoption potential (will others actually use it)

You need to make a strong case on all four. A small-to-medium Marathi test set is useful but not groundbreaking.

### Issue 3: Curation Quality Is Hard to Verify

How do you ensure the dialect labels are accurate? Who validates the transcripts? What's the inter-annotator agreement? Reviewers of resource papers are very particular about annotation quality.

### Issue 4: Standalone Benchmark Paper Is a Hard Sell as Your First Paper

A benchmark paper is easier to publish when:
- You already have a reputation in the field (people trust your curation)
- The benchmark enables a result that wasn't possible before (and you show it in the paper)
- There's demonstrated community demand

As a first-time author, a standalone benchmark paper is an uphill battle.

---

## Suggested Fixes

1. **Don't publish this as a standalone paper.** Instead:
   - Build the evaluation harness as part of Paper 1 (it's Contribution 3 in Paper 1)
   - Curate the test set as part of your thesis infrastructure
   - Release both as open-source artifacts alongside Paper 1 or Paper 2
   - Cite your own artifact in later papers

2. **If you do want a standalone paper**, include at least one baseline technique's full results in the paper. A benchmark that shows "here's how SraVaani performs per-dialect under constrained CPU" is far more compelling than "here's a benchmark you could run models on."

3. **Target resource-specific venues only:**
   - LREC (Language Resources and Evaluation Conference) — explicitly welcomes resource papers
   - Interspeech resource/dataset track
   - Language Resources and Evaluation journal (Springer)
   - Do NOT submit this to ICASSP or TASLP — it will be rejected as lacking technical contribution

4. **Build in parallel with Paper 1.** The evaluation harness is infrastructure you need anyway. Building it as a well-documented tool rather than a one-off script costs minimal extra effort and pays dividends across all later papers.

---

## Relationship to Other Papers

- **Infrastructure for everything else** — every other permutation benefits from a standardized evaluation harness and dialect-stratified test set
- **Paper 1's Contribution 3 IS this paper's Contribution 2** — the constrained-CPU protocol. Build once, use twice.
- **RESPIN dialect labels feed into both this paper and Permutation C** — the dialect annotation work is shared

## Target Venues

| Venue | Fit |
|---|---|
| **LREC** | Best fit — explicitly welcomes language resource papers |
| **Interspeech (resource track)** | Good fit for ASR-specific evaluation resources |
| **Language Resources and Evaluation** (Springer journal) | Journal version |

## Feasibility

- **Timeline:** ~6–10 weeks (data curation + harness engineering + baseline run + documentation + writing)
- **Hardware needed:** Minimal — CPU for evaluation, GPU only if running a baseline model
- **Data needed:** Vaani Marathi, RESPIN
- **Risk level:** LOW — this is mostly engineering and curation, nothing can "fail" experimentally
- **Recommendation:** Build it as infrastructure in parallel with Paper 1, but don't try to publish it as your first standalone paper. It's the lowest-risk work but also the lowest-reward paper. Its real value is in making all your other papers stronger.

---

## The Strategic View

Think of Permutation F not as a paper but as **force-multiplying infrastructure**:

| Without Permutation F | With Permutation F |
|---|---|
| Paper 1 evaluates on ad-hoc Marathi test set | Paper 1 evaluates on a documented, reproducible, dialect-stratified benchmark |
| Permutation C hand-rolls dialect evaluation | Permutation C uses the same benchmark with dialect splits already defined |
| Each paper builds its own constrained-CPU evaluation script | Each paper runs the same harness, results are directly comparable |
| Reviewers question evaluation rigor | Reviewers see standardized, reproducible methodology |

Build it early, even if you never publish it as its own paper.

---

## Novelty Upgrades (How to Make This Paper Stronger)

### Upgrade 6A: Automated Dialect Difficulty Scoring

For each utterance, compute a **dialect difficulty score** — how OOD is this sample relative to training data? Train a dialect classifier on RESPIN, measure its uncertainty (entropy of dialect posterior) per test utterance, correlate with ASR WER. Release difficulty scores as metadata. This turns a curation paper into a **diagnostic tool** paper.

**Why novel:** Existing benchmarks provide categorical labels (dialect A/B/C). A continuous difficulty score that predicts model failure is more useful. Nobody has done this for Indic ASR evaluation.

### Upgrade 6B: Device Personas Grounded in Market Data

Instead of arbitrary resource caps (2 threads, 1GB RAM), define device profiles based on **real market data** about which phones Marathi speakers actually use (IDC India, Counterpoint, TRAI reports on Maharashtra smartphone penetration by price segment). Define 3 device personas from real sales data, map each to cgroups config.

**Why novel:** Existing constrained evaluation uses arbitrary caps. Grounding in actual device market data for the target population is a genuine contribution.

### Upgrade 6C: Cross-Corpus Dialect Alignment Tool

Build a tool mapping RESPIN's dialect labels onto Vaani's regional metadata via acoustic similarity between dialect/region clusters. Enables using Vaani's larger data with RESPIN's labels as supervision.

**Why modest but useful:** Not major research, but a concrete artifact enabling Permutation C and making the benchmark interoperable.

### Revised Contributions After Upgrades

| # | Contribution | Novelty |
|---|---|---|
| 1 | Dialect difficulty scoring (continuous, predictive of WER) | **PARTIALLY NOVEL** |
| 2 | Market-data-grounded device evaluation personas | **NOVEL framing** |
| 3 | Cross-corpus dialect alignment tool | **Useful utility** |

**Revised risk:** LOW → LOW. Still infrastructure, but Upgrade 6A gives it a technical hook publishable at LREC or Interspeech data track.
