# Research questions and hypotheses

## Design constraint: multi-year persona persistence

The workload this project stress-tests is a persistent AI-companion persona expected to hold
its character, relationship history, and emotional continuity across **years** of conversation
— on the order of millions of tokens of cumulative interaction, none of which can live in the
1B model's context window at once. This is why the project's central bet (RQ0/RQ1) is
externalizing that continuity into memory/retrieval/state rather than relying on parametric
capacity or brute-force context length: no context window realistically holds years of a
relationship, so the system's ability to compress, retrieve, and reconstruct the *relevant
slice* of that history per turn — accurately, without fabrication (FMR), and without drifting
persona (PCS) — is the actual product being evaluated, not raw dialogue fluency. This shapes
every benchmark and prompt set in this repo: dialogue evaluation prioritizes emotional
attunement and relationship-memory recall over generic chit-chat or trivia (see
`scripts/model_bakeoff.py`'s prompt categories).

## Primary research question

**RQ0.** To what extent can post-training, external memory, retrieval, state modeling,
distillation, and inference-time architecture recover conversational and personalized
capability in a ~1B parameter LLM relative to (a) the same model unaugmented and (b)
substantially larger unaugmented models — and how much of that recovery survives 4-bit
quantization and on-device execution?

## Secondary research questions

| ID | Question |
|---|---|
| RQ1 | Memory compensation — does external memory measurably improve personalized-recall, continuity, and consistency, and by how much? |
| RQ2 | Post-training gains — does SFT/LoRA improve interaction quality and memory utilization specifically? |
| RQ3 | Preference optimization — does DPO/ORPO reduce persona drift, and at what alignment-tax cost? |
| RQ4 | Distillation transfer — how much behavior transfers from teacher to 1B student, and where does it plateau? |
| RQ5 | Retrieval design — which strategy maximizes quality per context token? |
| RQ6 | Context economy — what is the quality-vs-context-tokens curve, and where is the peak? |
| RQ7 | Explicit state — does a state vector improve consistency beyond retrieval alone? |
| RQ8 | Reflection/consolidation — does it improve retrieval precision, or mostly introduce errors? |
| RQ9 | Quantization survival — how much of the RQ1–RQ8 gain survives INT8/Q5/Q4? |
| RQ10 | Mobile viability — achievable tok/s, TTFT, RAM, storage, thermal profile on-device? |
| RQ11 | Fundamental limits — which failure modes are not fixed by any external architecture? |
| RQ12 | Cross-over — can 1B+scaffold beat a 7–8B unaugmented model on narrow personalized-memory tasks? |

## Hypotheses

| ID | Hypothesis | Predicted effect | Null (H0) | Tested by |
|---|---|---|---|---|
| H1 | External memory improves personalized recall accuracy on multi-session dialogue | +25 to +45 pp absolute on PRA vs no-memory | Memory produces no significant change in PRA | Exp D vs A |
| H2 | External memory does not improve general reasoning or instruction following | ≤ ±2 pp on general benchmarks | Memory changes general capability significantly | Exp D vs A |
| H3 | Memory increases hallucination rate on questions whose answer is absent from memory | +3 to +10 pp false-assertion rate | No change in unanswerable-question behavior | Exp D vs A, unanswerable subset |
| H4 | Conversational SFT improves dialogue quality but not factual recall | +0.4 to +0.8 on 5-pt dialogue quality; ≤ ±2 pp factual | No change in dialogue quality | Exp B vs A |
| H5 | Memory-aware SFT beats generic SFT when evaluated with memory present | +8 to +15 pp on Memory Utilization Rate | No difference between SFT variants | Exp B1 vs B2, both with memory |
| H6 | Preference optimization improves persona consistency more than SFT alone | +0.5 to +1.0 on persona consistency score | No difference | Exp C vs B |
| H7 | Preference optimization incurs a measurable alignment tax on general capability | −1 to −4 pp on IFEval-style instruction following | No degradation | Exp C vs B |
| H8 | Teacher distillation beats hand-written SFT data at equal example count | +0.3 to +0.7 dialogue quality | No difference | Exp I vs B |
| H9 | Quality-filtered synthetic data beats unfiltered at equal post-filter size | positive but small; large at equal pre-filter size | Filtering is neutral | Exp I-filter ablation |
| H10 | Quality vs retrieved-memory-count is non-monotonic (inverted U), peaking at 4–8 memories | peak 4–8; degradation beyond ~12 | Monotonic non-decreasing | Exp Context sweep |
| H11 | Hybrid retrieval (dense + BM25 + recency + importance) beats pure dense | +5 to +12 pp retrieval precision@5 | No difference | Exp Retrieval sweep |
| H12 | Cross-encoder reranking improves precision but its latency is unjustifiable on-device | +5 to +10 pp precision; +150–500 ms latency | Latency is negligible | Exp F |
| H13 | Explicit state modeling improves cross-session consistency beyond retrieval alone | +0.3 to +0.6 consistency score | No effect | Exp G vs E |
| H14 | Reflection/consolidation improves retrieval precision but introduces fabricated memories at a measurable rate | +5 to +15 pp precision; 3–15% fabrication rate | Consolidation is lossless / no precision gain | Exp H vs G |
| H15 | Q4 quantization costs <5% on dialogue quality but >10% on structured/long-context tasks | asymmetric degradation | Uniform degradation across task types | Exp Quant |
| H16 | 1B + full scaffold beats an 8B unaugmented model on personalized-memory tasks | +15 to +35 pp PRA | No win, or the 8B wins | Exp Crossover |
| H17 | 1B + full scaffold never beats the 8B model on multi-step reasoning | 8B wins by a wide margin | 1B+scaffold closes the reasoning gap | Exp Crossover |
| H18 | Continued pretraining on companion-domain text degrades instruction following unless mixed with instruction data | −3 to −10 pp IFEval without replay | CPT is harmless | Exp CPT |
| H19 | Structured (schema) memory formatting beats prose at equal token count | +3 to +8 pp memory utilization | No difference | Exp Context format |
| H20 | Full system TTFT on-device exceeds 1.5 s at 2k context on a flagship phone | measured | TTFT under 1.5 s | Exp J |

Every `experiments/EXP-xxx/hypothesis.md` in this repo must cite the RQ and H IDs it tests,
committed **before** the run starts — git history is the pre-registration record.

**Explicit falsification commitment.** If H1 fails — if memory does not measurably improve
personalized recall — that result is reported as-is, not hidden or reframed after the fact.

## How this repo operationalizes the RQs

| RQ/H | Where it's tested |
|---|---|
| RQ1, H1–H3 | `src/onebee/memory/`, `src/onebee/retrieval/`, System A vs D in the eval grid |
| RQ2, H4–H5 | `src/onebee/training/sft.py`, System A vs B, B1 vs B2 |
| RQ3, H6–H7 | `src/onebee/training/dpo.py`, `orpo.py` |
| RQ4, H8–H9 | `src/onebee/training/distill.py`, `src/onebee/data/` filtering pipeline |
| RQ5, H11–H12 | `src/onebee/retrieval/strategies/`, `fusion.py`, `rerank.py` |
| RQ6, H10 | `src/onebee/context/budget.py`, the k-sweep in `scripts/run_matrix.py` |
| RQ7, H13 | `src/onebee/state/` |
| RQ8, H14 | `src/onebee/memory/reflection/`, `consolidation/` |
| RQ9, H15 | `configs/quantization/`, quant sweep in the eval grid |
| RQ10, H20 | `mobile/`, `benchmarks/device/` |
| RQ12, H16–H17 | Crossover experiment: 1B+scaffold vs 8B unaugmented in the eval grid |

Every experiment config under `configs/experiment/` sets an `rq_ids` / `hypothesis_ids` field
so results can be traced back to the question they answer.
