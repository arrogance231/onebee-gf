# SFT v0 dataset

Memory-aware conversational SFT data: [persona card + retrieved memories + recent turns + user
turn] -> response, generated via the real retrieval pipeline (HybridRetriever, k=8) against
populated memory stores built from 4 personas disjoint from the PMB-v0 eval set
(`data/benchmarks/sft_personas_v0/`, `data/stores/sft_personas_v0/`), with target responses from
a live teacher model (gpt-5.6-luna).

- Total examples: 225 (202 train / 23 val)
- By kind: {"irrelevant_retrieval": 1, "memory_relevant": 223, "abstention": 1}
- Not yet checked against PMB-v0-full for contamination — run
  `scripts/check_contamination.py --train-glob "data/sft/v0/train.jsonl"
  --eval-glob "data/benchmarks/pmb_v0_full/probes.jsonl"` before training if this matters to you.
- Personas are disjoint from the PMB-v0 eval set by construction (separate generation run,
  seed 9999, output to a separate directory) but persona NAMES may coincidentally overlap
  (shared name pool) — this does not constitute data leakage since facts/conversations differ.
- Not human-reviewed.
