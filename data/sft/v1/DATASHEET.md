# SFT v1 dataset (proper scale)

Memory-aware conversational SFT data: [persona card + retrieved memories + recent turns + user
turn] -> response, generated via the real retrieval pipeline (HybridRetriever, k=8) against
populated memory stores built from 40 personas disjoint from the PMB-v0 eval set
(`data/benchmarks/sft_personas_v1/`, `data/stores/sft_personas_v1/`), with target responses from
a live teacher model (gpt-5.6-luna).

- Total examples: 2480 (2232 train / 248 val)
- By kind: {"irrelevant_retrieval": 136, "memory_relevant": 2239, "abstention": 105}
- Not yet checked against PMB-v0-full for contamination — run
  `scripts/check_contamination.py --train-glob "data/sft/v1/train.jsonl"
  --eval-glob "data/benchmarks/pmb_v0_full/probes.jsonl"` before training if this matters to you.
- Personas are disjoint from the PMB-v0 eval set by construction (separate generation run,
  seed 31415, output to a separate directory) but persona NAMES may coincidentally overlap
  (shared name pool) — this does not constitute data leakage since facts/conversations differ.
- Not human-reviewed.
