# RA-SCE: Risk-Aware Selective Cross-Encoder Refinement

This repository contains the experimental artifacts for the paper:

> **Risk-Aware Selective Neural Refinement for Reliable Dense Retrieval**
> Khaled Albishre, IEEE Access (Under Review, 2026)
> Manuscript ID: Access-2026-13995

## Overview

RA-SCE is a training-free, inference-time framework that selectively applies cross-encoder refinement based on per-query risk estimation. The framework occupies a distinct operating point on the effectiveness-cost-reliability trade-off, designed for latency-constrained, reliability-sensitive deployments.

**Key results:**
- ~3x reduction in cross-encoder cost vs. uniform reranking (CE-All)
- 0.0% harmed-query rate on TREC-DL'19 BM25 (vs. 9.3% for CE-All)
- 4.7% harmed-query rate on TREC-DL'19 Contriever (vs. 6.7% for CE-All)
- Statistically significant nDCG@10 gains on TREC-DL'19 (+2.97%, p=0.012)
- Zero-shot generalization across three BEIR domains (SciFact, NFCorpus, SCIDOCS)

## Repository Structure

```
ra-sce-public/
├── src/                       # Source code
│   ├── policy/
│   │   └── ra_sce_v4.py       # Core RA-SCE policy implementation
│   ├── eval/                  # Evaluation utilities
│   ├── experiments/           # Experiment runners
│   ├── main.py
│   └── metrics.py
│
├── configs/                   # Experiment configurations
│   ├── datasets.yaml
│   ├── defaults.yaml
│   └── ra_sce/                # Per-experiment configs
│       ├── dl19_safe.yaml     # Contriever + DL'19 (primary, rho=0.3)
│       ├── dl20_safe.yaml     # Contriever + DL'20 (primary, rho=0.3)
│       ├── dl19_bm25.yaml     # BM25 + DL'19 (cross-architecture)
│       ├── dl20_bm25.yaml     # BM25 + DL'20 (cross-architecture)
│       ├── dl19_medium.yaml   # rho=0.4 variant
│       └── dl19_aggressive.yaml  # rho=0.5 variant
│
├── scripts/                   # Reproduction scripts
│   ├── benchmark_ra_sce_v4.py         # Main Contriever benchmark
│   ├── benchmark_ra_sce_bm25.py       # BM25 cross-architecture
│   ├── benchmark_ra_sce_beir.py       # BEIR zero-shot
│   ├── sensitivity_analysis.py        # Table 6 coefficient sweep
│   ├── build_index.py                 # Index construction
│   └── *.sh                           # Shell orchestrators
│
├── tools/                     # Analysis utilities
│   ├── eval_pytrec.py
│   └── aggregate_sweeps.py
│
└── results/                   # Empirical results
    ├── final/                 # Clean v3 results from paper
    │   ├── bootstrap_ci_results.json
    │   ├── contriever_latency.json
    │   ├── dl19_bm25_sensitivity.json
    │   └── ...
    └── full/                  # Complete experimental artifacts
        ├── bm25_validation/
        ├── latency/
        ├── sensitivity/
        └── ...
```

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- pyserini (for BM25 baselines)
- pytrec_eval (for IR evaluation)
- sentence-transformers, transformers

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the Main Results

### Primary configuration (Contriever + MiniLM-L-6-v2 on TREC-DL'19)

```bash
python scripts/benchmark_ra_sce_v4.py --config configs/ra_sce/dl19_safe.yaml
```

### Cross-architecture validation (BM25 + MiniLM-L-12-v2)

```bash
python scripts/benchmark_ra_sce_bm25.py --config configs/ra_sce/dl19_bm25.yaml
```

### Coefficient sensitivity analysis (Table 6 in paper)

```bash
python scripts/sensitivity_analysis.py --config configs/ra_sce/dl19_bm25.yaml
```

### BEIR zero-shot generalization

```bash
python scripts/benchmark_ra_sce_beir.py --config configs/ra_sce/dl19_safe.yaml
```

### Run all experiments

```bash
bash scripts/run_all_experiments.sh
```

## Notes

- **Index building:** TREC-DL relevance judgments and the MS MARCO collection are required. Use `scripts/build_index.py` to build the BM25 index (requires pyserini).
- **Results format:** All result files are in `results/`, organized as JSON with per-query metrics.
- **Paths in configs:** Configs use relative paths from the repo root; adjust if running from a different directory.

## Citing This Work

If you use RA-SCE in your research, please cite:

```bibtex
@article{albishre2026rasce,
  title={Risk-Aware Selective Neural Refinement for Reliable Dense Retrieval},
  author={Albishre, Khaled},
  journal={IEEE Access},
  year={2026},
  note={Under review (Manuscript ID: Access-2026-13995)}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This research was supported by Umm Al-Qura University, Saudi Arabia, through grant number 26UQU4320004GSSR05.

## Contact

For questions, please open a GitHub Issue or contact via [GitHub profile](https://github.com/khaledalbishre).
