# Verified Empirical Results

Summary of empirical results reported in the paper. Raw data in `results/final/` (clean v3) and `results/full/` (complete).

## TREC-DL'19 (n=43 queries)

### Contriever + MiniLM-L-6-v2 (primary)

| Method      | nDCG@10    | MRR        | Harmed % | Gated % |
|-------------|------------|------------|----------|---------|
| Base@k0     | 0.5634     | 0.8245     | -        | -       |
| CE-All      | 0.6141     | 0.8910     | 6.7%     | 100%    |
| Random-30%  | 0.5807     | 0.8650     | ~2.5%    | 30%     |
| **RA-SCE**  | **0.5801** | **0.8672** | **4.7%** | **30.2%** |

**Statistical test:** p=0.012 (paired t-test), Cohen's d=0.41 (small-to-medium)
**Holm-Bonferroni corrected:** significant

### BM25 + MiniLM-L-12-v2 (cross-architecture)

| Method      | nDCG@10    | MRR        | Harmed % | Gated % |
|-------------|------------|------------|----------|---------|
| Base@k0     | 0.5058     | 0.7400     | -        | -       |
| CE-All      | 0.7003     | 0.9200     | 9.3%     | 100%    |
| **RA-SCE**  | **0.5382** | **0.7892** | **0.0%** | **32.6%** |

## TREC-DL'20 (n=54 queries)

### Contriever + MiniLM-L-6-v2

| Method      | nDCG@10    | MRR        | Harmed %  | Gated % |
|-------------|------------|------------|-----------|---------|
| Base@k0     | 0.5661     | 0.8123     | -         | -       |
| CE-All      | 0.6092     | 0.8845     | 5.6%      | 100%    |
| **RA-SCE**  | **0.5693** | **0.8456** | **1.85%** | **29.6%** |

**Statistical test:** p=0.16 (n.s.), Cohen's d=0.09 (negligible)

## Latency Measurements (RTX 4090)

| Configuration               | CE-All Mean (ms) | P95 (ms) | RA-SCE Effective (ms) | Speedup |
|-----------------------------|------------------|----------|------------------------|---------|
| BM25 + MiniLM-L-12-v2       | 32.11            | 38.4     | 10.4                   | 3.07x   |
| Contriever + MiniLM-L-6-v2  | 11.92            | 12.5     | 3.6                    | 3.33x   |

## Coefficient Sensitivity (Table 6)

Nine coefficient combinations from (0.10, 0.01) to (1.0, 0.20):
- **DL'19:** nDCG@10 spread = 0.0048 (excluding gap-only ablation)
- **DL'20:** nDCG@10 spread = 0.0011
- **Harmed rates:** below 6% throughout the tested range
- **Default:** w_ent=0.25, w_disp=0.05

## BEIR Zero-Shot Generalization

| Dataset    | Base nDCG@10 | RA-SCE nDCG@10 | Harmed % | p-value |
|------------|--------------|----------------|----------|---------|
| SciFact    | 0.6791       | 0.6791         | 0.3%     | 1.000   |
| NFCorpus   | 0.3318       | 0.3520         | 4.0%     | <0.001  |
| SCIDOCS    | 0.1620       | 0.1738         | 4.4%     | 0.001   |

## Data Files

Raw results available in:
- `results/final/` - clean results from final paper revision (v3)
- `results/full/` - complete experimental artifacts including ablations
