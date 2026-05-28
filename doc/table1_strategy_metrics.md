| Strategy               | NPS Range   | Safety   |   Plasticity |   Stability |   Throughput | Memory           | Forgetting   |
|                        |             |          |              |             |              | Overhead         | Risk         |
|:-----------------------|:------------|:---------|-------------:|------------:|-------------:|:-----------------|:-------------|
| Buffered Linear + EWC  | > 0.7       | High     |         0.25 |        0.95 |         0.3  | High (3× buffer) | Low          |
| Interleaved Mini-Batch | 0.3 – 0.7   | Medium   |         0.55 |        0.65 |         0.6  | Medium           | Medium       |
| High-Speed Parallel    | < 0.3       | Low      |         0.9  |        0.25 |         0.95 | Low (1× buffer)  | High         |

**Table 1: Ingestion Strategy Comparison — Plasticity, Stability, and Throughput metrics across three strategies.**