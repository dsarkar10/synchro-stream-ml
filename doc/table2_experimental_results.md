| shift_bin   |   Trials |   Mean NPS |   Std NPS |   Plasticity |   Stability |   Throughput |   Buffer Factor | Preferred Strategy        |
|:------------|---------:|-----------:|----------:|-------------:|------------:|-------------:|----------------:|:--------------------------|
| 0–0.3       |       25 |     0.6895 |    0.3889 |       0.442  |      0.746  |       0.492  |            1.48 | Buffered Linear Ingestion |
| 0.3–0.6     |       23 |     0.6235 |    0.405  |       0.5152 |      0.6674 |       0.5652 |            1.44 | Buffered Linear Ingestion |
| 0.6–0.9     |       28 |     0.5396 |    0.3998 |       0.5607 |      0.6179 |       0.6107 |            1.32 | Buffered Linear Ingestion |
| 0.9–1.2     |       34 |     0.4452 |    0.3875 |       0.6382 |      0.5353 |       0.6882 |            1.26 | High-Speed Parallel       |
| 1.2–1.5     |       34 |     0.599  |    0.3915 |       0.5412 |      0.6412 |       0.5912 |            1.39 | Buffered Linear Ingestion |
| 1.5–2.0     |       56 |     0.6199 |    0.3348 |       0.4848 |      0.7054 |       0.5348 |            1.33 | Buffered Linear Ingestion |

**Table 2: Experimental results across distribution shift bins. N=200 total trials, each with 32 samples, 10 features.**