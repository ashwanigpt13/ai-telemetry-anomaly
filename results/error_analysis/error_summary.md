# Error Analysis Summary

Frozen benchmark results: `D:\telemetory code\ai-telemetry-anomaly\results\FinalResults\results.json`

## Error Counts

| Model | TP | TN | FP | FN | FPR | FNR | Balanced Accuracy | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical | 283 | 7120 | 803 | 49 | 0.101350 | 0.147590 | 0.875530 | 0.436560 |
| Hybrid | 332 | 178 | 7745 | 0 | 0.977534 | 0.000000 | 0.511233 | 0.030388 |

## Model Comparison

Samples corrected by hybrid: 49
Samples corrected by classical: 6942

Measured observations:
- Classical false positives: 803; hybrid false positives: 7745.
- Classical false negatives: 49; hybrid false negatives: 0.
- Classical balanced accuracy: 0.875530; hybrid balanced accuracy: 0.511233.
- Classical MCC: 0.436560; hybrid MCC: 0.030388.
