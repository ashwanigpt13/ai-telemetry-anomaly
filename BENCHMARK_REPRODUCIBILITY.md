# Frozen Benchmark Configuration

This document records the threshold calibration artifacts used by the frozen classical-vs-hybrid benchmark. It is documentation only; it does not change benchmark behavior or results.

## Threshold Calibration Policy

Each model is calibrated independently using its own training reconstruction-error distribution. The classical threshold and hybrid threshold are intentionally separate because each model produces reconstruction errors on a different scale.

Thresholds must not be recomputed unless intentionally creating a new benchmark version.

## Classical Threshold

source:

`global_stats.json`

value:

`0.6131619691848755`

selection:

95th percentile of classical training reconstruction errors.

## Hybrid Threshold

source:

`train/hybrid_global_stats.json`

compatibility path used by the frozen benchmark:

`train/global_stats.json`

value:

`0.49388029724359506`

selection:

95th percentile of hybrid training reconstruction errors.

## Provenance Note

The frozen benchmark currently loads the hybrid threshold from `train/global_stats.json` for compatibility with already-frozen outputs. The clearer provenance artifact `train/hybrid_global_stats.json` is a copy of that file with additional metadata. The numeric statistics are unchanged.

The final benchmark outputs record the historical compatibility path:

`train/global_stats.json.percentiles.p95`

This path resolves to the same threshold value as `train/hybrid_global_stats.json.percentiles.p95`.
