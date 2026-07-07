# Daphnet FoG Audit Notes

Date: 2026-04-29

## Inventory

- Raw files: whitespace-delimited TXT files under `<DATASET_ROOT>/daphnet_fog/dataset`, one recording per `SxxRyy.txt`.
- Raw schema: timestamp in milliseconds, ankle/thigh/trunk 3-axis acceleration in mg, and annotation.
- Raw labels: `0` outside experiment, `1` experiment/no freeze, `2` freeze.
- Rebuilt gait labels: `0 -> -1`, `1 -> 0`, `2 -> 1`.
- Processed contract: `DaphnetFOG_SxxRyy_<placement>_acc_20Hz_<duration>s.parquet` plus shared `DaphnetFOG_SxxRyy_labels.parquet`.
- In-scope placements: `ankle`, `thigh`, `trunk`; all are present for all 17 recordings.

## Audit Outcome

- Decision: Step `3b` full rebuild.
- Fixes: filtered `resample_poly` signal rebuild, nearest-neighbor label rebuild, gait passthrough labels `-1/0/1`.
- Loader: passthrough-int, `UNWANTED_LABELS['daphnet_fog'] = []`, `PRIORITY_LABELS['daphnet_fog'] = [('1', 0.2)]`, `LABEL_FREQUENCIES['daphnet_fog'] = ['0', '1']`.
- Splits: all five standard 1 s-stride split files regenerated and verified.
- Promotion: completed. Old production archived as `processed_20260429_before_daphnet_fog_fix`; rebuilt data is now `processed`.

## Verification

- Coverage: signal files `51 -> 51`, label files `17 -> 17`, total rows `599,332 -> 599,342`.
- Rebuilt timestep label counts: `-1: 242,842`, `0: 321,883`, `1: 34,617`.
- Valid 30 s / 20 Hz / 1 s-stride windows at threshold `0.20`: `17,878`, with window labels `0: 14,589`, `1: 3,289`.
- Fresh re-diff recordings: `S01R02`, `S02R02`, `S05R01`; signal `ok=True`, label agreement `1.0`.
- PatchTST linear-probe smoke test: job `2212132`, `acc=0.8138`, `bal_acc=0.6923`, `auroc=0.8855`.
