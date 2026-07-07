# Downstream Dataset Audit Playbook

Reusable procedure for auditing each preprocessed downstream dataset under
`<DATASET_ROOT>/<DATASET>/` against its raw
source.

**Goal (loose, do not narrow):** verify that each dataset's processed signals
and labels faithfully represent the raw recordings under our 20 Hz target
and our project conventions, and fix what does not. The named failure modes
in this doc are illustrative, not exhaustive. Stay open to other findings.

## Before You Start

- Work inside `<repo_root>`; use
  `conda activate motion`.
- Raw/processed data lives under
  `<DATASET_ROOT>/<DATASET>`.
- Current production data is `<DATASET>/processed`; write rebuilt data to
  `<DATASET>/processed_rebuild_tmp` until final approval.
- Downstream training loads files through `<repo_root>/motion_dataloader.py`; every
  rebuild must preserve loader-compatible signal filenames and the dataset's
  existing label-pairing convention.
- Split JSONs live under `inertia1/eval/splits_*`; signal-length changes mean
  dataset-specific split regeneration.
- Label mapping / keep-drop changes require user sign-off before editing
  `<repo_root>/motion_dataloader.py`.
- If the intended downstream sample uses multiple sensor streams
  (accelerometer / gyroscope / magnetometer, or multiple required
  placements), preprocessing should trim them to the **maximum common time
  overlap** and write paired outputs. Do not rely on the loader's duration
  tolerance or post-load truncation as the primary alignment mechanism.
- Follow-up after the full audit pass: revisit `motion_dataloader.py` and set
  an explicit **default placement** for each dataset based on the audited
  processed data and the intended evaluation scope. Do not leave this as an
  implicit afterthought once dataset fixes are done.

## Conventions

- `DATASET_ROOT = <DATASET_ROOT>/<DATASET>`
- Raw layout is dataset-specific (CSV/TXT/MAT/HDF5/parquet, varying schema,
  timestamp units, grouping keys). Discover it in Step 0; assume nothing.
- Processed layout: `processed/*.parquet` plus optional summary CSVs.
- Target sample rate: **20 Hz**.
- Standardized downsampling:
  - Signals: 4th-order Butterworth low-pass at **cutoff = 10 Hz** (Nyquist),
    then `scipy.signal.resample_poly(up, down)` with `up=20` and
    `down=int(round(effective_fs))`. If the filter helper guards
    `cutoff >= nyq`, it silently no-ops at 10 Hz; use `cutoff > nyq` or pass
    `9.999`. Skip filter+resample only when `effective_fs <= 20`.
  - Labels: **nearest-neighbor only** (`pandas.merge_asof` on timestamps, or
    index arithmetic). Never `scipy.signal.resample`, never linear/cubic
    interpolation, never rounding of float-resampled codes.
- Use `conda activate motion` for parquet IO.
- Never write directly into `processed/` during audit or rebuild. Write new
  artifacts to a sibling temp directory first:
  `DATASET_ROOT/processed_rebuild_tmp`. Keep the old production data in
  `DATASET_ROOT/processed` until Step 5 passes and the user explicitly
  approves promotion. If promotion is approved, archive the old directory as
  `processed_YYYYMMDD_before_<fix>` before replacing it.
- Reprocessing scripts (Step 3a/3b) must parallelize across recordings using
  **24 workers** (e.g. `joblib.Parallel(n_jobs=24)` or
  `multiprocessing.Pool(24)`). Inside each worker, also set thread caps
  (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) so 24 processes do not
  oversubscribe BLAS. Step 1's reference run on 3-5 recordings can stay
  serial.
- Trust the recording/device selection encoded in current processed
  filenames (e.g. if processed lacks a placement, that placement is out of
  scope). Trust nothing else without verification.

## Filename Contract

Processed filenames are part of the dataloader contract, not cosmetic. Do
**not** impose a unified naming scheme across datasets.

Decision rule:

- If the current naming correctly represents the rebuilt recording semantics,
  preserve it exactly.
- If the current naming is semantically wrong, change it deliberately. Example:
  a filename token says subject-only but the rebuilt unit is actually
  subject-device, or a token causes the loader to split/group by the wrong
  entity. In that case, the naming fix is part of the dataset fix, not a
  cosmetic refactor.
- Any intentional naming change requires matching updates to
  `motion_dataloader.py`, all relevant split files, and the audit notes.

When rebuilding, preserve the current pattern in that dataset's `processed/`
directory:

- Same dataset prefix/capitalization (`WISDM`, `HARTH`, `DaphnetFOG`,
  `PAMAP2`, ...), even when it differs from the folder name.
- Same subject/session token format, because `motion_dataloader.py` parses
  subjects from filenames and frozen splits index the resulting sorted sample
  order.
- Same placement/device/sensor tokens, because sensor pairing is based on
  parsed sensor type, placement, and duration.
- Same label convention:
  - sibling label per signal: `HHAR`, `WISDM`, `FoGTurning`;
  - shared label per subject/session/trial: `HARTH`, `DaphnetFOG`,
    `Opportunity`, `OdayFoG`, `WEAR`, `HAR70Plus`, `MHEALTH`, `PAMAP2`;
  - embedded labels, no separate label parquet: `Recofit`.

If you intentionally change naming, first update the relevant parser in
`motion_dataloader.py` (`parse_subject_id`, `get_sensor_type_from_filename`,
`get_placement_from_filename`, `get_label_file_for_data_file`) and regenerate
all split files for that dataset. Otherwise, keep names byte-for-byte
compatible with the existing pattern.

## Step 0. Inventory

For the dataset under audit, capture in a short notes block:

1. Raw layout: file extension + reader, schema, **per-recording grouping
   key** (e.g. `(subject)`, `(subject, device)`, file-per-recording),
   timestamp source + units, label source, documented activity vocabulary.
2. Processed layout: filename pattern, label-pairing convention, and the
   **in-scope set of placements/devices/sensors** that processed actually
   contains. Also record whether all in-scope placements/devices are available
   on all recordings, and list any placement/device-specific missing-recording
   exceptions. Copy examples from the existing `processed/` directory into the
   audit notes before writing any rebuilt files.
3. Any preprocessing script in `scripts/preprocessing/<dataset>.py`.

## Step 1. Reference-and-diff (primary path)

Decide correctness by re-running a trusted preprocessor on a few raw
examples and diffing against `processed/`. No exploratory analysis required.

1. **Reference preprocessor.** Use `scripts/preprocessing/<dataset>.py` if it
   reflects current intent; otherwise write a minimal one that follows the
   conventions above and Step 0's discovered raw schema. Fix known bugs
   (anti-alias guard, label resampling method, grouping key, encoding) so
   the reference is what you would write today.
2. **Sample.** Pick 3-5 raw recordings covering: the most common subject +
   device/placement combo, one short and one long recording, and any subject
   with unusual sampling (sparse devices, missing channels) if `ls -lS`
   makes it visible.
3. **Run** the reference into `DATASET_ROOT/processed_rebuild_tmp` or another
   clearly named temp directory. Do not overwrite `processed/`.
4. **Diff** (snippet below):
   - Signal: same row count and channel set; per-channel
     `max_abs_diff < 1e-3` or Pearson r > 0.999.
   - Label: same row count; agreement >= 0.99 after mapping both sides
     through the canonical encoding.
5. **Verdict drives the next step:**
   - All match: pipeline correct -> jump to Step 4 (loader sanity).
   - Signals match, labels do not -> Step 3a (labels-only rebuild).
   - Signals do not match -> Step 3b (full rebuild). Labels are almost
     certainly broken too in this case.

Skip the fast path only when raw is unavailable or a faithful reference is
impractical to write. In that case, fall back to the deep diagnostics
checklist below to choose between Step 3a and Step 3b.

### Deep diagnostics fallback (only if Step 1 cannot run)

Run as a checklist; any "no" condemns the corresponding side.

Signal:

- Per-recording effective rate from raw timestamps roughly matches the
  documented nominal rate.
- No multi-recording mixing inside a single processed file. Watch out for
  `processed_rows == raw_rows * 20 / nominal_rate` matching exactly across
  merged sources.
- Anti-alias cutoff `<= 10 Hz` was actually applied (not skipped by a
  `cutoff >= nyq` guard).
- No NaN/Inf, expected channel set and units.

Label:

- Vocabulary is exactly the documented activity set plus an explicit null
  token. Anything else (e.g. interpolated integer codes) is an artifact of
  Fourier-resampling categorical labels.
- `len(labels) == len(signal)`.
- Integer encoding matches `LABEL_ENCODERS[<DATASET>]` in
  `motion_dataloader.py` (beware alphabetical `Categorical.cat.codes`).
- Null/transition rows map to `-1`, not a real class index.

## Step 2. Decision matrix

| Signal | Labels | Action |
| --- | --- | --- |
| OK | OK | Step 4 (loader sanity) only. |
| OK | broken | Step 3a (labels-only rebuild). |
| broken | any | Step 3b (full rebuild). |

If the signal is broken, never just rebuild labels onto it. Rebuild signal
first.

## Step 3a. Labels-only rebuild

Wrap the per-recording loop below in a 24-worker pool (see Conventions). For
each existing signal parquet, regenerate its `..._labels.parquet`:

1. Load the raw recording it came from, using the same grouping key encoded
   in the filename.
2. Build a label time series at the raw rate from the activity column,
   mapping activity strings to the canonical integer encoding and
   null/missing to `-1`.
3. Downsample to 20 Hz **by nearest neighbor** (`pandas.merge_asof` or
   index arithmetic). Never `scipy.signal.resample` on labels.
4. Truncate or right-pad to exactly the row count of the existing signal
   parquet.
5. Write `[timestamp, label]` using the dataset's existing label-pairing
   convention from the Filename Contract. This may be a sibling label per
   signal or one shared label file per subject/session/trial.

Then propose the `motion_dataloader.py` changes (do **not** commit yet):

- `LABEL_ENCODERS[<DATASET>]` -> the canonical mapping you used.
- `LABEL_FREQUENCIES[<DATASET>]` -> sorted by descending count under the new
  labels.
- `UNWANTED_LABELS[<DATASET>]` -> documented null/transition tokens only.
- `TOP_K_LABELS[<DATASET>]` -> see Step 4 for keep/drop discussion.

**Gate: discuss with the user before applying any of these mapping changes.**
Bring the new label distribution, the proposed encoder, and the proposed
keep/drop list. Wait for explicit sign-off before editing the loader.

Do not touch existing signal parquets.

## Step 3b. Full rebuild of the in-scope subset

Reprocess **per recording** (one tuple of the grouping key from Step 0). Do
not merge across the grouping key. Run the per-recording loop in a 24-worker
pool (see Conventions).

For each recording:

1. Sort by raw timestamp; convert to seconds.
2. If raw has explicit timestamps, split into segments at gaps > ~1 s.
   Before proceeding past this point:
   - First consider whether concatenating rebuilt post-gap segments back into
     one processed recording is semantically reasonable for this dataset. For
     example, scripted HAR tasks with brief between-activity dead time may
     tolerate concatenation better than free-living or event-timing-sensitive
     datasets.
   - Then estimate the downstream window loss if you enforce strict
     segment-boundary-aware windows instead of concatenation, using the default
     audit reference setting of **30 s windows with 1 s stride**
     (`window_size = 600` at 20 Hz, `stride = 20`).
   - Discuss that tradeoff with the user before proceeding to the next rebuild
     step. Do not silently choose concatenation vs. strict boundaries when the
     choice can materially change downstream window count or semantics.
3. Within each segment, apply a 4th-order Butterworth low-pass at
   **cutoff = 10 Hz** (with the guard fix above). Skip the filter only when
   the segment rate is already <= 20 Hz.
4. Resample to a uniform 20 Hz timeline (`scipy.signal.resample_poly` or
   interpolation onto `np.arange(t0, t1, 1/20)`).
5. If multiple sensor types or required placements belong to the same
   downstream sample, trim all rebuilt streams to their **maximum common
   overlap** before writing files, so the loader sees one semantically paired
   sample instead of dropping mismatched files later.
6. Build aligned 20 Hz labels by nearest-timestamp assignment, mapping
   null/missing to `-1`.
7. Write signal and label parquets into
   `DATASET_ROOT/processed_rebuild_tmp`, **reusing the dataset's existing
   signal naming scheme and label-pairing convention**. Do not invent a new
   one.

Then propose `motion_dataloader.py` updates exactly as in Step 3a, including
the **discuss-with-user gate** before committing any label-mapping change.

## Step 4. Splits + label keep/drop

Splits live under `<repo_root>/inertia1/eval/` in
several variants (`splits_30s_stride1s`, `splits_60s_stride1s`,
`splits_5hz_stride1s`, ...). Each is `<DATASET>_splits.json` with window
indices that depend on per-recording row count, `window_size`,
`sampling_rate`, `mode`, `top_k_labels`, and `UNWANTED_LABELS`.

- **No fix needed**: nothing to do for splits. Just run the loader smoke
  test below.
- **Step 3a (labels only)**: window indices stay valid. Recompute label
  distribution per split with the new encoding. If any class is missing
  from one split or proportions shift by more than ~10 pp, regenerate
  splits across the **five standard 1 s-stride variants** for this dataset:
  `splits_1hz_stride1s`, `splits_5hz_stride1s`, `splits_10s_stride1s`,
  `splits_30s_stride1s`, and `splits_60s_stride1s`. Use the existing split
  generator (group-stratified by subject; check the `metadata` block in any
  `*_splits.json` for the strategy).
- **Step 3b (full rebuild)**: window counts and indices are invalid.
  Regenerate and replace `<DATASET>_splits.json` in each of the same five
  standard 1 s-stride split folders:
  `splits_1hz_stride1s`, `splits_5hz_stride1s`, `splits_10s_stride1s`,
  `splits_30s_stride1s`, and `splits_60s_stride1s`.

Before deciding whether to keep or replace old split files, check whether
they are still semantically intact under the rebuilt data:

- `n_windows` must equal the rebuilt loader length for the same window size,
  sampling rate, placement, stride, and mode.
- All stored indices must be in range under the rebuilt dataset.
- The old subject/grouping semantics must still match the rebuilt filenames.
  If preprocessing changed grouping (for example, from per-user to
  per-user-device) or signal duration, regenerate the split even if indices
  happen to be in range.
- Record whether placement/device availability is aligned across recordings. If
  some placements/devices are missing on a subset of recordings, note that in
  the audit output so a future default-placement switch knows to re-check split
  validity. This is an information requirement, not a directive to pre-generate
  splits for every placement/device.
- Per-split label distributions must still contain all intended classes.

Keep/drop decisions for labels:

- Default: keep every documented activity, drop only documented
  null/transition tokens via `UNWANTED_LABELS`.
- If a class has < ~50 windows in train OR is absent from val/test,
  consider dropping it.
- If the final approved keep/drop decision changes the effective encoded label
  set (for example via `UNWANTED_LABELS`, `TOP_K_LABELS`, or an updated
  `LABEL_ENCODERS[<DATASET>]` mapping), regenerate the frozen split files after
  that decision. Treat the old splits as stale even if all stored indices are
  still in range, because split stratification / label coverage is defined on
  the final encoded window labels, not just on window positions.
- After finalizing the label mapping / keep-drop decision, run the rebuilt
  dataset through `MotionDataset` with the downstream eval window settings
  (usually 30 s, target sampling rate, overlap/stride from the split config)
  and report:
  - window-level majority-label prevalence across the whole dataset and per
    split;
  - the distribution of the majority-label fraction within each window (label
    purity), e.g. quantiles / histogram buckets.
  Use this to detect transition-heavy windows, unexpectedly noisy labels, or
  a class mix that looks very different from the per-timestep prevalence.
- **Gate: bring the proposed keep/drop list to the user and wait for
  sign-off before committing.**

## Step 5. Final verification

Mandatory before announcing the dataset as audited. All checks must pass.

1. **Re-diff (Step 1) on a fresh sample.** Pick 2-3 recordings not used in
   the original Step 1 sample and rerun `diff_signal` / `diff_labels`
   against `processed_rebuild_tmp`. Both must return `ok=True`.
2. **Old-vs-new coverage comparison.** Compare current `processed/` against
   `processed_rebuild_tmp` before promotion:
   - number of signal/label files and in-scope recordings,
   - placement/device availability by recording, including any placements that
     are missing on only a subset of recordings,
   - per-sensor total rows and total 20 Hz duration,
   - raw usable span retained, after excluding intentional large gaps,
   - global and per-split label distributions,
   - valid downstream window count from the loader.
   Any apparent reduction must be explained as intentional (e.g. removing
   gap spans, fake oversampling, unpaired sensor tails, or out-of-scope
   placements). If the rebuilt data loses usable paired recordings or classes
   unexpectedly, return to Step 3 before promotion.
3. **Loader smoke test.** Instantiate the dataset via the eval pipeline
   path (see `eval_motion.sbatch`) with a small `max_files_per_dataset`.
   Confirm:
   - discovered subjects match the in-scope inventory,
   - each sample has all required sensor types and a resolved `label_file`,
   - `dataset.num_classes` matches the canonical class count,
   - one batch returns no `-1`-only windows and no NaN signal.
  The required end-to-end smoke test is a **PatchTST linear probe** run via
  `<repo_root>/scripts/eval/eval_motion.sbatch`
  on the audited dataset only for 50 epochs, using
  `CKPT_PATH=<CKPT_ROOT>/patchtst/last.ckpt`. Record the job ID
  and the resulting metric summary in the audit notes.
4. **Frozen-split check.** For every `splits_*/<DATASET>_splits.json` you
   regenerated (or kept), confirm `n_windows == len(dataset)`, all indices
   are in range, grouping semantics match the rebuilt filenames, and
   per-split label distributions look balanced.
5. **Window-label audit.** With the final label mapping and keep/drop policy,
   compute majority-vote window prevalence and majority-fraction purity under
   the eval window settings. Confirm no intended class disappears at the
   window level unexpectedly, and summarize the purity distribution so the
   user can judge whether windows are dominated by single activities or are
   transition-heavy.
6. **Visual spot-check.** Plot 30 s of signal with the corresponding label
   track for one recording. Activity transitions should align with visible
   signal changes.
7. **Promotion gate.** Only after checks 1-6 pass, ask the user whether to
   promote `processed_rebuild_tmp` to production `processed/`. Do not rename
   or delete production data without explicit approval.
8. **Sign-off log.** Append a one-line entry to the worked-example section
   below: dataset name, date, in-scope subset, fix path used (3a/3b/none),
   old-vs-new coverage/window summary, and the diff numbers from check 1.

Only after Step 5 passes do you announce the dataset as audited and fixed.

## Reusable snippet: reference-and-diff

Run in `conda activate motion`.

```python
import numpy as np
import pandas as pd
from pathlib import Path

def diff_signal(ref_path: Path, proc_path: Path, atol: float = 1e-3) -> dict:
    a = pd.read_parquet(ref_path)
    b = pd.read_parquet(proc_path)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {"ok": False, "ref_cols": list(a.columns), "proc_cols": list(b.columns),
                "ref_rows": len(a), "proc_rows": len(b)}
    diffs = a.to_numpy(np.float64) - b.to_numpy(np.float64)
    max_abs = float(np.nanmax(np.abs(diffs)))
    rs = []
    for c in a.columns:
        x = a[c].to_numpy(np.float64); y = b[c].to_numpy(np.float64)
        rs.append(float("nan") if np.std(x) == 0 or np.std(y) == 0
                  else float(np.corrcoef(x, y)[0, 1]))
    ok = max_abs <= atol or all(not np.isnan(v) and v >= 0.999 for v in rs)
    return {"ok": ok, "max_abs_diff": max_abs, "pearson_per_channel": rs, "rows": len(a)}

def diff_labels(ref_path: Path, proc_path: Path,
                proc_to_ref_map: dict | None = None) -> dict:
    a = pd.read_parquet(ref_path); b = pd.read_parquet(proc_path)
    col_a = "label" if "label" in a.columns else a.columns[-1]
    col_b = "label" if "label" in b.columns else b.columns[-1]
    la = a[col_a].to_numpy(); lb = b[col_b].to_numpy()
    if proc_to_ref_map is not None:
        lb = np.array([proc_to_ref_map.get(int(v), v) for v in lb])
    if len(la) != len(lb):
        return {"ok": False, "ref_rows": len(la), "proc_rows": len(lb)}
    agree = float((la == lb).mean())
    return {"ok": agree >= 0.99, "agreement": agree,
            "ref_unique": sorted(map(int, np.unique(la))),
            "proc_unique": sorted(map(int, np.unique(lb)))}
```

Loop over your 3-5 sample recordings and stop at the first `ok=False`. Step 2
table picks the next step from which dict failed.

## Worked example: HHAR

One concrete data point. Other datasets may share none of these failure
modes.

- Raw: four CSVs in `Activity_recognition_exp/` keyed by `(User, Device)`
  with `Creation_Time` (ns) and `gt` for activity.
- In-scope: watch only (phone processed parquets exist but are out of scope).
  Grouping key: `(User, Device, Sensor)`.
- Diff verdict: signals **broken**, labels **broken**.
  - Per-user processed row counts equal `raw_watch_rows * 20 / 100` exactly,
    so all watch devices were concatenated per user and resampled by ratio.
  - Effective raw rates per device range from ~7 Hz to ~202 Hz, so the
    `100 Hz` nominal assumption is wrong globally.
  - Anti-alias `cutoff = 10 Hz` is silently skipped by the `cutoff >= nyq`
    guard.
  - Label vocabulary contains `-2`, `6`, `7` (Fourier-resampled codes).
    Encoding looks alphabetical (`bike=0, sit=1, ...`) and contradicts
    `LABEL_ENCODERS["HHAR"]`.
- Action: Step 3b for the watch subset with cutoff = 10 Hz and guard fix,
  then split regen, then loader sanity.

## Sign-off Log

- 2026-04-29: `HHAR` audited and completed.
- 2026-04-29: `harth` audited and completed.
- 2026-04-29: `wisdm` audited and completed.
- 2026-04-29: `har70plus` audited and completed.
- 2026-04-29: `OpportunityUCIDataset` audited, promoted, and completed on the full processed subset (default eval scope remains paired `LWR`/`RWR` accelerometer), using Step `3b`; coverage `456 -> 447` signal files after dropping 9 fake missing-stream placeholders, labels `24 -> 24`, 30 s loader windows `47,626 -> 47,626`; fresh re-diff on `S2-ADL2`, `S3-ADL1`, `S4-Drill` returned signal `ok=True` and label agreement `1.0`.
- 2026-04-29: `PAMAP2_Dataset` audited, promoted, and completed on the full processed subset (12 streams per subject across `hand`/`chest`/`ankle` and `acc1`/`acc2`/`gyro`/`mag`), using Step `3b`; archived the pre-audit production directory as `processed_20260429_before_pamap2_audit` and promoted rebuilt `processed/` with `108` signal files + `9` shared label files and `38,806` valid 30 s / 20 Hz / 1 s-stride hand-accelerometer windows; fresh re-diff on `subject103`, `subject105`, and `subject107` returned signal `ok=True` for every rebuilt stream and label agreement `1.0`, and the required PatchTST linear-probe smoke test completed as job `2211330`.
- 2026-04-29: `MHEALTHDATASET` audited, promoted, and completed on the full 10-subject processed subset (default eval scope remains `rarm` accelerometer), using Step `3b`; archived the pre-audit production directory as `processed_20260429_before_mhealth_audit` and promoted rebuilt `processed/`, preserving `80 -> 80` signal files and `10 -> 10` shared label files while correcting four 20 Hz off-by-one row losses (`486,294 -> 486,298` total signal rows) and replacing null/rest with rebuilt `-1` labels under the audited 12-class encoding; the final 30 s / 20 Hz / 1 s-stride loader length is `6,822 -> 6,819`, fresh re-diff on `subject2`, `subject4`, and `subject7` returned signal `ok=True` on every rebuilt stream and label agreement `1.0`, and the required PatchTST linear-probe smoke test completed as job `2211394`.
- 2026-04-29: `wear_dataset` audited, promoted, and completed on the full 24-subject processed subset (default eval scope is now `right_arm` accelerometer), using Step `none`; archived the pre-audit production directory as `processed_20260429_before_wear_audit` and promoted rebuilt `processed/`, dropping the raw-corrupted `WEAR_S10_left_arm_acc_20Hz_3985.00s.parquet` while changing coverage `96 -> 95` signal files and preserving `24 -> 24` shared label files and `42,294 -> 42,294` valid 30 s / 20 Hz / 1 s-stride right-arm windows; fresh re-diff on `S1`, `S5`, and `S21` returned signal `ok=True` on every available stream and label agreement `>= 0.9996`, and the required PatchTST linear-probe smoke test completed as job `2211515` (`acc=0.8021`).
- 2026-04-29: `Recofit` audited and completed on the full 126-visit right-arm accel+gyro processed subset, using Step `3b`; rebuilt `processed_rebuild_tmp/` preserves `252 -> 252` signal files, paired-stream coverage `126 -> 126` visits, and total 20 Hz rows `5,718,230 -> 5,718,230`, with placement/device availability aligned across all visits (single in-scope `rightarm` placement; no missing accel/gyro exceptions), while the final approved 22-class exercise setup changes valid 30 s / 20 Hz / 1 s-stride windows `89,964 -> 89,977`; fresh re-diff on `s059v000`, `s058v001`, and `s088v000` returned signal `ok=True` and label agreement `1.0` for both sensors, and the required PatchTST linear-probe smoke test completed as job `2211851` (`acc=0.7434`, `bal_acc=0.7128`, `f1_macro=0.7282`).
- 2026-04-29: `OdayFoG` audited and completed pre-promotion on the 59-recording processed subset, using Step `3b`; rebuilt `processed_rebuild_tmp/` drops fake placeholder streams and the unmatched `subject7_v12_t2` `walklr=1` branch under the user-approved kept naming, changing coverage `1,320 -> 908` signal files while preserving `59 -> 59` shared label files and increasing label rows `101,252 -> 101,296`; the audited downstream default is now `ankle_l` accelerometer with a `40%` FoG window threshold, yielding `3,332` valid 30 s / 20 Hz / 1 s-stride windows, fresh re-diff on `subject1_v34_t2`, `subject5_v27_t1`, and `subject6_v41_t2` returned signal `ok=True` and label agreement `1.0`, and the required PatchTST linear-probe smoke test completed as job `2211891` (`acc=0.8070`, `bal_acc=0.5264`, `auroc=0.6078`).
