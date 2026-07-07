# Downstream Dataset Audit Playbook (Gait Datasets)

Gait-specific variant of `DATASET_AUDIT_PLAYBOOK.md`. The overall pipeline
(Steps 0-5, the reference-and-diff snippet, the promotion gate, the
sign-off log) is identical; this file only adds the gait-specific
overrides. Defer to the general playbook for anything not explicitly
covered here.

Scope: any dataset whose downstream task is a binary, time-resolved gait
event (e.g. freezing of gait, fall risk, on/off, freezing-vs-walking).
Examples in the repo: `OdayFoG`, `daphnet_fog`, `FoGTurning`. Raw I/O is
dataset-specific; assume nothing about the container.

**Goal:** verify that each gait dataset's processed signals and labels
faithfully represent the raw recordings under our 20 Hz target, our
project conventions, and the gait-specific label semantics (binary event
detection with percentage thresholding instead of majority voting).

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

Gait-specific defaults that override the general playbook:

- Raw timestep labels live in `{0, 1}`; null/missing/transition rows
  must map to `-1`. No other integer codes are allowed.
- Gait datasets stay on the loader's **passthrough_int** path: do not
  add an entry to `LABEL_ENCODERS`. Rebuilt `[timestamp, label]`
  parquets carry `{-1, 0, 1}` directly. As a consequence,
  `dataset.num_classes` is `None` after construction; the downstream
  eval infers `num_classes = 2` from observed labels.
- Window labels come from
  `PRIORITY_LABELS[<DATASET>] = [('1', <threshold>)]`, not majority
  voting. Default reference threshold is `0.40`; every gait audit
  must run a `{0.20, 0.40, 0.50}` sweep before locking it (see Step 4).
- Pick the audited default placement from the placements most
  informative for the gait event (typically `ankle_l` / `ankle_r` /
  `lumbar` / `foot_*` for FoG-style tasks).

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

- Same dataset prefix/capitalization (`OdayFoG`, `DaphnetFOG`,
  `FoGTurning`, ...), even when it differs from the folder name.
- Same subject/session token format, because `motion_dataloader.py` parses
  subjects from filenames and frozen splits index the resulting sorted sample
  order.
- Same placement/device/sensor tokens, because sensor pairing is based on
  parsed sensor type, placement, and duration.
- Same label convention. For gait datasets the conventions seen so far are:
  - sibling label per signal: `FoGTurning`;
  - shared label per subject/session/trial: `daphnet_fog`, `OdayFoG`.
  Other gait datasets may use either pattern; preserve whichever the
  current `processed/` directory uses. Embedded-label gait datasets are
  also possible but follow the general playbook's embedded-label rules.

If you intentionally change naming, first update the relevant parser in
`motion_dataloader.py` (`parse_subject_id`, `get_sensor_type_from_filename`,
`get_placement_from_filename`, `get_label_file_for_data_file`) and regenerate
all split files for that dataset. Otherwise, keep names byte-for-byte
compatible with the existing pattern.

## Step 0. Inventory

For the dataset under audit, capture in a short notes block:

1. Raw layout: file extension + reader, schema, **per-recording grouping
   key** (e.g. `(subject)`, `(subject, visit, walk, trial)`,
   file-per-recording), timestamp source + units, label source, and the
   exact gait-event label column (e.g. `freeze_label`, `fog`, `gait_event`).
   Confirm the label is binary-coded at the raw timestep with an explicit
   null/missing token; if it is encoded as multi-class, decide and document
   the binarization rule (which raw codes collapse to `1` vs `0`) before
   touching anything.
2. Processed layout: filename pattern, label-pairing convention, and the
   **in-scope set of placements/devices/sensors** that processed actually
   contains. Also record whether all in-scope placements/devices are available
   on all recordings, and list any placement/device-specific missing-recording
   exceptions. Copy examples from the existing `processed/` directory into the
   audit notes before writing any rebuilt files.
3. Filename token semantics. Gait datasets often encode several orthogonal
   keys in the raw filename (e.g. `subject`, `visit`, `walk_direction`,
   `trial`). Inventory which of those keys the current processed naming
   actually preserves vs collapses, and check whether collapsed keys cause
   distinct raw recordings to land on the same processed `record_id`. If
   they do, treat that as a real audit finding (see Step 3b notes).
4. Any preprocessing script in `scripts/preprocessing/<dataset>.py`.

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
   - Label: same row count; agreement >= 0.99 on the raw `{-1, 0, 1}`
     integers (passthrough, no encoder mapping).
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

- Vocabulary is exactly `{0, 1}` plus the explicit null token (`-1`).
  Anything else (floats, interpolated integers like `2`/`-2`, or stray raw
  codes) is an artifact of Fourier-resampling or of binarization being done
  inconsistently per recording.
- `len(labels) == len(signal)`.
- The dataset is on the loader's **passthrough_int** path: there must be
  **no** entry in `LABEL_ENCODERS[<DATASET>]`. Verify only that
  `LABEL_FREQUENCIES[<DATASET>]` is exactly `['0', '1']` or `['1', '0']`
  (sorted by descending count under the rebuilt labels),
  `UNWANTED_LABELS[<DATASET>] == []`, `TOP_K_LABELS[<DATASET>]` is `None`,
  and `PRIORITY_LABELS[<DATASET>] == [('1', <threshold>)]` with the
  audited threshold from Step 4.
- Null/transition rows map to `-1`, not to `0` and not to `1`.

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
2. Build a label time series at the raw rate from the gait-event column
   identified in Step 0, applying the documented binarization rule to
   produce integer values in `{0, 1}` and mapping null/missing to `-1`.
   Do not introduce any other integer codes; the loader expects
   passthrough_int over `{-1, 0, 1}`.
3. Downsample to 20 Hz **by nearest neighbor** (`pandas.merge_asof` or
   index arithmetic). Never `scipy.signal.resample` on labels.
4. Truncate or right-pad to exactly the row count of the existing signal
   parquet.
5. Write `[timestamp, label]` using the dataset's existing label-pairing
   convention from the Filename Contract. This may be a sibling label per
   signal or one shared label file per subject/session/trial.

Then propose the `motion_dataloader.py` changes (do **not** commit yet). The
gait loader surface is narrow:

- `LABEL_ENCODERS[<DATASET>]` -> not present (passthrough_int).
- `LABEL_FREQUENCIES[<DATASET>]` -> `['0', '1']` or `['1', '0']`, sorted
  by descending count under the rebuilt labels.
- `UNWANTED_LABELS[<DATASET>]` -> `[]`. `-1` is already handled by the
  loader's null-mask path.
- `TOP_K_LABELS[<DATASET>]` -> `None`.
- `PRIORITY_LABELS[<DATASET>] = [('1', <threshold>)]` — the threshold
  itself is chosen and signed off in Step 4; carry forward whatever is
  already there until then.

**Gate: bring the rebuilt label distribution to the user before editing
the loader.** Threshold and default-placement sign-off happen in Step 4.

Do not touch existing signal parquets.

## Step 3b. Full rebuild of the in-scope subset

Reprocess **per recording** (one tuple of the grouping key from Step 0). Do
not merge across the grouping key. Run the per-recording loop in a 24-worker
pool (see Conventions).

For each recording:

1. Sort by raw timestamp; convert to seconds.
2. If raw has explicit timestamps, split into segments at gaps > ~1 s.
   Before proceeding past this point:
   - Gait events are time-resolved, so the default for this playbook is
     **strict segment-boundary-aware windows**: do not concatenate
     post-gap segments unless you can argue the gait-event labeling
     stays semantically intact across the gap. Concatenation should be
     the rare exception, not the default.
   - Estimate the downstream window loss if you enforce strict
     segment-boundary-aware windows, using the default audit reference
     setting of **30 s windows with 1 s stride** (`window_size = 600`
     at 20 Hz, `stride = 20`).
   - If you are tempted to concatenate anyway, surface the concrete
     window-count tradeoff to the user first. Do not silently choose
     concatenation vs. strict boundaries when the choice can materially
     change downstream window count or event-timing semantics.
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
8. Honor only the sensor streams that the raw recording actually contains.
   If the existing `processed/` directory holds placeholder streams that
   are uniformly null/zero/all-NaN (a common gait-dataset failure mode
   where every advertised placement gets a parquet whether the raw IMU
   was worn or not), drop them in the rebuild and explicitly call out the
   coverage delta in Step 5.2.
9. Filename-token collisions: if Step 0 found that the existing processed
   `record_id` collapses two distinct raw recordings (e.g. omitting a
   `walk_direction` / `trial` token), surface the collision to the user
   before rebuilding. Two valid resolutions are (a) introduce the
   missing token and produce two rebuilt files, or (b) keep the existing
   naming and pick the single raw branch whose row count matches the
   currently processed file. Either way it is a user-gated decision; do
   not silently pick one.

Then propose `motion_dataloader.py` updates as in Step 3a. The rebuilt
signal coverage will usually also move `available_sensors`,
`placement_patterns`, and `default_placement` in the dataset's
`DatasetConfig`; bring those to the same Step 4 gate.

## Step 4. Splits + label keep/drop

Splits live under `<repo_root>/inertia1/eval/` in
several variants (`splits_30s_stride1s`, `splits_60s_stride1s`,
`splits_5hz_stride1s`, ...). Each is `<DATASET>_splits.json` with window
indices that depend on per-recording row count, `window_size`,
`sampling_rate`, `mode`, `top_k_labels`, and `UNWANTED_LABELS`. For gait
datasets `top_k_labels` is `None` and `UNWANTED_LABELS` is `[]`, so the
split shape is driven entirely by row count, window size, sampling
rate, and the `PRIORITY_LABELS` threshold chosen below.

- **No fix needed**: nothing to do for splits. Just run the loader smoke
  test below.
- **Step 3a (labels only)**: window indices stay valid. Recompute the
  per-split positive-window fraction at the existing `PRIORITY_LABELS`
  threshold. If any split loses class `0` or class `1` entirely, or the
  positive fraction shifts by more than ~10 pp on any split, regenerate
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
- Per-split window labels must still contain both `0` and `1` after
  applying the chosen `PRIORITY_LABELS` threshold.

Window-label policy:

- Both classes are the task — never prune `0` or `1`. The only "drop"
  is the loader's `max_unwanted_frac` cap on per-window `-1` count,
  which decides window validity. Negative-only windows always survive
  and become `0`.
- The window label comes from `_get_window_label`: if the fraction of
  `1`s in the window meets the `PRIORITY_LABELS` threshold, the window
  is labeled `1`; otherwise the loader majority-votes over non-`-1`
  samples, which for gait collapses to `0`.
- **Threshold sweep (mandatory).** Run the rebuilt dataset through
  `MotionDataset` at 30 s / 20 Hz / 1 s stride on the audited default
  placement, for each of `{0.20, 0.40, 0.50}`, and report per
  candidate:
  - global positive-window fraction,
  - per-subject positive-window fraction (call out subjects with zero
    positives at this threshold),
  - per-split positive-window fraction across the five standard
    1 s-stride split variants,
  - total valid window count after `_get_window_label` runs.
  Avoid thresholds that wipe out positives in any test subject.
- **Gate: bring the sweep table to the user and wait for an explicit
  threshold choice (and default-placement choice if not yet
  committed).** After the threshold is committed in `PRIORITY_LABELS`,
  regenerate all five 1 s-stride split files; treat prior splits as
  stale even if their indices are in range, because stratification
  depends on the new window labels.

## Step 5. Final verification

Mandatory before announcing the dataset as audited. All checks must pass.

1. **Re-diff (Step 1) on a fresh sample.** Pick 2-3 recordings not used in
   the original Step 1 sample and rerun `diff_signal` / `diff_labels`
   against `processed_rebuild_tmp`. Both must return `ok=True`.
2. **Old-vs-new coverage comparison.** Compare current `processed/` against
   `processed_rebuild_tmp` before promotion:
   - number of signal/label files and in-scope recordings,
   - placement/device availability by recording, including any placements
     that are missing on only a subset of recordings (call out
     placeholder streams that the rebuild dropped, per Step 3b sub-step
     8),
   - per-sensor total rows and total 20 Hz duration,
   - raw usable span retained, after excluding intentional large gaps,
   - raw-timestep positive prevalence (fraction of `1`s in `{0, 1}`
     after the null mask), globally and per recording,
   - global and per-split positive-window prevalence at the chosen
     `PRIORITY_LABELS` threshold,
   - valid downstream window count from the loader.
   Any apparent reduction must be explained as intentional (e.g. removing
   gap spans, dropped placeholder streams, unpaired sensor tails, or
   out-of-scope placements). If the rebuilt data loses usable paired
   recordings or drops a class to zero in any split, return to Step 3
   before promotion.
3. **Loader smoke test.** Instantiate the dataset via the eval pipeline
   path (see `eval_motion.sbatch`) with a small `max_files_per_dataset`.
   Confirm:
   - discovered subjects match the in-scope inventory,
   - each sample has all required sensor types and a resolved `label_file`,
   - `dataset.num_classes is None` (expected under passthrough_int) and
     the observed window-label set is exactly `{0, 1}`,
   - one batch returns no `-1`-only windows and no NaN signal.
  Then run the required end-to-end **PatchTST linear probe** via
  `<repo_root>/scripts/eval/eval_motion.sbatch`
  on the audited dataset only for 50 epochs, using
  `CKPT_PATH=<CKPT_ROOT>/patchtst/last.ckpt`. Record
  the job ID and metric summary in the audit notes.
4. **Frozen-split check.** For every `splits_*/<DATASET>_splits.json` you
   regenerated (or kept), confirm `n_windows == len(dataset)`, all
   indices are in range, grouping semantics match the rebuilt filenames,
   and each split contains both window classes (`0` and `1`) with a
   non-trivial positive prevalence at the chosen `PRIORITY_LABELS`
   threshold. Gait datasets are intrinsically imbalanced, so do not
   require equal-class balance; just ensure neither class is empty.
5. **Window-label audit.** With the final `PRIORITY_LABELS` threshold,
   recompute global and per-split positive-window prevalence under the
   eval window settings and confirm it matches the value the user signed
   off on in Step 4. Also report the per-window positive-fraction
   distribution (e.g. quantiles or histogram buckets) so transition-heavy
   or near-threshold windows are visible.
6. **Visual spot-check.** Plot 30 s of signal with the corresponding
   binary label track for one recording. Onsets and offsets of the gait
   event (e.g. FoG bouts) should align with visible signal changes;
   long stretches of class `0` should sit on a visibly different motion
   regime than class `1` segments.
7. **Promotion gate.** Only after checks 1-6 pass, ask the user whether to
   promote `processed_rebuild_tmp` to production `processed/`. Do not rename
   or delete production data without explicit approval.
8. **Sign-off log.** Append a one-line entry to the Sign-off Log section
   below: dataset name, date, in-scope subset, fix path used (3a/3b/none),
   old-vs-new coverage/window summary, the chosen `PRIORITY_LABELS`
   threshold, and the diff numbers from check 1.

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

## Worked example: OdayFoG

One concrete data point. Other gait datasets may share none of these
failure modes.

- Raw: per-recording XLSX files under
  `OdayFoG/data/raw/imus{6,11}_subjects{4,7}/`, one workbook per
  `(subject, visit, walk_direction, trial)` recording, ~128 Hz timestamp
  column with no large gaps and a binary `freezing` column at the raw
  timestep.
- In-scope: subjects with at least one accelerometer placement; gyroscope
  added when present in the raw workbook (varies by subject cohort).
  Grouping key: `(subject, visit, walk_direction, trial)`.
- Diff verdict: signals **broken**, labels **broken**.
  - Existing `processed/` advertised every placement/sensor for every
    recording, but most were placeholder parquets full of None arrays
    for sensors the raw IMU never recorded.
  - Existing processed parquets stored a single row of array payloads
    instead of standard x/y/z columns.
  - Filename token collision: the `walk_direction` (`walklr`) token was
    dropped from the processed naming, so two distinct raw recordings
    (`subject7_v12_t2` `walklr=0` and `walklr=1`) collapsed onto one
    processed `record_id`.
  - Two raw cohorts (`imus6_subjects7`, `imus11_subjects4`) cover
    overlapping subjects with different sensor counts; the existing
    pipeline picked between them inconsistently.
- Action: Step 3b full rebuild — prefer the richer `imus11_subjects4`
  cohort, drop placeholder streams, resolve the `walklr` collision by
  user-approved kept naming (pick the raw branch whose row count
  matches the existing processed file), and write standard x/y/z
  parquets. Loader updates: `available_sensors`,
  `placement_patterns`, and `default_placement = 'ankle_l'`. Step 4
  threshold sweep `{0.20, 0.40, 0.50}` -> user ultimately picked
  `0.20` as the downstream default. Step 5 PatchTST smoke test passed
  end-to-end.

## Sign-off Log

(Gait audits only. Non-gait dataset sign-offs live in the general
`DATASET_AUDIT_PLAYBOOK.md`.)

- 2026-04-29: `OdayFoG` audited, promoted, and completed on the 59-recording processed subset, using Step `3b`; archived the pre-audit production directory as `processed_20260429_before_odayfog_audit` and promoted rebuilt `processed/`, dropping fake placeholder streams and the unmatched `subject7_v12_t2` `walklr=1` branch under the user-approved kept naming, changing coverage `1,320 -> 910` signal files while preserving `59 -> 59` shared label files and increasing label rows `101,252 -> 101,296`; the audited downstream default is now `ankle_l` accelerometer with a `20%` FoG window threshold, yielding `3,332` valid 30 s / 20 Hz / 1 s-stride windows, fresh re-diff on `subject1_v34_t2`, `subject5_v27_t1`, and `subject6_v41_t2` returned signal `ok=True` and label agreement `1.0`, and the required PatchTST linear-probe smoke test completed as job `2211891` (`acc=0.8070`, `bal_acc=0.5264`, `auroc=0.6078`).
- 2026-04-29: `daphnet_fog` audited, promoted, and completed on the 17-recording processed subset, using Step `3b`; archived the pre-audit production directory as `processed_20260429_before_daphnet_fog_fix` and promoted rebuilt `processed/`, preserving coverage at `51 -> 51` signal files and `17 -> 17` shared label files across `ankle`/`thigh`/`trunk`, switching signals from filtered FFT resampling to filtered `resample_poly`, mapping raw annotations `0/1/2` to gait passthrough labels `-1/0/1`, and changing total rows `599,332 -> 599,342`; the audited downstream default remains `ankle` accelerometer with a `20%` FoG window threshold, yielding `17,878` valid 30 s / 20 Hz / 1 s-stride windows with global positive-window prevalence `18.40%`, fresh re-diff on `S01R02`, `S02R02`, and `S05R01` returned signal `ok=True` and label agreement `1.0`, and the required PatchTST linear-probe smoke test completed as job `2212132` (`acc=0.8138`, `bal_acc=0.6923`, `auroc=0.8855`).
