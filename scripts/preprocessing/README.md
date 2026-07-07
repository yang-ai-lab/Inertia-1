# Downstream dataset preprocessing

These scripts turn raw public HAR / FoG accelerometry datasets into the
canonical **20 Hz parquet** format consumed by the downstream evaluation
dataloader (`motion_dataloader.py`).

We are not allowed to redistribute the datasets themselves, so this directory
ships only the *preprocessing pipeline*. Download each dataset from its
original source, then run the matching script.

## Output contract

Every script produces continuous, gap-aware time series resampled to 20 Hz and
written under:

```
<DATA_ROOT>/<dataset_name>/processed/
    <prefix>_<subject/session>_<placement>_<sensor>_20Hz_<duration>s.parquet   # signal: columns x,y,z
    <prefix>_<subject/session>_labels.parquet                                  # labels: timestamp,label (+ extra tracks)
```

`<dataset_name>` must match the key used in `motion_dataloader.py`
(`DATASET_CONFIGS`). The evaluation loader globs `processed/*.parquet`, pairs
each signal file with its label file by filename convention, windows on the
fly (default 30 s @ 20 Hz, 1 s stride), and assigns window labels by majority
vote (HAR) or a coverage threshold (FoG, see `PRIORITY_LABELS`).

Shared conventions (resampling, gap splitting, label alignment, filename
contract) are documented in [`docs/DATASET_AUDIT_PLAYBOOK.md`](docs/DATASET_AUDIT_PLAYBOOK.md)
(general / HAR) and [`docs/DATASET_AUDIT_PLAYBOOK_GAIT.md`](docs/DATASET_AUDIT_PLAYBOOK_GAIT.md)
(FoG / gait). A completed worked example is in
[`docs/daphnet_fog_audit_notes.md`](docs/daphnet_fog_audit_notes.md).

> **Note:** Each script reads its raw-input root from the `DATASET_ROOT`
> environment variable (default `./data/raw/<dataset>`). Set `DATASET_ROOT`, pass
> `--input_dir` / `--output_dir` where supported, or edit the path constants at
> the top of the script for your environment.

## Scripts

### HAR (`har/`)

| Script | Dataset (`dataset_name`) | Default placement |
|--------|--------------------------|-------------------|
| `wisdm.py` | `wisdm` | watch |
| `hhar.py` | `HHAR` | watch |
| `harth.py` | `harth` | thigh |
| `har70plus.py` | `har70plus` | thigh |
| `pamap2.py` | `PAMAP2_Dataset` | hand |
| `mhealth.py` | `MHEALTHDATASET` | rarm |
| `opportunity.py` | `OpportunityUCIDataset` | lwr / rwr |
| `wear_dataset.py` | `wear_dataset` | right_arm |
| `recofit.py` | `Recofit` | rightarm |

`wear_dataset.py` is an audit/copy step (drops a known-corrupted file) rather
than a full raw rebuild, since WEAR ships already-processed parquet.

### FoG (`fog/`)

| Script | Dataset (`dataset_name`) | Default placement |
|--------|--------------------------|-------------------|
| `daphnet_fog.py` | `daphnet_fog` | ankle |
| `odayfog.py` | `OdayFoG` | ankle_l |

FoG labels use the passthrough integer convention `{-1: outside experiment, 0:
no freeze, 1: freeze}`; window labels are derived by a coverage threshold
rather than majority vote.

## Frozen evaluation splits

The evaluation uses *frozen* train/val/test splits (subject-grouped) stored as
index lists. These are **not shipped** (they are derived from the
non-redistributable datasets). After preprocessing a dataset, generate splits
with `inertia1/eval/motion_split_analysis.py`. Because split indices are positions
into the sorted window list, renaming output files invalidates existing splits.
