# Inertia-1: An Open Exploration of Wearable Motion Foundation Models
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://yang-ai-lab.github.io/Inertia-1/paper/)
[![Website](https://img.shields.io/badge/website-demo-blue)](https://yang-ai-lab.github.io/Inertia-1/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)


Inertia-1 is an open exploration of **wearable motion foundation models**. Motion sensing is fragmented — datasets disagree on sampling rate, window length, sensor modality, body placement, and even signal format, and nearly every task gets its own bespoke model. Inertia-1 studies the full lifecycle of motion models (data, sensing, objectives, and scale) inside a single, controlled space, asking whether one self-supervised representation learned from raw accelerometry can transfer across the body, devices, and downstream tasks.

The headline finding: a representation pretrained on **wrist accelerometry alone** transfers to placements and sensor types it never saw during training, and fusing additional streams makes it both more accurate and cleaner. This repository is an intentionally **minimal, end-to-end slice** of that pipeline. Our internal codebase supports many model families; here we ship two representative ones — **AR-Transformer** (patch-level autoregressive Transformer) and **PatchTST** (masked patch reconstruction) — so the workflow is easy to read start to finish. The datasets we used (NHANES plus public HAR/FoG corpora) are not redistributable, so we ship the *preprocessing pipelines* and *loader/placeholder code* instead, with pointers below.

---

## 📰 News
- **[2026-07-06]** [Project website](https://yang-ai-lab.github.io/Inertia-1/) is live!
- **[2026-07-06]** Code released on GitHub!
- **[Coming soon]** Paper release on arXiv.

---

## ✨ What you can do with this repo

- **Self-supervised pretraining** on raw accelerometry, with the key exploration axes exposed as first-class knobs (uniaxial vs. triaxial, sampling rate, window length).
- **Downstream evaluation** on human-activity-recognition (HAR) and freezing-of-gait (FoG) datasets via linear probing / full finetuning.
- **Patient-level disease & medication detection** via attention-based multiple-instance learning (MIL) on frozen embeddings.

---

## Repository layout

```
motion-fm/
├── inertia1/                       # the core package (pip install -e .)
│   ├── run.py                    # unified entry: pretrain + HAR/FoG finetune
│   ├── config/                   # layered YAML configs
│   │   ├── default.yaml
│   │   ├── methods/{ar_transformer,patchtst}.yaml
│   │   └── presets/{small,medium,large}.yaml
│   ├── data/                     # pretraining datamodule (subject-grouped windows)
│   ├── methods/                  # registry + ar_transformer / patchtst modules
│   ├── models/backbones/patchtst # PatchTST backbone
│   ├── augmentations/            # patch masking
│   ├── eval/                     # HAR/FoG downstream probe + split generator
│   │   ├── motion_linear_probe.py
│   │   └── motion_split_analysis.py   # generate your own frozen splits
│   └── scripts/eval_nhanes_labels.py  # patient-level MIL disease detection
├── motion_dataloader.py          # HAR/FoG dataset + windowing + split loading
├── scripts/
│   ├── pretrain/                 # pretrain_{ar_transformer,patchtst}.sh
│   ├── eval/eval_motion.sh       # HAR/FoG evaluation
│   ├── disease_detection/        # disease MIL eval + placeholder-data generator
│   └── preprocessing/            # HAR/FoG dataset preprocessing (har/, fog/, docs/)
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
python -m pip install -e .          # reads pyproject.toml
# or: python -m pip install -r requirements.txt
```

Python ≥ 3.10. A CUDA-capable GPU is recommended for pretraining.

---

## 1. Pretraining

Entry point: `python -m inertia1.run` with layered configs. The three exploration
axes are plain `data.*` overrides, so a single command covers any ablation:

| Axis | Override | Values |
|------|----------|--------|
| Window length | `data.window_sec` | e.g. `10`, `30`, `60` |
| Sampling rate | `data.hz` | any divisor of `data.native_hz` (e.g. `20`, `10`, `5`, `1`) |
| Uniaxial vs. triaxial | `data.axes` | `3` (X/Y/Z) or `1` (L2 magnitude, via `data.axis_reduce`) |

```bash
# Triaxial, 30 s @ 20 Hz, small AR-Transformer
DATA_ROOT=./data/pretrain bash scripts/pretrain/pretrain_ar_transformer.sh

# Uniaxial, 60 s @ 10 Hz, PatchTST
WINDOW_SEC=60 HZ=10 AXES=1 DATA_ROOT=./data/pretrain \
  bash scripts/pretrain/pretrain_patchtst.sh

# Or call the launcher directly
python -m inertia1.run \
  --config inertia1/config/default.yaml \
  --config inertia1/config/methods/ar_transformer.yaml \
  --config inertia1/config/presets/small.yaml \
  run.stage=pretrain data.window_sec=30 data.hz=20 data.axes=3 \
  data.data_root=./data/pretrain
```

Model size is set by the preset (`small` / `medium` / `large`).

### Pretraining data (NHANES)

We pretrained on **NHANES** accelerometry. NHANES raw accelerometer data is
publicly available but governed by its own usage terms, so we do not
redistribute it here.

- **Download:** [NHANES (CDC/NCHS)](https://wwwn.cdc.gov/nchs/nhanes/default.aspx)
- **Preprocessing:** we convert raw NHANES into the per-subject parquet layout
  using the **UKBB-style accelerometer preprocessing pipeline**
  ([UK Biobank](https://www.ukbiobank.ac.uk/)).

Expected on-disk layout (point `data.data_root` here):

```
<data_root>/<subject_id>/**/*.parquet     # columns: X, Y, Z  (native 20 Hz)
```

An optional subject-level split CSV (`data.split_csv`, columns
`patient_id,split` with `split ∈ {train,val}`) keeps memory bounded on large
corpora.

---

## 2. Downstream evaluation (HAR / FoG)

We evaluate frozen (linear probe) or finetuned backbones on public HAR and FoG
datasets. The interesting, reusable part is the **dataloader**
(`motion_dataloader.py`): per-dataset windowing, label mapping, placement
selection, and frozen, subject-grouped splits. Everything else (the
probe/finetune loop) is deliberately small.

Frozen splits are **not shipped** (they are derived from non-redistributable
datasets). After preprocessing your own data copies, generate splits with
`inertia1/eval/motion_split_analysis.py`, then point the evaluation at them:

```bash
METHOD=ar_transformer MODE=linear_probe \
  CKPT_PATH=./outputs/<run>/checkpoints/last.ckpt \
  DATA_ROOT=./data/downstream \
  SPLIT_DIR=./splits \
  bash scripts/eval/eval_motion.sh
```

Datasets covered: `wisdm`, `HHAR`, `harth`, `har70plus`, `PAMAP2_Dataset`,
`MHEALTHDATASET`, `OpportunityUCIDataset`, `wear_dataset`, `Recofit`,
`capture24_willetts` (HAR) and `daphnet_fog`, `OdayFoG`, `FoGTurning` (FoG).

### Downstream data

These datasets are also not redistributable. Download each from its original
source and preprocess into the canonical 20 Hz parquet layout with the scripts
in [`scripts/preprocessing/`](scripts/preprocessing/README.md):

```
<data_root>/<dataset_name>/processed/*.parquet
```

The preprocessing pipeline (resampling, gap handling, label alignment, filename
contracts) is fully documented there.

---

## 3. Disease detection (patient-level MIL)

We detect patient-level outcomes (e.g. depression severity, sleep complaints,
medication use) by pooling a *bag* of a patient's accelerometer windows through
a positional attention MIL head on top of a frozen backbone.

Internally this runs on **NHANES** (closed-source, as above), so we ship a
**generic placeholder-data loader** and keep patient/bag construction high
level. The real pipeline builds per-day bags keyed on wear-time over a fixed 24 h
grid; the shipped loader places each patient's windows onto a fixed grid of
`bag_size` slots with a learned "missing slot" embedding — the same MIL
mechanics, without NHANES-specific internals.

Run it end-to-end on synthetic placeholder data:

```bash
# 1) generate a tiny synthetic dataset (noise; smoke-test only)
python scripts/disease_detection/make_placeholder_data.py --out ./data/disease

# 2) train the MIL probe on a frozen backbone
CKPT_PATH=./outputs/<run>/checkpoints/last.ckpt \
  bash scripts/disease_detection/eval_disease.sh
```

Placeholder data format:

```
<data_root>/<patient_id>.npy   # float array [N_windows, C, T]
labels.csv                     # patient_id,label
split.csv                      # patient_id,split  (train/val/test)
```

To use real data, replace the placeholder files with your own in this format.
NHANES download / preprocessing pointers are the same as in
[Pretraining data](#pretraining-data-nhanes).

---

## Notes & scope

- Two methods are shipped (`ar_transformer`, `patchtst`); the config/registry
  are structured so adding a method is a localized change.
- The datasets (NHANES + public HAR/FoG corpora) are **not** included. This repo
  ships the code, configs, frozen splits, and preprocessing pipelines needed to
  reproduce the workflow on your own copies of the data.

---

## 📝 Citation

If you use Inertia-1 in your research, please cite the paper:

```bibtex
@article{inertia1,
  title={Inertia-1: An Open Exploration of Wearable Motion Foundation Models},
  author={Xu, Zongzhe and Anand, Aakarsh and Jiang, Sarah and Zhuang, Chuntung and Shuai, Zitao and Sankararaman, Sriram and Yang, Yuzhe},
  journal={arXiv preprint arXiv:<PLACEHOLDER>},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Pretraining data: **NHANES** accelerometry (governed by its own usage terms).
- Downstream benchmarks: the public **HAR** and **FoG** corpora listed above, each from its original source.
- Model architecture inspiration: PatchTST (https://github.com/yuqinie98/PatchTST).
