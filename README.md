# tomo-center

Standalone AI rotation-axis center picker and fine-tuner for pre-reconstructed
TIFF slices.

The model and inference code are **vendored from
[tomocupy/develop](https://github.com/stang292/tomocupy/tree/develop/src/tomocupy/ai)**
(BSD-3, UChicago Argonne LLC). This repo packages just the bits needed to run
the classifier outside the full `tomocupy` reconstruction stack — no CUDA build,
no SWIG, no HDF5 reader.

Two subcommands:

| Command | What it does |
| --- | --- |
| `tomo-center find`  | Score a folder of reconstructed slices (one per candidate center) and report the best center. |
| `tomo-center train` | Fine-tune the classifier head on labeled slices to adapt the shipped checkpoint to a new sample family. |

## Install

Use a dedicated conda env — `torch` and a pinned `numpy<2` would conflict with
most existing envs.

```bash
conda create -n tomo-center python=3.10 pip -y
conda activate tomo-center
pip install -e /path/to/tomo-center
```

The editable install pulls every runtime dep declared in `pyproject.toml`:

| Package    | Why                                                          |
| ---------- | ------------------------------------------------------------ |
| `numpy<2`  | array ops; pinned because `torch 2.2.x` wheels were built against NumPy 1.x |
| `pillow`   | image resize (PIL `Image.fromarray` / `BILINEAR`) inside the inference pipeline |
| `tifffile` | reading TIFF slices from the input folder                    |
| `torch`    | DINOv2 backbone + classifier head                            |
| `einops`   | tensor rearrange used inside `model_archs.py`                |

GPU is automatic when CUDA is available — install the matching `torch` CUDA
wheel for your system (see https://pytorch.org/get-started). CPU works but is
slow.

## Get the model checkpoint

```
https://anl.box.com/s/4o8qcig6pl9k8p7x4z3qqbrpgnjipolq
```

## `find` — pick the best center

```bash
tomo-center find /path/to/recons \
    --model-path /path/to/model.pt \
    --out-dir    /path/to/out
```

### Example

200 slices reconstructed at centers 974.5–1074.0, run on tomo4 (GPU):

```console
$ tomo-center find \
    /data2/2BM/2023-04/Strumendo-2023-04_rec/try_center/CaCO3room_001/ \
    --model-path /home/beams/2BMB/models/datav2_518_full_finetune.pt \
    --out-dir    ~/tomo-center-out/CaCO3room_001 \
    --plot
2026-06-05 20:03:03,134 - Loading TIFFs from /data2/2BM/.../CaCO3room_001 ...
2026-06-05 20:03:05,058 -   200 slices, shape (2048, 2048), dtype float32
2026-06-05 20:03:05,059 -   center range: 974.5 .. 1074.0
2026-06-05 20:03:06,894 - starting model inference...
2026-06-05 20:03:06,894 - Downsample factor is 1. No resizing applied.
2026-06-05 20:03:11,113 - done. Elapsed time is 4.22 s.
2026-06-05 20:03:11,121 - Best center(s):
2026-06-05 20:03:11,121 -   1023.0
```

Score curve written to `<out-dir>/scores.png`:

![Score curve example](docs/source/img/scores.png)

### How centers are paired with TIFFs

By default the **last numeric token in each filename stem** is used as that
slice's candidate center, e.g. `recon_0001_1234.50.tif → 1234.5`.

If your filenames don't carry the center, supply a sidecar file:

```bash
tomo-center find /path/to/recons \
    --model-path /path/to/model.pt \
    --centers-file centers.txt
```

`centers.txt` is one float per line, in the same sorted order as the TIFFs.
Lines starting with `#` are ignored.

### Multi-scale inference

The upstream pipeline supports running multiple `(downsample, num_windows, window_size)`
scales and combining their features. Pass matching-length lists:

```bash
tomo-center find /path/to/recons --model-path model.pt \
    --downsample-factor 1 2 \
    --num-windows       4 4 \
    --window-size       224 224
```

### Output

- `center_of_rotation.txt` — one center per line (appended, not overwritten).
- `predicts_all.npz` — raw model logits + the center list, if
  `--save-intermediate` is passed.
- `scores.png` (or a custom path) — per-slice probability vs. candidate
  center, if `--plot` is passed.

### Diagnostic plot

```bash
pip install -e '.[plot]'              # adds matplotlib

tomo-center find /path/to/recons --model-path model.pt --plot
# → <out-dir>/scores.png, opens an interactive window if $DISPLAY is set
```

A sharp peak with neighbors tapering off → confident pick (see the
[example](#example) above). A flat curve or several near-ties at the top → the
sweep was too coarse, too narrow, or the slices don't carry enough signal for
the classifier to discriminate; re-sweep finer around the picked value and run
again.

## `train` — fine-tune the classifier head

Use this to adapt the shipped checkpoint to a new sample family. Only the
**1,538-parameter classifier head** is trained — the DINOv2 backbone stays
frozen (the vendored `ClassificationModel.forward` enforces that). Full
backbone fine-tuning is out of scope for this repo.

### Data layout

```
labels/
  centered/
    well_centered_001.tif
    well_centered_002.tif
    ...
  off_centered/
    bad_center_001.tif
    bad_center_002.tif
    ...
```

Each TIFF is one 2D reconstructed slice. Label = which subfolder it's in.
You produce this by running a CoR sweep on a representative scan, eyeballing
which reconstructions look right, and sorting them.

### Run

```bash
tomo-center train /path/to/labels \
    --resume /path/to/datav2_518_full_finetune.pt \
    --out    /path/to/finetuned_model.pt
```

`--resume` is strongly recommended: starting fresh requires downloading
DINOv2's public weights from Meta via `torch.hub`, which needs internet (won't
work on private beamline networks).

### Defaults and key flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--epochs` | 20 | Head-only training is fast — try a few values. |
| `--batch-size` | 8 | Bump up to fit GPU memory. |
| `--lr` | 5e-5 | AdamW. |
| `--val-split` | 0.1 | Held-out fraction for per-epoch val accuracy logging. |
| `--window-size` | 224 | Must match the value you use at inference. |
| `--no-augment` | off | By default training uses random horizontal flip + small (±sz/8) crop offset. |
| `--base-model` | `dinov2_vitb14` | Backbone variant for the (rare) from-scratch path. |

### Output

A `.pt` file at `--out`, saved only when val accuracy improves. Structure:

```python
{"epoch": ..., "state_dict": ..., "args": {...}, "val_acc": ...}
```

Load it back with `tomo-center find --model-path <out>.pt`.

### Hub access on offline boxes

When `--resume` is **not** given, `train` calls
`torch.hub.load('facebookresearch/dinov2', ...)`, which goes to the internet.
On private networks (e.g., APS `tomo4`) this fails with a clear error message —
use `--resume` instead.

## Attribution

- `src/tomo_center/ai/inference.py` — vendored from
  `tomocupy/src/tomocupy/ai/inference.py`. Changes vs. upstream: internal
  import path; switched `print()` to the package logger; fixed an
  `UnboundLocalError` in the single-instance branch (`patch_corner` →
  `patch_corners`).
- `src/tomo_center/ai/model_archs.py` — vendored verbatim from
  `tomocupy/src/tomocupy/ai/model_archs.py`. It in turn includes a DINOv2 ViT
  (Apache-2.0, Meta) and attention pooling (MIT, Ilse & Tomczak).
- `src/tomo_center/logging.py` — adapted from
  `tomocupy/src/tomocupy/logging.py` (same colored-console formatter, scoped
  to the `tomo_center.*` logger tree).
- `src/tomo_center/train.py` — new. Single-GPU head-only fine-tune harness;
  the AdamW gain/bias weight-decay split and cosine-LR-with-warmup recipe are
  borrowed from an internal training script by S. Tang.
- See `LICENSE` for the upstream BSD-3 terms.
