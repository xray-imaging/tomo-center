# tomo-center-ai

Standalone AI rotation-axis center picker for pre-reconstructed TIFF slices.

The model and inference code are **vendored from
[tomocupy/develop](https://github.com/stang292/tomocupy/tree/develop/src/tomocupy/ai)**
(BSD-3, UChicago Argonne LLC). This repo packages just the bits needed to run
the classifier outside the full `tomocupy` reconstruction stack — no CUDA build,
no SWIG, no HDF5 reader.

## What it does

Given a folder of TIFF slices — each one already reconstructed at a different
candidate center of rotation — the DINOv2-based classifier scores every slice
and writes the candidate(s) with the highest "correct center" probability.

Input shape per slice: 2D, any dtype (cast to float32). All slices must share
the same `(H, W)`.

## Install

Use a dedicated conda env — `torch` and a pinned `numpy<2` would conflict with
most existing envs.

```bash
conda create -n tomo-center-ai python=3.10 pip -y
conda activate tomo-center-ai
pip install -e /path/to/tomo-center-ai
```

The editable install (`pip install -e .`) pulls every runtime dep declared in
`pyproject.toml`. The full list:

| Package    | Why                                                          |
| ---------- | ------------------------------------------------------------ |
| `numpy<2`  | array ops; pinned because `torch 2.2.x` wheels were built against NumPy 1.x |
| `pillow`   | image resize (PIL `Image.fromarray` / `BILINEAR`) inside the inference pipeline |
| `tifffile` | reading TIFF slices from the input folder                    |
| `torch`    | DINOv2 backbone + classifier head                            |
| `einops`   | tensor rearrange used inside `model_archs.py`                |

GPU inference is automatic when CUDA is available — install the matching
`torch` CUDA wheel for your system (see https://pytorch.org/get-started). CPU
works but is slow.

## Get the model checkpoint

```
https://anl.box.com/s/4o8qcig6pl9k8p7x4z3qqbrpgnjipolq
```

## Run

```bash
tomo-center-ai /path/to/recons \
    --model-path /path/to/model.pt \
    --out-dir   /path/to/out
```

### How centers are paired with TIFFs

By default the **last numeric token in each filename stem** is used as that
slice's candidate center, e.g. `recon_0001_1234.50.tif → 1234.5`.

If your filenames don't carry the center, supply a sidecar file:

```bash
tomo-center-ai /path/to/recons \
    --model-path /path/to/model.pt \
    --centers-file centers.txt
```

`centers.txt` is one float per line, in the same sorted order as the TIFFs.
Lines starting with `#` are ignored.

### Multi-scale inference

The upstream pipeline supports running multiple `(downsample, num_windows, window_size)`
scales and combining their features. Pass matching-length lists:

```bash
tomo-center-ai /path/to/recons --model-path model.pt \
    --downsample-factor 1 2 \
    --num-windows       4 4 \
    --window-size       224 224
```

## Output

- `center_of_rotation.txt` — one center per line (appended, not overwritten).
- `predicts_all.npz` — raw model logits + the center list, if
  `--save-intermediate` is passed.
- `scores.png` (or a custom path) — per-slice probability vs. candidate
  center, if `--plot` is passed.

## Diagnostic plot

```bash
pip install -e '.[plot]'                # adds matplotlib

tomo-center-ai /path/to/recons \
    --model-path /path/to/model.pt \
    --plot                              # → <out-dir>/scores.png
# or:
tomo-center-ai /path/to/recons --model-path model.pt --plot my-scan.png
```

A sharp peak with neighbors tapering off → confident pick. A flat curve or
several near-ties at the top → the sweep was too coarse, too narrow, or the
slices don't carry enough signal for the classifier to discriminate; re-sweep
finer around the picked value and run again.

## Attribution

- `src/tomo_center_ai/ai/inference.py` — vendored from
  `tomocupy/src/tomocupy/ai/inference.py` (only the internal import path changed).
- `src/tomo_center_ai/ai/model_archs.py` — vendored verbatim from
  `tomocupy/src/tomocupy/ai/model_archs.py`. It in turn includes a DINOv2 ViT
  (Apache-2.0, Meta) and attention pooling (MIT, Ilse & Tomczak).
- See `LICENSE` for the upstream BSD-3 terms.
