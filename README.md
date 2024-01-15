# FineVLM-Probe: Fine-grained Probing of Vision-Language Models

> Code for the report *FineVLM-Probe: A Lightweight Suite for Probing Fine-grained
> Visual-Language Alignment in Frozen VLMs.* In submission; pre-print pending.

This repository contains the evaluation code, probe definitions, and dataset loaders
used in the report. The goal is **not** to introduce yet another VQA benchmark, but
to provide a small, hackable harness for asking targeted questions like:

- *Does this VLM actually look at the object, or just the global scene gist?*
- *How does fine-grained alignment degrade as we scale image resolution down?*
- *Is the text encoder doing the heavy lifting, or the vision encoder?*

We support frozen-encoder evaluation across CLIP, SigLIP, BLIP-2, and LLaVA-1.5
variants. Adding a new model means writing a ~30-line adapter (see `models/`).

---

## Table of contents

- [What is in here](#what-is-in-here)
- [Probes](#probes)
- [Supported models](#supported-models)
- [Datasets](#datasets)
- [Quick start](#quick-start)
- [Running a full sweep](#running-a-full-sweep)
- [Reproducing the report numbers](#reproducing-the-report-numbers)
- [Adding a new model](#adding-a-new-model)
- [Adding a new probe](#adding-a-new-probe)
- [Caveats and known issues](#caveats-and-known-issues)
- [Citation](#citation)
- [License](#license)

---

## What is in here

```
finevlm_probe/
  models/            # thin adapters: load model, expose .encode_image, .encode_text, .score
  probes/            # each probe is a self-contained file producing a metric
  datasets/          # loaders for COCO subsets, Winoground, ARO, EqBench, our small Cantonese subset
  runners/           # CLI entry points
  reporting/         # aggregate JSON outputs into LaTeX/markdown tables
configs/             # YAML configs for each (model, probe, dataset) combination
scripts/             # download/prepare helpers; see scripts/README.md
tests/               # unit + smoke tests
```

## Probes

| Probe ID         | Question                                                            | Metric         |
|------------------|----------------------------------------------------------------------|----------------|
| `attr-swap`      | Can the model distinguish "red cube on blue ball" from "blue cube on red ball"?  | top-1 accuracy |
| `count-coarse`   | 1 vs 3 vs 5+ objects of same class                                  | macro-F1       |
| `spatial-binary` | left-of / right-of / above / below                                  | top-1 accuracy |
| `resolution-sweep` | re-run any probe at 224 / 336 / 448 / 672                          | curve          |
| `text-shuffle`   | does shuffling word order destroy the model's score (bag-of-words check)? | delta |
| `object-occlusion` | partially masked target object still recoverable from caption?    | top-1 accuracy |
| `cantonese-cap`  | Cantonese captions over the same images: does alignment hold?       | top-1 accuracy |

Each probe is a single Python file under `finevlm_probe/probes/`. They share a
`Probe` protocol (`name`, `prepare()`, `step(sample, model) -> dict`, `summarize()`).

## Supported models

We test against the following at the time of writing:

| Family   | Variants                                                  |
|----------|-----------------------------------------------------------|
| CLIP     | ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336                |
| SigLIP   | base-patch16-224, large-patch16-384, so400m-patch14-384   |
| BLIP-2   | flan-t5-xl (frozen Q-Former mode)                         |
| LLaVA-1.5| 7b, 13b (uses HF transformers `LlavaForConditionalGeneration`)  |

Adding a model means writing an adapter in `models/<name>.py`; see the
[adding a new model](#adding-a-new-model) section.

## Datasets

Downloaders live in `scripts/`. We do **not** ship the data.

| Dataset                | Used by                                  | Notes                          |
|------------------------|------------------------------------------|--------------------------------|
| COCO val2017 (subset)  | attr-swap, count-coarse, spatial-binary  | 2k images; subset list in `configs/coco_subset.json` |
| Winoground             | text-shuffle                             | obtain from HF datasets (gated) |
| ARO                    | attr-swap (cross-check)                  | https://github.com/mertyg/vision-language-models-are-bows |
| EqBench                | count-coarse                             | https://github.com/Wangt-CN/EqBen |
| Cantonese-CC (ours)    | cantonese-cap                            | tiny: 300 images, human-written Cantonese captions; see `data/CANTONESE_CC.md` for collection notes |

## Quick start

```bash
# 1. install
git clone https://github.com/mrvellang/vlm-probe-suite.git
cd vlm-probe-suite
pip install -e .[dev]

# 2. download the COCO val2017 subset we use (puts ~3GB under data/)
python scripts/get_coco_subset.py

# 3. run one probe on one model
python -m finevlm_probe.runners.run_one \
    --model clip_vit_b32 \
    --probe attr-swap \
    --dataset coco-subset \
    --out runs/clip_b32_attrswap.json
```

`run_one` emits a single JSON file. To aggregate:

```bash
python -m finevlm_probe.reporting.aggregate runs/*.json --format markdown > table.md
```

## Running a full sweep

`runners.sweep` reads a YAML and dispatches each (model, probe, dataset) cell.
GPUs are picked round-robin by default; pin with `CUDA_VISIBLE_DEVICES=0,1`.

```bash
python -m finevlm_probe.runners.sweep configs/main_sweep.yaml --out runs/main/
```

`configs/main_sweep.yaml` mirrors the report's Table 2. Expect ~6 GPU-hours on a
single A100-40G for the full main sweep; the resolution-sweep takes another ~3.

## Reproducing the report numbers

The exact configs that produced the report:

```bash
bash scripts/reproduce_table2.sh    # main results
bash scripts/reproduce_table3.sh    # resolution sweep
bash scripts/reproduce_figure4.sh   # cantonese subset
```

If your numbers move by more than ~0.5 acc., open an issue with your environment
(PyTorch / CUDA / transformers versions). Most drift we've seen is from
`transformers` BLIP-2 changes between 4.36 and 4.40.

## Adding a new model

```python
# finevlm_probe/models/my_model.py
from .base import ModelAdapter

class MyModel(ModelAdapter):
    name = "my-model-v1"

    def __init__(self, device="cuda"):
        self.device = device
        # load weights here

    def encode_image(self, images):  # PIL.Image list -> [N, D] tensor
        ...

    def encode_text(self, texts):    # list[str] -> [N, D] tensor
        ...
```

Then register in `finevlm_probe/models/__init__.py` and reference by name in your
config. The `score(image, text)` method is optional; if absent we use cosine of
the embeddings.

## Adding a new probe

A probe is one file that subclasses `Probe`:

```python
from finevlm_probe.probes.base import Probe

class MyProbe(Probe):
    name = "my-probe"

    def prepare(self, dataset):
        self.samples = list(dataset.iter_for_probe(self.name))

    def step(self, sample, model):
        # do the eval, return per-sample dict
        ...

    def summarize(self, records):
        # aggregate per-sample dicts into a metric dict
        ...
```

Register in `finevlm_probe/probes/__init__.py`.

## Caveats and known issues

- **Frozen-encoder only.** We do not fine-tune anything. Some VLMs perform very
  differently after a tiny adapter LoRA; that is out of scope.
- **Cantonese subset is tiny.** 300 images, 1 caption per image, by one annotator
  (the author). It is intended as a *sanity probe*, not a benchmark.
- **No human eval.** All metrics here are automatic.
- **BLIP-2 score path.** We use the image-text-matching head logits rather than
  generation likelihood; this matters for `attr-swap` results.
- **LLaVA models are slow.** A full sweep with LLaVA-13B takes ~40 GPU-hours.
  Use `--max-samples` to subset during development.
- We have observed numerical drift when running on CPU vs CUDA for SigLIP; the
  ranking is stable, the absolute numbers move by ~0.3 acc.
- TODO: Add Korean and Hindi subsets like we did for Cantonese.
- FIXME: `resolution-sweep` re-downloads the image at each resolution; it should
  cache the largest and resize locally.

## Citation

If you use this suite, please cite the report:

```bibtex
@techreport{cheung2025finevlmprobe,
  title  = {FineVLM-Probe: A Lightweight Suite for Probing Fine-grained
            Visual-Language Alignment in Frozen VLMs},
  author = {Cheung, Ka Yiu},
  year   = {2025},
  institution = {HKUST}
}
```

For the Cantonese subset specifically, also cite the data notes in `data/CANTONESE_CC.md`.

## License

Code is Apache-2.0 (see `LICENSE`). The Cantonese caption annotations under
`data/cantonese_cc/` are released CC-BY-4.0 as noted in that directory's README.

## Acknowledgements

Thanks to labmates for catching the Winoground loader bug in #14, and to the
authors of CLIP, SigLIP, BLIP-2, LLaVA, and the ARO and EqBench teams whose
work this suite stands on.

<!-- siglip variants now registered, see models/__init__.py -->
