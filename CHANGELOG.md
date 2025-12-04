# Changelog

## 0.6.2 - 2025-12
- Aggregator: don't crash if a JSON is missing the configured metric.
- More tolerant CocoSubset loader when images are missing.

## 0.6.1 - 2025-09
- Fix Winoground loader path (#14, thanks @r-zhang).
- Cantonese subset notes moved into data/CANTONESE_CC.md.

## 0.6.0 - 2025-06
- Add Cantonese subset (300 imgs).
- Add `resolution-sweep` probe family.
- Switch to module-level `_REGISTRY` for models so dynamic ones can register.

## 0.5.0 - 2025-02
- BLIP-2 adapter (frozen Q-Former) added.
- Reporting: LaTeX table output via `--format latex`.

## 0.4.0 - 2024-11
- LLaVA-1.5 adapter (7B and 13B).
- macro-F1 metric for count-coarse.

## 0.3.0 - 2024-07
- SigLIP family (base/large/so400m).
- Sweep runner with round-robin GPU pinning.

## 0.2.0 - 2024-03
- Refactor probe API: prepare/step/summarize.
- COCO subset loader.

## 0.1.0 - 2023-11
- Initial: attr-swap probe over CLIP-B/32.
