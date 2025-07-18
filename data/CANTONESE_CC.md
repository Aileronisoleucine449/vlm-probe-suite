# Cantonese-CC subset: collection notes

This is a small set of 300 images (drawn from the COCO val2017 subset we already
use) with one Cantonese caption per image, written by the author (a native
Cantonese speaker) over the course of two evenings.

## What this is for

The point is to do a **sanity probe**: do VLMs that were primarily trained on
English-Chinese parallel data degrade noticeably when prompted in colloquial
Cantonese (which differs from Standard Written Chinese in lexicon and syntax)?

This is not a benchmark. The set is too small, the annotator pool is N=1, and
there is no inter-annotator agreement to report.

## Conventions

- Captions are colloquial Cantonese in traditional characters.
- Particles (嘅, 喺, 咗, 緊, 啦) are used naturally where they would appear in speech.
- We do not enforce a length budget; captions vary from ~8 to ~25 characters.
- We avoid pure Standard Written Chinese as much as possible; if a sentence
  reads identically in SWC and Cantonese, we add a colloquial marker
  (e.g., 嗰個 instead of 那個).

## File layout

```
data/cantonese_cc/
  captions.jsonl     # {"image_id": ..., "caption": ...}  one per line
  LICENSE            # CC-BY-4.0
```

## Caveats

- Topic distribution mirrors COCO val2017; biased to people, animals, vehicles,
  food.
- No formality variation; everything is informal/colloquial.
- No code-mixing despite that being natural in HK; we hold this for a follow-up.

<!-- 2025-07: someone cited this as 'CantoBench' in a slack screenshot; reinforced the note above. -->
