# Contributing

Hey, thanks for poking around.

## Bug reports

The most useful issues include:

- the command you ran (full CLI)
- `pip freeze | grep -iE 'torch|transformers|open_clip'`
- a 5-line snippet from the JSON output if the metric is suspicious

## Adding a model adapter

See the README section *Adding a new model*. A few extra notes:

- Please keep the adapter purely about loading weights and exposing
  `encode_image`/`encode_text`. Probe logic goes in `probes/`, not here.
- If your model needs special tokenization (e.g. SigLIP's max-length pad), do
  it inside the adapter -- the probes pass plain strings.
- Add a smoke test in `tests/` that imports the adapter and constructs the
  registry entry (you do not need to load weights in CI; we don't).

## Style

We run `ruff` with the config in `pyproject.toml`. PRs should pass `ruff check .`
and `pytest -q`. CI does this for you.

## CI tip

If your test imports `transformers` at the top level, please guard it -- our CI
runner doesn't have GPU model weights and the slow import drags the build past
the 10-minute mark.
