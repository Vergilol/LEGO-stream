# Evaluation

`run_eval.py` runs LEGO-Stream over a corpus and writes a hypothesis file.
Scoring is a separate step, deliberately — see [Scoring](#scoring).

## Manifest

One JSON list. `segments` is the reference; the streaming run never reads it,
which is what keeps the evaluation causal.

```json
[
  {
    "audio_name": "EN2002a",
    "audio_path": "/abs/path/EN2002a.wav",
    "segments": [
      {"start": 0.0, "end": 1.90, "speaker": "D", "text": "funky stuff like that"},
      {"start": 0.94, "end": 7.07, "speaker": "A", "text": "wonder how much of"}
    ]
  }
]
```

Reference speaker names are arbitrary strings — cpWER scores under the best
permutation, so they need not match the system's labels.

## Run

```bash
python -m eval.run_eval \
    --manifest data/ami_ihm.json \
    --run-root runs/ami_ihm \
    --speaker-model /path/to/eres2netv2_dir \
    --speakerlab-root /path/to/3D-Speaker \
    --endpoint http://127.0.0.1:19300/v1
```

Output under `--run-root`:

| Path | Contents |
|---|---|
| `hypothesis.json` | all recordings, ready for the scorer |
| `reference.json` | references copied from the manifest |
| `run_summary.json` | per-recording status, RTF, the protocol actually used |
| `recordings/<name>/` | per-recording `config.json`, `hypothesis.json`, tick trace |

A recording that fails is recorded and skipped, so one bad file does not lose a
sweep. Use `--fail-fast` to stop instead.

Other flags: `--only NAME` (repeatable), `--limit N`,
`--shard-index i --shard-count n` for parallel sweeps across GPUs or hosts.

Defaults are the settings behind the reported results; change `--context-sec` (60)
or `--step-sec` (2) only if you mean to change the method. `--calibration` and
`--crfs` are optional and off — see
[Optional features](../README.md#optional-features).

## Scoring

Built on
[MeetEval](https://github.com/fgnt/meeteval), with punctuation removed, Chinese
tokenized by character and English by word, and each recording scored on its own.
**cpWER / cpCER** is the primary metric — concatenated per speaker under the best
permutation, charging recognition and attribution together. **WER / CER**
ignores labels, and **DER** sums missed speech, false alarms, and confusion, at a
0-s collar with overlap scored.

> [!TIP]
> MeetEval's alignment is order-sensitive. Sort segments by `(start, end)` before
> scoring — scoring in commit order instead shifts cpWER by 0.01–0.02.
