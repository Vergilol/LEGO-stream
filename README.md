# LEGO-Stream

Reference implementation of **LEGO-Stream: Local-to-Global Speaker Tracking for
Streaming Speaker-Attributed ASR with a Speech LLM**.

A speech LLM marks who speaks only relative to the window it just decoded, and
never separates a hypothesis it may still revise from a label it has already
shown. LEGO-Stream keeps speaker identity outside the backbone, so nothing caps
the duration or the speaker count, and separates the four things one speaker
label conflates:

- **L — local.** The speaker index inside one decode's snapshot, renumbered
  freely and never shown.
- **E — evidence.** The embeddings of a committed turn's causally available
  audio. Recorded once, immutable.
- **G — global.** The persistent set of speakers for the recording. The only
  mutable state.
- **O — output.** The label shown to the listener. Never revised.

One constrained assignment resolves each chunk's committed turns at once, and
opening a speaker carries an explicit cost rather than crossing a distance
threshold, so the set grows on agreement between separate sources.

## Install

```bash
git clone https://github.com/Vergilol/LEGO-stream && cd LEGO-stream
pip install -e '.[embedding,audio]'
```

The framework core needs only NumPy and SciPy. The speech LLM is reached over HTTP
rather than imported, so the `embedding` extra pulls in `torch` for the speaker
encoder alone.

Requires Python ≥3.10.

### External dependencies

Two models and one repository, none of which pip can fetch for you. Both models
are used exactly as released.

**1. Speech LLM — served over HTTP.** Any OpenAI-compatible endpoint exposing
`/v1/audio/transcriptions` and emitting `[start][Sxx]text[end]` works, and the
client lives in a single adapter module. The paper uses
[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
0.9B:

```bash
# weights: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
huggingface-cli download OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --local-dir ./checkpoints/moss-transcribe-diarize

# serve (vLLM >= 0.23; the paper's runs used 0.23.1, bfloat16, one A40)
pip install vllm
vllm serve ./checkpoints/moss-transcribe-diarize \
    --served-model-name moss-transcribe-diarize \
    --trust-remote-code --dtype bfloat16 --port 19300
```

`--trust-remote-code` is required: the audio front-end ships as custom modelling
code with the checkpoint. Point `endpoint` at `http://127.0.0.1:19300/v1`.

**2. Speaker encoder — imported in-process.**
[ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common).
The `speakerlab` package is not on PyPI, so you need a
[3D-Speaker](https://github.com/modelscope/3D-Speaker) checkout *and* the
checkpoint:

```bash
# code
git clone https://github.com/modelscope/3D-Speaker     # -> --speakerlab-root

# weights (215 MB): pretrained_eres2netv2w24s4ep4.ckpt
# https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
pip install modelscope
modelscope download --model iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common \
    --local_dir ./checkpoints/eres2netv2                # -> --speaker-model
```

A stock upstream clone is enough — only the embedding model and the fbank
front-end are used. `--speaker-model` may be the checkpoint file or the directory
containing it.

CAM++ is also accepted (`embedding.model_name: "campplus"`). For any other
encoder, pass `embedding_store=` to `StreamingSpeakerASRSession`: an object
exposing `embedding_views(...)` that returns unit-norm vectors for a span. Keep
`embedding.dimension` in sync with it.

## Quickstart

Copy the example config and fill in four fields:

```bash
cp configs/stream.example.json my.json
```

`sample.audio_path`, `embedding.speakerlab_root`, `embedding.model_dir`, and
`sample.duration_sec` — the last drives the window schedule, so set it to the real
length. A config that disagrees with the audio is rejected rather than silently
truncated.

Then run it:

```bash
lego-stream run-stream --config my.json --output-dir out/
```

`out/hypothesis.json` holds the committed transcript. The same thing in Python:

```python
import json
from lego_stream import run_streaming_file

config = json.load(open("my.json"))
result = run_streaming_file(config, "out/")
```

To drive a live source chunk by chunk, use the session API the demo is built on:

```python
from lego_stream import StreamingSpeakerASRSession

session = StreamingSpeakerASRSession(config, "out/")
for end in [2.0, 4.0, 6.0]:
    row = session.process_tick(end, final=False)
    row["delta_segments"]    # append-only: newly published, never revisited
    row["pending_segments"]  # the revisable tail
result = session.finalize()
```

`delta_segments` is the only append-only payload; `local_segments` and
`mapped_segments` are whole-window snapshots that repeat old audio by design.
Records arrive in *release* order, not timeline order — sort by `start` if you
need it.

### One audio file, one command

`scripts/run_inference.sh` skips the config: it probes the audio length itself and
fills in the protocol defaults, so nothing has to be kept in sync by hand.

```bash
export SPEAKERLAB_ROOT=/path/to/3D-Speaker
./scripts/run_inference.sh /data/EN2002b.wav --out runs/en2002b
```

The same one-shot inference is available as a plain script,
`python scripts/infer.py --help`; both expect the speech LLM to be serving
already. Transcript lines are printed and written to `<out>/hypothesis.json`.

## Demo

TODO

## Evaluation

```bash
python -m eval.run_eval \
    --manifest data/ami_ihm.json --run-root runs/ami_ihm \
    --speaker-model <dir> --speakerlab-root <dir> \
    --endpoint http://127.0.0.1:19300/v1
```

This writes `runs/ami_ihm/hypothesis.json` and `reference.json`. Defaults are the
protocol behind the reported results: `C=60s`, `Δ=2s`. The scorer itself is not
bundled.

Every system is scored with [MeetEval](https://github.com/fgnt/meeteval) under one
protocol: punctuation removed, Chinese scored by character and English by word,
each recording initialised and scored on its own. We report

- **WER / CER** — labels ignored; recognition alone.
- **cpWER / cpCER** — concatenated per speaker under the error-minimizing
  permutation; recognition and attribution together. **Primary metric.**
- **DER** — missed speech, false alarms, and confusion, at a 0-s collar with
  overlap scored.

The manifest is one JSON list of recordings:

```json
[{"audio_name": "rec_001", "audio_path": "/abs/rec_001.wav",
  "segments": [{"start": 0.0, "end": 1.9, "speaker": "A", "text": "..."}]}]
```

`segments` is the reference. The streaming run never sees it — it is copied out
for the scorer, which is what keeps the evaluation causal.

## Results

Streaming, causal, matched scoring. Lower is better. Each cell reports the three
metrics in the order given in its language header — `WER / cpWER / DER` for
English, `CER / cpCER / DER` for Chinese — with **bold** marking the best
streaming score and `—` an unavailable result. SF denotes Streaming Sortformer.

| System | AISHELL-4 | AliM. test | AliM. eval | AMI | CH109 | Fisher |
|:---|---:|---:|---:|---:|---:|---:|
| NeMo MT | — | — | — | **20.77** / **19.07** / 33.81 | 22.66 / 18.57 / 36.32 | 23.78 / 21.21 / 33.09 |
| Nemotron+SF | 27.23 / 54.77 / 28.65 | 37.06 / 39.92 / **25.75** | 38.50 / 41.55 / **26.82** | 29.10 / 30.82 / 38.86 | 21.54 / 22.50 / 18.33 | 20.63 / 20.94 / **18.05** |
| JEDIS-LLM† | — | — | — | — | — / 19.09 / — | — / 16.60 / — |
| Paraformer+SF | 24.82 / 59.30 / 41.54 | 32.43 / 37.37 / 35.34 | 33.36 / 40.47 / 37.49 | 54.73 / 59.32 / 46.16 | 53.31 / 56.37 / 29.13 | 54.16 / 54.41 / 30.96 |
| **LEGO-Stream** | **17.62** / **27.11** / **18.97** | **29.19** / **34.84** / 28.03 | **25.62** / **39.89** / 26.95 | 23.37 / 30.98 / **33.20** | **15.86** / **15.90** / **18.15** | **16.68** / **15.85** / 20.81 |

† Published results on the same telephone settings, not rescored here.

VibeVoice-ASR-Streaming-7B is the closest streaming counterpart, but its released
checkpoints cap input at 480 s, so that comparison spans the first 480 s of each
recording rather than the whole file — hence the different numbers. Each cell
reads `WER / cpWER / DER` for English corpora and `CER / cpCER / DER` for
Chinese, as above.

| Corpus | LEGO-Stream | VibeVoice-Stream |
|:---|---:|---:|
| AISHELL-4 | **12.13** / **21.86** / 15.70 | 19.09 / 22.35 / **12.18** |
| AliM. test | **23.74** / **31.22** / **21.82** | 33.83 / 39.04 / 33.55 |
| AliM. eval | **23.56** / **28.88** / **18.29** | 38.15 / 42.09 / 32.00 |
| AMI | **22.82** / **30.46** / **31.48** | 25.06 / 54.96 / 48.16 |
| CH109 | **18.02** / 18.29 / **19.98** | 19.93 / **18.14** / 29.50 |
| Fisher | **17.44** / **16.58** / 21.24 | 19.10 / 17.60 / **21.01** |

**Runtime.** One stream on a single A40, with the speaker encoder on a separate
A40: mean per-recording RTF **0.74**, worst case 0.83, median commit delay about
4 s, with no speculation.

## Optional features

Both are off by default, and neither is needed for the results above.

### Calibration

An optional post-release stage that re-examines each label once, at the moment it
is released. It is off everywhere unless a caller asks for it — the library, the
runner, the demo, and `configs/stream.example.json` all default to off. Enable it
with `--calibration` on `eval.run_eval`, at `γ=0.05`, `κ=0.50`.

### Speculative decoding

Chunks are 2 s apart inside a 60 s window, so consecutive windows share almost
all their audio and the previous response is a strong proposal for the next one.
The target model still verifies every proposed token before it can reach the
frontier, so this does not change what the system is.

**This is a server-side setting.** Restart the endpoint with stock vLLM's suffix
speculation — no patch, no draft model, no second checkpoint:

```bash
vllm serve ./checkpoints/moss-transcribe-diarize \
    --served-model-name moss-transcribe-diarize \
    --trust-remote-code --dtype bfloat16 --port 19300 \
    --speculative-config '{"method":"suffix","num_speculative_tokens":16}'
```

Nothing in this repository needs changing. Confirm it took by looking for
`SpeculativeConfig(method='suffix', …)` in the server log. Measured **2.22× median
speedup** on a single A40 with a 16-token suffix budget.

Optionally also set the client hint, `"inference": { "causal_frontier_request_ids":
true }` or `--crfs` on `eval.run_eval`. It stamps a per-recording `X-Request-Id`
carrying the chunk timestamp, for a server that keys its suffix cache to the exact
frontier. Stock vLLM ignores the header, and the proposer that reads it is not
part of this release.


## Limitation

**Decoding is not bit-reproducible.** vLLM does not guarantee batch invariance by
default, so the same request can come back with slightly different turn boundaries
depending on what else is being served. A turn's speaker embedding comes from a
specific span of its audio, so those shifts can occasionally nudge a speaker
assignment: recognition is essentially unaffected, cpWER may drift a little
between runs over the same input. Treat the numbers above as one draw, not a fixed
value.

## Citation

```bibtex
@inproceedings{legostream,
  title     = {LEGO-Stream: Local-to-Global Speaker Tracking for Streaming
               Speaker-Attributed ASR with a Speech LLM},
  author    = {Zhang, Ruiyu and Wang, Huaxuan and Li, Yingjie and Duan, Yitao},
  booktitle = {Under review},
  year      = {2026}
}
```

Apache 2.0 — see [LICENSE](LICENSE).
