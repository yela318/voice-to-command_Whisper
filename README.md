# voice-to-command

Speech → text via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Give it an audio file or a mic recording, get back a string. Korean in → Korean
out, English in → English out; the language is detected per clip. Whatever
consumes the string is wired up elsewhere.

## Install

```bash
git clone https://github.com/yela318/voice-to-command.git
cd voice-to-command
pip install -e .          # add .[mic] for microphone input
```

Defaults assume a GPU. CTranslate2 needs the **CUDA 12** runtime — CUDA 13 does
not satisfy it:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Run

```bash
v2c sample/voice_kor.m4a   # file → text on stdout
v2c --listen               # mic (Enter to start, Enter to stop) → text
v2c --serve                # stay resident: load the model once, then loop
```

`v2c <file>` and `v2c --listen` reload the model every time (~5 s).
`v2c --serve` pays it once and keeps going. Same thing from Python:

```python
from voice_to_command import transcribe, record, warmup

warmup()                         # load once at startup
text = transcribe(record())      # mic
text = transcribe("clip.m4a")    # or a file
```

## Change the model

```bash
V2C_MODEL=small v2c clip.m4a
```

Default is `large-v3-turbo`. Must be multilingual — no `.en`, no `distil-*`.

## Run on CPU

```bash
V2C_DEVICE=cpu V2C_MODEL=small v2c clip.m4a
```

`V2C_DEVICE` is `auto`, so this is only needed to force it: with no usable
card — or one that fails to load — v2c falls back to CPU on its own and says
so on stderr. Pick `small` there; turbo is the slowest option on CPU.

On Windows PowerShell use `$env:V2C_DEVICE="cpu"` instead of the prefix form.

## Remote GPU

Run the model on the GPU box, drive it from a laptop that has none. The client
ships audio bytes over plain HTTP and prints what comes back — no auth, so use a
trusted LAN or an SSH tunnel.

```bash
v2c --serve --http                  # on the GPU box (0.0.0.0:8756)
v2c clip.m4a --server_ip gpu-box    # on the laptop
v2c --listen --server_ip gpu-box    # mic local, inference remote
```

## Config (env vars)

| var | default | |
|---|---|---|
| `V2C_MODEL` | `large-v3-turbo` | whisper size, or a local path |
| `V2C_DEVICE` | `auto` | GPU if there is one, else CPU. Force with `cuda` \| `cpu` |
| `V2C_COMPUTE` | `int8` | `auto` picks `float16` where the card offers it |
| `V2C_LANG` | auto | `ko` pins the source language |
| `V2C_MIC` | system default | input device index, or a substring of its name |
| `V2C_SERVER_IP` / `V2C_SERVER_PORT` | — / `8756` | remote inference host |
| `V2C_TIMING` | `1` | `0` silences the `[timing]` lines on stderr |

## Translation

Need English out instead? Add `--translate`. It mistranslates homonyms —
`사과 주세요` comes back "Please apologize".

```bash
v2c clip.m4a --translate
```

## Layout

```
src/voice_to_command/
  core.py      _load() caches the model; transcribe(path | samples) -> str
  capture.py   record() — push-to-talk mic, returns float32 samples
  remote.py    serve_http() + transcribe_remote() — inference on a GPU box
  cli.py       v2c <file> | --listen | --serve [--http] [--server_ip …]
sample/        TTS clips: voice_kor* / voice_eng_* pairs
```

MIT licensed — see [LICENSE](LICENSE).
