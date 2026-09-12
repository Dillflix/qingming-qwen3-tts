# CustomVoice API

This adapter serves only `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. It uses one
persistent Qingming HIP process on gfx1100, with requests and text segments
executed sequentially. It never loads a Base checkpoint, extracts reference-audio
embeddings, or accepts a `ref_audio` field. `clone:` voices produce the explicit
`voice_cloning_not_supported` error. No voice-cloning endpoint is provided.

## Verified starting point

The user-supplied `benchmark-customvoice.gojZSW` archive was produced with commit
`5736b1d` and ROCm 10 on the user's RX 7900 XT. All three samples reached natural
EOS. The user confirmed that calm and cheerful delivery sounded as expected.

Subsequent native resident, HTTP MP3, and SDK PCM/MP3 checks passed on the RX 7900 XT
(HTTP testing was initially loopback without authentication).
See [deployment and LiteLLM/Open WebUI integration](DEPLOYMENT.md)
for the current evidence, service installation, and remaining host checks.

| Sample | Audio duration | Native TTFA | Native total |
|---|---:|---:|---:|
| Calm, first process | 2.560 s | 2.370 s | 2.897 s |
| Calm, second process | 2.560 s | 1.042 s | 1.694 s |
| Cheerful | 2.400 s | 0.945 s | 1.526 s |

The calm WAVs are byte-identical with SHA-256
`0f9ecd35133d17a492dba995ec2230abdf7f40a89131e6998401b296d1fdc01a`.
These are separate single-shot processes, **not resident or HTTP measurements**.

## Build and validate on the Linux host

Run from your checkout, currently `/home/jdillman/qingming-qwen3-tts`:

Python 3.11 or newer is required for the adapter and validator.

```bash
cmake -S . -B build/rx7900xtx-24g-1.7b \
  -DQINGMING_DEVICE=rx7900xtx-24g \
  -DQINGMING_MODEL_FAMILY=1.7b \
  -DROCM_PATH=/opt/rocm-10.0.0 \
  -DHIP_CLANG=/opt/rocm-10.0.0/lib/llvm/bin/clang++ \
  -DHOST_CXX=/opt/rocm-10.0.0/lib/llvm/bin/clang++ &&
cmake --build build/rx7900xtx-24g-1.7b --clean-first -j 8 &&
python3 -m venv .venv &&
.venv/bin/python -m pip install -r requirements-api.txt &&
.venv/bin/python scripts/validate_customvoice.py \
  --expected-calm-sha256 0f9ecd35133d17a492dba995ec2230abdf7f40a89131e6998401b296d1fdc01a
```

The validator runs two once requests followed by three requests through one
resident worker. It checks the pre-change calm hash, repeatability, resident/once
WAV equality, natural EOS, native float32 chunk equality with the accumulated
WAV, and reception of audio before the completion event. Logs, audio and the
report are archived even on failure. A different checkpoint revision can change
the baseline hash; do not suppress a mismatch without investigating it.

Keep the GPU-selection environment used for the successful baseline. You may
explicitly pass `--hip-device INDEX` to the validator and API. The new worker
reports its actual GPU name and architecture and refuses a non-gfx1100 device.
The historical executable/target name remains `rx7900xtx-24g`; it is not a claim
that the XT has 24 GB or the XTX's scheduler topology.

The API selects `--resident-cu-partition none`: existing shared HIP priority
streams, not the XTX-only 28/20 split. This does not change model kernels,
quantization or the single-shot path. Existing native CLI users retain the
default `xtx` partition unless they explicitly select `none`. Partition sizes of
zero in native result metadata mean no dedicated WGP reservation, not zero GPU use.

## Start the HTTP adapter

FFmpeg must be installed and provide MP3 (`libmp3lame`), Opus (`libopus`), AAC and
FLAC encoders. Validate native behavior first, then start in the foreground:

```bash
.venv/bin/python -m qingming_api --host 127.0.0.1 --port 8880
```

Startup creates only its own TTS worker; it does not stop or modify the LLM
service. `/healthz` is available while the model loads; wait for `/readyz` to
return HTTP 200 before speech requests. The native process stays resident across
requests. There is one worker, an eight-request waiting limit, a 30-second queue
wait, and a 180-second native per-segment deadline. Never run multiple HTTP
workers against this single-GPU configuration.

For LAN access, set `QINGMING_API_KEY` (preferred over a command-line secret),
then use `--host 0.0.0.0`. Non-loopback binding requires a key. Clients send
`Authorization: Bearer ...`; do not put real keys in verbose curl logs. Use a
TLS reverse proxy for untrusted networks. `--api-key-file` or
`QINGMING_API_KEY_FILE` is also supported. The systemd installer uses a protected
credential file. This version is not endurance-qualified.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/audio/speech` | Speech from preset speaker + style instruction |
| GET | `/v1/models` | Native name and accepted aliases |
| GET | `/v1/voices` | Checkpoint speakers, dialect metadata and valid aliases |
| GET | `/v1/audio/voices` | Same voice list |
| GET | `/healthz` | HTTP liveness, no authentication required |
| GET | `/readyz` | Resident readiness, 503 if unavailable |

Model aliases: `qwen3-tts`, `tts-1`, `tts-1-hd`,
`qingming-qwen3-tts-1.7b-customvoice`. Language suffixes such as `tts-1-en` are
advertised only if the checkpoint supports that language. These names route to
the same local model; they do not invoke OpenAI services or imply different model quality.

Speakers and dialects come from `talker_config.spk_id` and
`talker_config.spk_is_dialect`. Their config spelling is preserved and lookup is
case-insensitive. The supplied checkpoint has nine speakers; of the proposed
default aliases only `alloy -> vivian` and `echo -> ryan` resolve. Nonexistent
targets are not advertised. Use `--voice-aliases aliases.json` with a flat JSON
object to replace the alias table. Native names cannot be shadowed.

`input` is required, nonblank, at most 4096 characters. Defaults are model
`qwen3-tts`, voice `Vivian`, language `Auto`, `speed=1`, `stream=false`, and MP3.
`instruct` and OpenAI's `instructions` are accepted as equivalents; conflicting
values are rejected. Formats: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`.
Speed 0.25–4 is implemented with FFmpeg tempo filtering, not changed sample-rate
metadata. Text up to **400 characters** (after trimming) is passed intact to one
native generation, including embedded punctuation and newlines. Only longer text
is split: the existing algorithm chooses the last punctuation boundary within
the cap, then whitespace, then a Unicode-codepoint cut for unbroken text. Boundary
search starts at 20 characters; the final remainder can be shorter. This raises
the previous 70-character cap without changing the splitting algorithm. All
segments use the same speaker, language, instruction and seed (1234), with 120 ms
of silence between segments before speed adjustment. Separate generations still
reset acoustic state; fewer segments do not guarantee consistent prosody.

The native per-segment limit remains **512 audio frames**, approximately 40.96 s
at 1920 samples/frame and 24 kHz, not 512 text characters. The 400-character cap
is a starting point for English narration, not a guarantee for every language,
number-heavy text or slow delivery. Native completion must still report EOS;
exhaustion fails the request rather than silently accepting incomplete audio.
Post-generation `speed` adjustment does not increase the native generation budget.
The 4096-character HTTP input limit and 180-second native segment deadline are
unchanged. Longer segments may increase time to first playback, especially with
buffered MP3. Validate narration on the host as described in
[the deployment guide](DEPLOYMENT.md#validate-longer-segments).

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8880/v1", api_key="not-needed")
response = client.audio.speech.create(
    model="tts-1", voice="Ryan", input="Welcome to Dillflix, mailboxhead!",
    extra_body={"language": "English", "instruct": "Speak cheerfully."},
)
response.stream_to_file("speech.mp3")
```

### True streaming

Only `stream=true` with `response_format=pcm` is accepted initially. Compressed
streaming and optional SSE are explicitly rejected, not silently buffered or
misrepresented as implemented. `stream_format=audio` is accepted.

The native pipe uses float32, but public PCM is **signed 16-bit little-endian,
mono, 24 kHz**, matching the [official OpenAI speech format](https://developers.openai.com/api/docs/guides/text-to-speech#supported-output-formats).
Accordingly the original handoff's `ffplay -f f32le` must be `-f s16le` for HTTP:

```bash
curl --fail-with-body --no-buffer http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"Ryan","input":"Welcome to Dillflix, mailboxhead!","language":"English","instruct":"Speak cheerfully.","response_format":"pcm","stream":true}' |
ffplay -nodisp -autoexit -f s16le -ar 24000 -ac 1 -
```

Input validation errors use OpenAI-shaped JSON. The adapter checks for real audio
before returning streaming headers. An error after headers aborts the stream; it
cannot retroactively replace HTTP 200 with JSON. A disconnected/interrupted
generation terminates only the owned native worker and sets readiness false,
preventing reuse of partially updated state. The supervisor waits for cleanup,
then replaces that worker with bounded backoff. Queued requests recheck readiness.
Requests are never automatically replayed. See [recovery behavior](DEPLOYMENT.md#recovery-behavior).

## Native protocol

`--lifecycle resident --task custom-voice --protocol jsonl-v1 --audio-fd FD`
enables machine-only stdout, beginning with a JSON `ready` event. JSONL requests
contain `request_id`, `text`, `language`, `speaker`, `instruct`, `output`, `seed`,
`max_new_tokens`, and `stream_audio`. IDs and seeds are unsigned 32-bit integers;
the owner sends UTF-8 with `ensure_ascii=False`. Task/model changes and reference
audio are rejected. `{ "command": "shutdown" }` ends the worker.

The inherited writable FD (>=3) carries 24-byte little-endian headers:
`<4sIQII`: magic `QAF1`, kind, request ID, payload length, reserved=0. Kind 1 is
native float32 audio; kind 2 is completion JSON; kind 3 is error JSON. Payloads
are bounded at 1 MiB. A completion/error frame ends the request. A failed or
stalled pipe causes failure rather than intermixing bytes with stdout. The
optional callback runs immediately after `decoder->append(pending)` and before
the same samples are appended to the native accumulated waveform. Without the
callback, existing CLI WAV generation is unchanged.

## Local tests and limits of evidence

```bash
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover request validation, real FFmpeg encoding, the OpenAI Python client,
live streaming order with a fake engine, serialization, and native IPC glue
compiled against a stub HIP engine. CMake/Ninja and a C++20 compiler enable the
build/IPC tests (`QINGMING_TEST_CMAKE`, `QINGMING_TEST_NINJA`,
`QINGMING_TEST_CXX` override executable discovery). Mock or host-compiler passes
do not qualify the HIP binary, actual streaming latency, GPU memory behavior,
or long-running service stability. Run the host validator and HTTP smoke before
deployment.
