# Supervised TTS service, LiteLLM Proxy and Open WebUI

For replacement with one Base worker and enrolled voices, see
[BASE-PRODUCTION.md](BASE-PRODUCTION.md). This page describes the original
CustomVoice installation retained as the rollback path.

The intended route is **Open WebUI → LiteLLM `/v1/audio/speech` → Qingming**.
The existing chat/LLM route does not change. This is CustomVoice on the RX 7900 XT;
the iGPU LLM service is not stopped, rebuilt, or reconfigured by these scripts.

## Evidence and remaining host check

On the RX 7900 XT, commit `cde9734` passed the native resident/once WAV comparison,
the default HTTP MP3 listening check, and the SDK MP3/PCM check. The reported
warm streamed PCM was byte-identical to non-streamed PCM: first audio 316.66 ms,
completion 857.09 ms, audio duration 2.4 s. This is a short warm sample, not a
latency guarantee or endurance test. The new recovery/service layer has local
fault-injection tests; systemd and the user's actual proxy/UI must still be
checked on the Linux host. No new HIP build is needed for this Python-only update.

## Install on the host

Use the existing non-root account that successfully ran the native benchmark.
`--host 0.0.0.0` accepts LAN connections; use loopback instead if the proxy shares
the host network. A container's `127.0.0.1` refers to that container, not this host.
Choose a strong **backend** API key when prompted and keep it for LiteLLM.

```bash
(
set -Eeuo pipefail
cd /home/jdillman/qingming-qwen3-tts
git pull --ff-only
.venv/bin/python -m pip install -r requirements-api.txt
sudo /usr/bin/python3 scripts/install_service.py \
  --user jdillman --host 0.0.0.0 --port 8880 --install
)
```

The installer verifies the rendered unit with the host's `systemd-analyze`
before changing system configuration. It writes only `/etc/systemd/system/qingming-tts.service` and
`/etc/qingming-tts/{environment,api-key}`, then reloads systemd's unit definitions.
It does **not** start or stop anything. The key is root-readable (0600), delivered
to the service via [systemd `LoadCredential`](https://systemd.io/CREDENTIALS/), not embedded in the public unit or command
line. Reinstall preserves the key and environment and backs up the previous
unit. Omit `--install` to preview nonsecret configuration. An existing key file
can be supplied with `--api-key-file /absolute/path` on first install.

If the successful foreground launch used a GPU-selection override, add the same
`--hip-device INDEX` at installation; do not guess an index from its marketing
name. Otherwise the native default is retained and refuses a non-gfx1100 GPU.
Systemd does not inherit exports from your interactive shell. To change an
installed setting, edit `sudoedit /etc/qingming-tts/environment`, then restart
**only** `qingming-tts.service`. Do not add `QINGMING_API_KEY` there: the installed
unit already selects its credential file and conflicting key sources are rejected.

Stop the existing foreground TTS API with **Ctrl+C in its own tmux pane** first.
Check that port 8880 is free; do not kill unrelated processes or the LLM service:

```bash
sudo ss -ltnp 'sport = :8880'
```

After the previous TTS instance has exited:

```bash
sudo systemctl enable --now qingming-tts.service
sudo systemctl status qingming-tts.service --no-pager
sudo journalctl -u qingming-tts.service -n 60 --no-pager
```

Wait for `CustomVoice worker ready` in the journal. The API binds before native
loading completes. Then run the short authenticated MP3 check; it asks for the
backend key without echo and saves audio plus a report in `benchmark-customvoice-check.*`:

```bash
cd /home/jdillman/qingming-qwen3-tts
.venv/bin/python scripts/check_speech.py --base-url http://127.0.0.1:8880/v1
```

Only `/healthz` is unauthenticated. `/readyz` requires the same bearer key and
returns 503 during model startup/recovery, 200 when ready. The check also accepts
`--api-key-file` or `QINGMING_TEST_API_KEY`; never paste real keys into verbose
curl logs. LAN HTTP is unencrypted: restrict reachability to the proxy/trusted
LAN, or terminate TLS before traffic crosses an untrusted network. This installer
does not change the firewall. Do not disable SELinux to resolve a startup failure;
inspect the unit journal and AVC records first.

## Recovery behavior

Clean audio-budget exhaustion is a **request failure, not a worker outage**.
The adapter returns buffered callers HTTP 422 (`speech_generation_limit`) and
keeps the worker resident after validating the complete native terminal frame and
audio length. It does not replay the failed text or silently return truncated
audio. Queued requests can run immediately after response cleanup. Actual native
failures and incomplete protocol exchanges still follow the restart policy below.

This reliability update is Python-only: no HIP rebuild, dependency change, new
generation limit, or systemd unit reinstall. Local tests exercise the real parser,
MP3/WAV/PCM handling, one failed request followed by five queued successes, retained
readiness, malformed completions, and aborted streaming bodies. They do not prove
GPU state reuse on the host.

For a short native reuse check, run the following during a quiet window, with the
production TTS worker stopped to avoid loading a second copy beside it. The script
does not control services; it starts and closes only its own native process:

```bash
cd /home/jdillman/qingming-qwen3-tts
.venv/bin/python scripts/validate_customvoice.py --limit-recovery-only
```

It obtains a short baseline, forces an **8-frame request** inside the existing
512-frame resident capacity, verifies the expected length error without a process
replacement, then checks five natural-EOS controls against the baseline PCM.
The forced-limit WAV is deliberately incomplete diagnostic evidence, not a valid
speech response. The report and audio/logs are archived even on failure. This is
not full-512-frame exhaustion, HTTP/LiteLLM, or endurance qualification. Restore
the previously running TTS service after testing; inspect any failure before
considering the update qualified. Use the same `--hip-device` selection as your
working service if it has an explicit override.

- One HTTP worker and one owned native process; requests/segments are serialized.
- EOF, native error, timeout, or interrupted PCM generation discards native state.
  A supervisor reaps the old worker, waits for request/encoder cleanup, and starts
  a fresh one. It never replays an interrupted request or changes another service.
- The failed request returns 502 before headers, or an incomplete stream if audio
  already started. New requests receive 503 while recovering. A queued request
  rechecks readiness before entering the engine. The existing queue limits remain
  eight waiters / 30 seconds; long requests may cause a 429 queue timeout.
- Recovery backs off 1, 2, 4, … up to 30 seconds; a worker that stays ready for
  60 seconds resets the backoff. Repeated invalid configuration remains unavailable
  and logged, not falsely healthy. `/healthz` reports HTTP liveness, not GPU readiness.
- Stopping the unit stops its whole process group, including its FFmpeg/native
  children. A Python API process failure is handled by systemd `Restart=on-failure`.
  An ordinary intentional native shutdown is no longer logged as an unexpected crash.

### Optional native-crash recovery check (during a quiet window)

This intentionally interrupts TTS. First run the MP3 check above. Then find the
**one direct native child of this unit's MainPID** and signal only that child:

```bash
(
set -Eeuo pipefail
cd /home/jdillman/qingming-qwen3-tts
API_PID=$(systemctl show qingming-tts.service --property=MainPID --value)
[[ "$API_PID" =~ ^[1-9][0-9]*$ ]] || { echo 'TTS API is not running'; exit 1; }
export QINGMING_CHECK_API_PID="$API_PID"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import signal

parent = int(os.environ['QINGMING_CHECK_API_PID'])
expected = (Path.cwd() / 'build/rx7900xtx-24g-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b').resolve()
children = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
matches = [int(pid) for pid in children if Path(f'/proc/{pid}/exe').resolve() == expected]
if len(matches) != 1:
    raise SystemExit(f'Refusing to signal: expected one owned native worker, found {len(matches)}')
pid = matches[0]
fd = os.pidfd_open(pid)  # Pin the target so PID reuse cannot signal another process.
try:
    status = Path(f'/proc/{pid}/status').read_text().splitlines()
    ppid = next(int(line.split()[1]) for line in status if line.startswith('PPid:'))
    if ppid != parent or Path(f'/proc/{pid}/exe').resolve() != expected:
        raise SystemExit('Worker identity changed; refusing to signal')
    signal.pidfd_send_signal(fd, signal.SIGKILL)
finally:
    os.close(fd)
print('Signalled only the owned TTS native worker. Wait for replacement readiness in the journal.')
PY
)
```

Watch `journalctl -fu qingming-tts.service`; once the replacement is ready,
rerun `scripts/check_speech.py`. The API MainPID should be unchanged. Stopping the
unit must leave no child processes in its cgroup. These checks do not simulate a
GPU-driver hang; some driver faults can require operator intervention.

## LiteLLM Proxy

Merge the entry in [`deployment/litellm-qingming.yaml`](../deployment/litellm-qingming.yaml)
into your existing `model_list`; retain all chat models and authentication settings.
Configure in the **LiteLLM process/container**:

```text
QINGMING_API_BASE=http://<TTS-host-LAN-IP-or-service-name>:8880/v1
QINGMING_API_KEY=<Qingming backend key>
```

`model_name: qingming-tts` is the public proxy name; `model: openai/tts-1` selects
the compatible speech adapter and sends `tts-1` to this local server. It does not
send speech to OpenAI. Keep the explicit `api_base`; do not append `/audio/speech`
to it. Allow `qingming-tts` on the LiteLLM virtual key used by Open WebUI. Review
existing proxy retries/fallbacks: do not add an unintended cloud fallback, and
never automatically replay a partially played stream. Generic `tts-1` cloud
pricing may need a local accounting override in your LiteLLM deployment; it is
not an actual OpenAI charge.

After applying the proxy configuration using your existing deployment method,
test from an environment that can reach the proxy. Supply the **LiteLLM client
key**, not the backend key, at the prompt:

```bash
cd /home/jdillman/qingming-qwen3-tts
.venv/bin/python scripts/check_speech.py \
  --base-url http://<LiteLLM-host>:4000/v1 --model qingming-tts --voice echo
```

Use your actual proxy scheme/address/port. This separates service failures from
proxy routing/auth failures before involving the UI. The recipe follows the
[LiteLLM speech endpoint](https://docs.litellm.ai/docs/text_to_speech) and
[compatible endpoint configuration](https://docs.litellm.ai/docs/providers/openai_compatible).

## Open WebUI

In **Admin Panel → Settings → Audio → Text-to-Speech**:

| Setting | Value |
|---|---|
| Engine | OpenAI (compatible API transport, not a model choice) |
| API Base URL | Your **LiteLLM** base URL, ending in `/v1` |
| API Key | LiteLLM key authorized for `qingming-tts` |
| Model | `qingming-tts` |
| Voice | `Aiden` (native preset name; no alias needed) |
| OpenAI Params | `{"response_format":"mp3"}` |
| Response splitting | Paragraphs |

Keep MP3 for browser playback; do not set `stream: true` or raw PCM for this path.
The API's tested true PCM streaming remains available to purpose-built clients.
Open WebUI's Punctuation setting fragments text into separate HTTP requests;
the backend cannot recombine them. Use **Paragraphs** so more context reaches
each request, then let the adapter apply its 400-character cap. This is not
native PCM streaming, and larger MP3 requests can delay first playback. Avoid
None for arbitrary long responses: the HTTP input limit is still 4096 characters
per request (including a single oversized paragraph).

Native speaker names such as `Aiden` and `Ryan` work directly at our endpoint;
`echo` remains a compatibility alias for Ryan, and `alloy` maps to Vivian.
Other hardcoded OpenAI voices (`nova`, `fable`, etc.) are not implicitly
mapped to nonexistent checkpoint speakers. Per-model/user voice overrides in
Open WebUI can override the system default: update those too if applicable.

If using initial environment configuration instead of the UI:

```text
AUDIO_TTS_ENGINE=openai
AUDIO_TTS_OPENAI_API_BASE_URL=http://<LiteLLM-host>:4000/v1
AUDIO_TTS_OPENAI_API_KEY=<LiteLLM client key>
AUDIO_TTS_MODEL=qingming-tts
AUDIO_TTS_VOICE=Aiden
AUDIO_TTS_OPENAI_PARAMS={"response_format":"mp3"}
```

As described in the [Open WebUI environment reference](https://docs.openwebui.com/reference/env-configuration/),
existing persisted admin settings can take precedence over environment defaults;
set **Response splitting → Paragraphs** in the saved Audio panel after initial
configuration. Don't change the chat model name or chat provider
hint. Test the speaker button on a short answer first, then a multi-sentence
answer and cancellation followed by another request.

The direct API supports `language`, `instruct` and `instructions`. Custom fields
are not guaranteed to survive every LiteLLM version's speech filtering; establish
plain MP3 first, then test style forwarding explicitly rather than silently
claiming it works. The guide is based on [Open WebUI's TTS integration and
response splitting docs](https://docs.openwebui.com/features/chat-conversations/audio/text-to-speech/openai-tts-integration/).
The user's installed proxy/UI versions have not been exercised locally.

## Validate longer segments

The segmentation update is Python-only: no HIP rebuild, model conversion, unit
reinstallation or dependency update is needed. Once the update is published,
pull it and restart **only TTS**, during a quiet window (active speech will be
interrupted):

```bash
(
set -Eeuo pipefail
cd /home/jdillman/qingming-qwen3-tts
git pull --ff-only
sudo systemctl restart qingming-tts.service
sudo journalctl -u qingming-tts.service -n 30 --no-pager
)
```

Wait for the new `CustomVoice worker ready` journal entry, then run the fixed
multi-sentence narration through LiteLLM. Enter the **LiteLLM client key**:

```bash
cd /home/jdillman/qingming-qwen3-tts
.venv/bin/python scripts/check_speech.py \
  --base-url http://192.168.0.54:4000/v1 \
  --model qingming-tts --voice Aiden --fixture narration
```

The checker sends one HTTP request and saves `speech.mp3` and `report.json` under
the printed results directory. The fixture needs two native segments at the
400-character cap. A transport PASS only means a decodable MP3 was returned;
listen for all the text, especially the final words "at its own pace", steady
pace/tone, and the segment join. This does not test Open WebUI's own splitting:
also try a new multi-sentence answer there with Paragraphs selected. A live
RX 7900 XT listening check is still required; local fake-worker tests cannot
qualify prosody, language-specific duration or native stability.
