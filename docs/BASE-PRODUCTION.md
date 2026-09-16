# Replace CustomVoice with one Base worker

This cutover serves registered, enrolled voices from one 1.7B Base model on the
RX 7900 XT. It does not run CustomVoice alongside Base. Each registered speaker
has a verified 4096-byte embedding held in host memory; requests do not re-encode
reference audio. Registration checks checkpoint and profile hashes before GPU
startup. The HTTP client can select a name, not supply an embedding or file path.

CustomVoice weights and its original binary/configuration stay on disk for
rollback. They are **not loaded** by the Base service. No LLM, ASR, Music3, firewall,
LiteLLM configuration, or credential is modified by this workflow.

## 1. Pull, build the isolated production binary, and enroll Will

Run on `.242`. Use tmux (`tmux new -s base-production`) if desired.

```bash
(
set -Eeuo pipefail
cd /home/jdillman/qingming-qwen3-tts
git pull --ff-only
cmake -S . -B build/base-production-1.7b \
  -DQINGMING_DEVICE=rx7900xtx-24g \
  -DQINGMING_MODEL_FAMILY=1.7b \
  -DROCM_PATH=/opt/rocm-10.0.0 \
  -DHIP_CLANG=/opt/rocm-10.0.0/lib/llvm/bin/clang++ \
  -DHOST_CXX=/opt/rocm-10.0.0/lib/llvm/bin/clang++
cmake --build build/base-production-1.7b -j 8
)
```

Wait for Music3/ASR GPU activity to finish. Then enroll Will with the existing,
tested offline binary. The helper pauses active TTS, runs four once-path samples,
and requests restoration of TTS on exit. It does not overwrite an existing ID.
The last argument is the RX 7900 XT HIP index, previously confirmed as `0`:

```bash
cd /home/jdillman/qingming-qwen3-tts
bash scripts/enroll_voice.sh \
  /home/jdillman/will_arnett_reference.wav will-arnett-v1 0
```

Listen to the samples before registering Will. This selects the first 15 seconds
of the reference. A successful technical test does not judge similarity or audio
quality. The helper's consent prompt reads `/dev/tty` to avoid pasted-command
input being mistaken for an answer.

## 2. Register approved voices and choose a default

This assumes the seven profiles were enrolled under the IDs below. Registration
fails before changing the service if any profile is absent, incomplete, altered,
or from another checkpoint. Remove an entry if you do not want that voice served.
Use only voices you have reviewed and are authorized to use.

```bash
(
set -Eeuo pipefail
cd /home/jdillman/qingming-qwen3-tts
read -rp 'Default voice (Samuel, Matthew, Nathan, Scarlett, Neil, Kristen, Will): ' BASE_DEFAULT </dev/tty
.venv/bin/python scripts/register_base_voices.py \
  --default "$BASE_DEFAULT" \
  --voice Samuel=samuel-reference-v1 \
  --voice Matthew=matthew-mcconaughey-v1 \
  --voice Nathan=nathan-fillion-v1 \
  --voice Scarlett=scarlett-johansson-v1 \
  --voice Neil=neil-patrick-harris-v1 \
  --voice Kristen=kristen-bell-v1 \
  --voice Will=will-arnett-v1
)
```

The private registry is `voice-library/production.json`; it is not published.
It is created exclusively, or an identical existing registry is validated again.
A different existing registry is not overwritten implicitly. Changes later need
explicit editing/revalidation and a service restart to refresh the snapshot.

## 3. Validate and deploy

This takes TTS offline during the gate. Do not run competing Music3/ASR work.
The script is for the existing service on port **8880**, with credentials in
`/etc/qingming-tts/api-key`, and refuses an already installed Base drop-in.

```bash
cd /home/jdillman/qingming-qwen3-tts
bash scripts/deploy_base_voices.sh 0
```

The gate performs all once controls first, then loads **one** resident Base model:

- Each registered voice: once versus resident PCM equality, natural EOS, and
  streamed native audio versus saved WAV equality.
- Voice switching through the registry and returning to the first voice.
- A forced one-frame generation limit followed by an identical successful
  control, without replacing the worker. This is a short recovery probe, not
  full-budget or endurance qualification.
- Streamed/nonstreamed API PCM equality and decodable default MP3 through the
  in-process ASGI application. Native chunk arrival is timed; the in-process
  HTTP transport is not a browser time-to-first-audio benchmark.

Only PASS permits installation of a new
`/etc/systemd/system/qingming-tts.service.d/50-base-voices.conf` override.
The installer checks that the report matches the binary, selected GPU index,
model files, registry, and embedding snapshots. It keeps the original unit,
environment file, credential, and CustomVoice binary/model untouched.

The deployment script starts the service, checks authenticated Base readiness,
and checks an actual local HTTP MP3 response. If a normal error occurs, it moves
only its newly installed drop-in into the private results directory, reloads
systemd, and requests restoration of TTS if it was initially active. Inspect the
journal/readiness if restoration fails; a power loss or `kill -9` cannot run a
shell cleanup trap. Results contain voice audio and are private, gitignored files.

## 4. LiteLLM and Open WebUI

The endpoint, key, and upstream `tts-1` model name stay the same. Keep the LiteLLM
public model `qingming-tts` mapping to `openai/tts-1` at `.242:8880/v1`.
In Open WebUI, replace `Aiden` with one of:

`Samuel`, `Matthew`, `Nathan`, `Scarlett`, `Neil`, `Kristen`, `Will`, or `default`.

Names are case-insensitive. `default`, `alloy`, and `echo` resolve to the selected
registry default. `Aiden` is rejected rather than secretly mapped to a different
person. `GET /v1/voices` lists the registered names and default, without private
paths or embeddings. Base rejects `instruct`/`instructions`; remove any previous
CustomVoice style setting from the client/proxy. Audio format and true PCM
streaming support remain unchanged.

Verify the existing proxy and read-aloud path after the local cutover:

```bash
cd /home/jdillman/qingming-qwen3-tts
.venv/bin/python scripts/check_speech.py \
  --base-url http://192.168.0.54:4000/v1 \
  --model qingming-tts --voice default --fixture narration
```

Enter the LiteLLM virtual key at its hidden prompt, not the backend credential.
Listen for complete speech, then try two named voices and a multi-paragraph
Open WebUI response. Proxy/browser behavior and prolonged use are not established
by the native/in-process gate alone.

## Roll back without deleting model weights or voice profiles

This moves only our exact Base override into a dated root-private backup, then
starts the original CustomVoice configuration. No reference recordings, profiles,
model files, or other systemd overrides are deleted.

```bash
(
set -Eeuo pipefail
sudo -v
BASE_DROPIN=/etc/systemd/system/qingming-tts.service.d/50-base-voices.conf
sudo test -f "$BASE_DROPIN"
BASE_BACKUP=$(sudo mktemp -d /etc/qingming-tts/base-rollback.XXXXXX)
sudo systemctl stop qingming-tts.service
sudo mv -- "$BASE_DROPIN" "$BASE_BACKUP/50-base-voices.conf"
sudo systemctl daemon-reload
sudo systemctl start qingming-tts.service
echo "Base override retained in $BASE_BACKUP"
sudo systemctl status qingming-tts.service --no-pager -l
)
```

Set the Open WebUI voice back to `Aiden` after rollback. Keeping the old checkpoint
on disk makes this possible without another download; deleting it is a separate,
explicit disk-cleanup decision, not part of production model replacement.
