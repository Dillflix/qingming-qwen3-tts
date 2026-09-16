# Offline Base voice library — RX 7900 XT

Prepare and validate voices now; later replace the CustomVoice checkpoint with
one 1.7B Base checkpoint serving those voices. This workflow does **not** alter
the HTTP API, LiteLLM, Open WebUI, systemd unit, or production binary. It does not
require keeping CustomVoice and Base loaded together.

The current native implementation uses **x-vector-only** cloning: the reference
is encoded into a 2048-dimensional BF16 speaker embedding. This is not training,
not a new copy of the model per voice, and not reference-transcript/audio-code
in-context learning. Base does not accept CustomVoice `instruct` controls.
Similarity, delivery, and pronunciation need listening review. A short reference
transcript is not used by this path.

## 1. Build and download, without touching the running service

After pulling the commit containing this workflow, run on `.242`:

```bash
(
set -Eeuo pipefail
cd /home/jdillman/qingming-qwen3-tts
git pull --ff-only

# A separate binary: do not rebuild the active CustomVoice service's build directory.
cmake -S . -B build/base-clone-1.7b \
  -DQINGMING_DEVICE=rx7900xtx-24g \
  -DQINGMING_MODEL_FAMILY=1.7b \
  -DROCM_PATH=/opt/rocm-10.0.0 \
  -DHIP_CLANG=/opt/rocm-10.0.0/lib/llvm/bin/clang++ \
  -DHOST_CXX=/opt/rocm-10.0.0/lib/llvm/bin/clang++
cmake --build build/base-clone-1.7b -j 8

# Separate download environment; no changes to the production API's .venv.
python3 -m venv .venv-clone
.venv-clone/bin/python -m pip install huggingface_hub
.venv-clone/bin/hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base

command -v ffmpeg
python3 scripts/clone_voice.py enroll --help
)
```

The historical target name contains `xtx`, but the binary targets `gfx1100` and
the once-path trial works on the RX 7900 XT. No resident XTX CU partition is used.
Compilation and downloads can affect other workloads' latency; do not benchmark
during either operation. This workflow needs disk space for the additional Base
checkpoint, but only one TTS checkpoint needs to be resident at a time.

## 2. Choose a reference

Use your own voice or a speaker who has permitted this use. Start with 10–20
seconds of clean, single-speaker speech: no music, competing speakers, clipping,
or long silences. Use the delivery you want to hear. This is guidance for the
first test, not a claim that longer references always improve similarity.

The script selects a local clip and converts it to mono PCM16 at 24 kHz with
FFmpeg. It does not automatically denoise, amplify, or alter speaking speed.
It checks duration (3–30 seconds), gross silence/clipping, and WAV integrity;
these checks cannot establish that the clip contains only one speaker or clean
speech. Listen to `reference.wav` as well as the generated samples.

## 3. Enroll with only Base running during the GPU trial

Open a tmux session if desired: `tmux new -s base-voices`. Wait for Music3/ASR GPU
work to finish before this test. No other services will be stopped by the block
below. It stops **only** TTS if TTS was active and restores that service on exit,
including a failed enrollment. A power loss or `kill -9` cannot run the trap;
in that case restore explicitly with `sudo systemctl start qingming-tts.service`.

Run the entire block, not fragments:

```bash
(
set -Eeuo pipefail
cd /home/jdillman/qingming-qwen3-tts

read -rp 'Absolute path to reference audio on this host: ' CLONE_REFERENCE
test -f "$CLONE_REFERENCE"
read -rp 'New lowercase voice ID (example: jason-v1): ' CLONE_ID
test -n "$CLONE_ID"
read -rp 'RX 7900 XT HIP device index (previously 0): ' CLONE_GPU
[[ "$CLONE_GPU" =~ ^[0-9]+$ ]]
read -rp 'I own this voice or have permission to clone it (type YES): ' CLONE_CONSENT
[[ "$CLONE_CONSENT" == YES ]]
sudo -v

CLONE_TTS_STATE=$(systemctl is-active qingming-tts.service || true)
case "$CLONE_TTS_STATE" in
  active|inactive|failed) ;;
  *) echo "Unexpected TTS state: $CLONE_TTS_STATE; no service changes made."; exit 1 ;;
esac
restore_clone_tts() {
  local CLONE_EXIT_STATUS=$?
  trap - EXIT
  if [[ "$CLONE_TTS_STATE" == active ]]; then
    if sudo systemctl start qingming-tts.service; then
      echo 'TTS start requested; check /readyz or the service journal for readiness.'
    else
      echo 'ERROR: restore TTS with sudo systemctl start qingming-tts.service' >&2
      CLONE_EXIT_STATUS=1
    fi
  else
    echo 'TTS was not active before the trial; not starting it.'
  fi
  exit "$CLONE_EXIT_STATUS"
}
trap restore_clone_tts EXIT
if [[ "$CLONE_TTS_STATE" == active ]]; then
  sudo systemctl stop qingming-tts.service
fi

python3 scripts/clone_voice.py enroll \
  --voice-id "$CLONE_ID" \
  --reference "$CLONE_REFERENCE" \
  --start-seconds 0 --duration-seconds 15 \
  --hip-device "$CLONE_GPU" \
  --consent-confirmed
)
```

Change the selection start/length if needed. If the recording ends before 15
seconds, the actual clip is accepted provided it is at least 3 seconds. Leave
out `--duration-seconds` only for a whole reference no longer than 30 seconds.
The script never retries a failed generation automatically.

Enrollment performs four sequential **once** requests, so only one Base process
runs at a time:

1. Reference-conditioned control, exporting its speaker embedding.
2. Same text/seed using only the saved embedding, with no reference WAV argument.
3. Conversational listening passage using the saved embedding.
4. Technical listening passage using the saved embedding.

It requires byte-identical generated PCM for the first pair, valid non-silent
audio, and natural EOS for every sample. These are reproducibility/completeness
checks, **not** automatic judgments of voice similarity or transcript accuracy.

Successful enrollment writes `voice-library/<id>/profile.json` last, alongside
`reference.wav` and `speaker.bf16`. Model weights, tokenizer/config files, payload
hashes, binary hash, normalization details, and consent acknowledgment are
recorded. Existing voice IDs are never overwritten. Failed attempts retain
their files but have no completed profile; inspect them and use a new ID when
retrying. No recursive deletion is performed.

Listen to `reference.wav` and the generated `reference.wav`, `saved-voice.wav`,
`conversation.wav`, and `technical.wav` in the printed results directory. Results
include native timing fields, audio duration, and wall time including model
loading. Once-path timing is not a resident-service latency benchmark, and native
TTFA is not browser/network time-to-first-audio. No VRAM/endurance qualification
is implied.

## 4. Audition more text without re-encoding the reference

Create a UTF-8 file containing a short test passage. In the same service-preserving
block above, replace only the `enroll` invocation with this command, supplying
the actual profile and text file paths:

```bash
python3 scripts/clone_voice.py audition \
  --profile /home/jdillman/qingming-qwen3-tts/voice-library/jason-v1 \
  --text-file /home/jdillman/voice-test.txt \
  --hip-device "$CLONE_GPU"
```

Profile/payload/checkpoint fingerprints are checked before GPU work. It reuses
the language selected during enrollment unless overridden. A changed checkpoint
requires re-enrollment; a changed executable is reported for deliberate
revalidation. A test that hits its generation budget fails rather than presenting
a truncated clip as complete. Changing the seed or reference requires new
listening review.

All profiles are private artifacts. New directories use mode 0700 and files use
a restrictive umask on Linux. The default library/results are gitignored.
`--archive` explicitly creates a **private** results archive with generated voice
audio, not the reference or embedding. Do not publish it or upload a voice library
to GitHub. Use a non-sensitive test passage if sending diagnostic results.

## Later production cutover (not implemented by this workflow)

After listening approval, extend the resident Base worker to load these
registered embeddings by voice ID, then test voice switching, natural EOS,
streaming, generation-limit recovery, and LiteLLM/Open WebUI read-aloud. Deploy
one Base checkpoint in place of CustomVoice only after those tests pass, keeping
the old service configuration available for rollback. The present HTTP API still
rejects cloning voices; adding a profile does not expose it publicly or switch
the current Aiden voice.
