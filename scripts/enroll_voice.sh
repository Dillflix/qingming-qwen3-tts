#!/usr/bin/env bash
# Pauses only this service; never deletes/replaces an existing voice profile.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# != 3 ]]; then
    echo "Usage: bash scripts/enroll_voice.sh /absolute/reference.wav new-voice-id HIP-device-index" >&2
    exit 2
fi
CLONE_REFERENCE=$1
CLONE_ID=$2
CLONE_GPU=$3
[[ "$CLONE_REFERENCE" == /* && -f "$CLONE_REFERENCE" ]]
[[ "$CLONE_ID" =~ ^[a-z][a-z0-9_-]{0,47}$ && "$CLONE_GPU" =~ ^[0-9]+$ ]]
[[ ! -e "voice-library/$CLONE_ID" ]]
test -x build/base-clone-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b
command -v ffmpeg >/dev/null
read -rp 'I own this voice or have permission to clone it (type YES): ' CLONE_CONSENT </dev/tty
[[ "$CLONE_CONSENT" == YES ]]
sudo -v
CLONE_STATE=$(systemctl is-active qingming-tts.service || true)
case "$CLONE_STATE" in active|inactive|failed) ;; *) echo "Unexpected service state: $CLONE_STATE"; exit 1 ;; esac
restore_tts() {
    local CLONE_EXIT=$?
    trap - EXIT
    if [[ "$CLONE_STATE" == active ]]; then
        if sudo systemctl start qingming-tts.service; then
            echo 'TTS start requested; check service readiness.'
        else
            echo 'ERROR: TTS restoration failed; run sudo systemctl start qingming-tts.service' >&2
            CLONE_EXIT=1
        fi
    fi
    exit "$CLONE_EXIT"
}
trap restore_tts EXIT
if [[ "$CLONE_STATE" == active ]]; then sudo systemctl stop qingming-tts.service; fi
python3 scripts/clone_voice.py enroll --voice-id "$CLONE_ID" --reference "$CLONE_REFERENCE" \
    --start-seconds 0 --duration-seconds 15 --hip-device "$CLONE_GPU" --consent-confirmed
