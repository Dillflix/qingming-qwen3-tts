#!/usr/bin/env bash
# Validate, install only our new drop-in, restart TTS, and roll back on failure.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ $# == 1 && "$1" =~ ^[0-9]+$ ]] || { echo 'Usage: bash scripts/deploy_base_voices.sh HIP-device-index' >&2; exit 2; }
BASE_GPU=$1
BASE_DROPIN=/etc/systemd/system/qingming-tts.service.d/50-base-voices.conf
test -x .venv/bin/python
test -x build/base-production-1.7b/qingming-qwen3-tts_rx7900xtx-24g_1.7b
test -f voice-library/production.json
# Check both validator and service dependencies before any service interruption.
.venv/bin/python -c 'import uvicorn; from scripts import validate_base, configure_base_service' || {
    echo 'Install dependencies first: .venv/bin/python -m pip install -r requirements-base-production.txt' >&2
    exit 1
}
sudo -v
if sudo test -e "$BASE_DROPIN" || sudo test -L "$BASE_DROPIN"; then
    echo 'Base drop-in already exists; refusing an implicit redeployment.' >&2
    exit 1
fi
BASE_OLD_STATE=$(systemctl is-active qingming-tts.service || true)
case "$BASE_OLD_STATE" in active|inactive|failed) ;; *) echo "Unexpected service state: $BASE_OLD_STATE"; exit 1 ;; esac
umask 077
BASE_RESULTS=$(mktemp -d "$PWD/benchmark-base-production.XXXXXX")
BASE_INSTALLED=0
BASE_SUCCESS=0
finish_base_trial() {
    local BASE_EXIT=$?
    trap - EXIT
    if [[ "$BASE_SUCCESS" != 1 ]]; then
        echo 'Cutover did not complete; restoring previous service configuration.' >&2
        if [[ "$BASE_INSTALLED" == 1 ]]; then
            sudo systemctl stop qingming-tts.service || BASE_EXIT=1
            if sudo test -f "$BASE_DROPIN"; then
                sudo mv -- "$BASE_DROPIN" "$BASE_RESULTS/failed-base-dropin.conf" || BASE_EXIT=1
            fi
            sudo systemctl daemon-reload || BASE_EXIT=1
        fi
        if [[ "$BASE_OLD_STATE" == active ]]; then
            sudo systemctl start qingming-tts.service || BASE_EXIT=1
        fi
    fi
    echo "Private validation results: $BASE_RESULTS"
    exit "$BASE_EXIT"
}
trap finish_base_trial EXIT
sudo systemctl stop qingming-tts.service
.venv/bin/python scripts/validate_base.py --hip-device "$BASE_GPU" --output-dir "$BASE_RESULTS"
sudo .venv/bin/python scripts/configure_base_service.py --hip-device "$BASE_GPU" \
    --validation-report "$BASE_RESULTS/report.json" --install
BASE_INSTALLED=1
echo 'Starting Base TTS service...'
sudo systemctl start qingming-tts.service
echo 'Waiting up to 180 seconds for authenticated Base readiness; inspect journalctl -u qingming-tts.service if startup fails.'
sudo .venv/bin/python scripts/check_base_ready.py
# Runs as root only to read the existing credential file; it never prints the key.
sudo .venv/bin/python scripts/check_speech.py --base-url http://127.0.0.1:8880/v1 \
    --voice default --api-key-file /etc/qingming-tts/api-key
BASE_SUCCESS=1
echo 'Base-only TTS deployment passed local readiness and MP3 checks. No CustomVoice model is loaded by this service.'
echo 'Update the Open WebUI voice from Aiden to a registered name or default.'
