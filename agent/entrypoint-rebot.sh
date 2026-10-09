#!/bin/bash
# entrypoint-rebot.sh — Hardware probe + env setup before ovs-agent.
# Mirrors voice-arm's /app/entrypoint.sh for the reBot B601-DM deployment.
set -e

# ── PulseAudio: free the reSpeaker mic for exclusive ALSA capture ─────
SOCKET="${PULSE_SERVER#unix:}"
if [ -n "$SOCKET" ] && [ -S "$SOCKET" ] && timeout 2 pactl info >/dev/null 2>&1; then
    echo "[entrypoint] PulseAudio detected at $SOCKET"
    pactl list short sources 2>/dev/null \
      | awk '$2 ~ /(respeaker|xvf|C16K6Ch)/ {print $2}' \
      | while read -r src; do
            pactl suspend-source "$src" 1 >/dev/null 2>&1 \
              && echo "[entrypoint] Suspended pulse source $src"
        done
else
    unset PULSE_SERVER
    echo "[entrypoint] No PulseAudio"
fi

# ── Resolve MIC_INDEX to a concrete PortAudio integer ─────────────────
MIC_INDEX=$(python3 -c "
from ovs_agent.audio.devices import resolve_input_index
print(resolve_input_index('${MIC_INDEX:-reSpeaker}'))
" 2>/dev/null) || MIC_INDEX=0
export MIC_INDEX
echo "[entrypoint] MIC_INDEX=$MIC_INDEX"

# ── Resolve SPEAKER_DEVICE the same way ──────────────────────────────
SPEAKER_DEVICE=$(python3 -c "
from ovs_agent.audio.devices import resolve_output_index
print(resolve_output_index('${SPEAKER_DEVICE:-reSpeaker}'))
" 2>/dev/null) || SPEAKER_DEVICE=""
export SPEAKER_DEVICE
echo "[entrypoint] SPEAKER_DEVICE=$SPEAKER_DEVICE"

# ── Force MIC_CHANNELS=6 for reSpeaker XVF3800 (6ch exclusive USB) ───
export MIC_CHANNELS="${MIC_CHANNELS:-6}"
echo "[entrypoint] MIC_CHANNELS=$MIC_CHANNELS"

exec ovs-agent run voice_rebot_arm
