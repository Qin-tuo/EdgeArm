#!/usr/bin/env bash
set -euo pipefail

if [[ -f /etc/openvoicestream.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/openvoicestream.env
  set +a
elif [[ -f /etc/seeed-local-voice.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/seeed-local-voice.env
  set +a
fi

if [[ -z "${OVS_PROFILE:-}" && -z "${OVS_PROFILE_JSON:-}" ]]; then
  if [[ -n "${OVS_PROFILE_DEFAULT:-}" ]]; then
    export OVS_PROFILE="${OVS_PROFILE_DEFAULT}"
  else
    case "${LANGUAGE_MODE:-zh_en}" in
      zh_en)
        export OVS_PROFILE="jetson-zh-en"
        ;;
      multilanguage)
        export OVS_PROFILE="jetson-qwen3asr-matcha-nx"
        ;;
    esac
  fi
else
  # When OVS_PROFILE is explicitly set (by user or OVS_PROFILE_DEFAULT),
  # the profile is the single source of truth for LANGUAGE_MODE /
  # ASR_BACKEND / TTS_BACKEND. Unset any image-baked or env-inherited
  # values so profile_loader can inject the profile's intended values
  # without tripping its CRITICAL_KEYS conflict guard (see
  # app/core/profile_loader.py:CRITICAL_KEYS).
  unset LANGUAGE_MODE ASR_BACKEND TTS_BACKEND
fi

exec "$@"
