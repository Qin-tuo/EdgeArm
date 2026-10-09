from ovs_agent.config import load_config


def test_slv_config_env_defaults_keep_numeric_types(tmp_path, monkeypatch):
    for name in (
        "TTS_SPEED",
        "TTS_SPEAKER_ID",
        "SAMPLE_RATE",
        "VAD_SILENCE_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "agent.yaml"
    path.write_text(
        "slv_config:\n"
        "  tts_speed: ${TTS_SPEED:-1.0}\n"
        "  tts_speaker_id: ${TTS_SPEAKER_ID:-52}\n"
        "  sample_rate: ${SAMPLE_RATE:-16000}\n"
        "  vad_silence_ms: ${VAD_SILENCE_MS:-600}\n",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.slv_config["tts_speed"] == 1.0
    assert isinstance(cfg.slv_config["tts_speed"], float)
    assert cfg.slv_config["tts_speaker_id"] == 52
    assert isinstance(cfg.slv_config["tts_speaker_id"], int)
    assert cfg.slv_config["sample_rate"] == 16000
    assert isinstance(cfg.slv_config["sample_rate"], int)
    assert cfg.slv_config["vad_silence_ms"] == 600
    assert isinstance(cfg.slv_config["vad_silence_ms"], int)
