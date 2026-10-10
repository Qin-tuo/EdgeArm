"""Defer PortAudio initialization until a real audio operation is requested."""
import importlib


class LazySoundDevice:
    def __getattr__(self, name):
        try:
            backend = importlib.import_module("sounddevice")
        except (ImportError, OSError) as exc:
            raise RuntimeError("sounddevice unavailable; install libportaudio2") from exc
        return getattr(backend, name)


sd = LazySoundDevice()
