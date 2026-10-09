"""Audio helpers for actuator/voice apps.

- :class:`~.tapped_audio_io.TappedAudioIO`: AudioIO subclass with
  multi-consumer capture taps + multi-channel mic downmix (reSpeaker).
- :mod:`.devices`: PortAudio device discovery (reSpeaker auto-select).

These are optional add-ons used by ``apps.voice_arm``; importing this
package does not pull in ``sounddevice``/``pyaudio`` until a submodule is
actually imported.
"""
