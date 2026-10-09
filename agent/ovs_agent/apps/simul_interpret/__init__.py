"""simul_interpret — simultaneous(-ish) speech interpretation from streaming
ASR. Loaded by the CLI as ``apps.simul_interpret.app:App``.
"""
from .app import App, SimulInterpretApp

__all__ = ["App", "SimulInterpretApp"]
