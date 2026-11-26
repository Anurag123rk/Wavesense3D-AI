"""WaveSense 3D AI - WiFi CSI human sensing for ESP32-S3 + NVIDIA Jetson, with an LLM assistant.

Pipeline:
    ESP32-S3 nodes (CSI over UDP) -> wavesense.server (Jetson) -> DSP + neural model -> 3D dashboard
"""

__version__ = "3.0.0"
APP_NAME = "WaveSense 3D AI"
