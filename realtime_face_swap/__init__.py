# Real-time Face Swap Extension for Wan2.2
# Copyright 2024-2025

__version__ = "1.0.0"
__author__ = "Real-time Face Swap Extension"
__description__ = "Sistema de intercambio de rostros en tiempo real usando Wan2.2-Animate"

from .webcam_capture import WebcamCapture, FramePreprocessor
from .face_swap_processor import RealtimeFaceSwap, AsyncFaceSwapProcessor
from .realtime_app import RealtimeFaceSwapApp

__all__ = [
    'WebcamCapture',
    'FramePreprocessor',
    'RealtimeFaceSwap',
    'AsyncFaceSwapProcessor',
    'RealtimeFaceSwapApp',
]
