"""
Spotify Hit Predictor - Audio Processing Module
Core infrastructure for audio processing, feature extraction, and data management.
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    AUDIO_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    SAMPLE_RATE,
    SUPPORTED_AUDIO_FORMATS,
)
from .audio_processor import AudioProcessor, create_audio_processor
from .feature_extractor import FeatureExtractor, create_feature_extractor
from .spotify_client import SpotifyClient, create_spotify_client

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "AUDIO_DIR",
    "DATASETS_DIR",
    "MODELS_DIR",
    "SAMPLE_RATE",
    "SUPPORTED_AUDIO_FORMATS",
    "AudioProcessor",
    "create_audio_processor",
    "FeatureExtractor",
    "create_feature_extractor",
    "SpotifyClient",
    "create_spotify_client",
]

__version__ = "0.1.0"
