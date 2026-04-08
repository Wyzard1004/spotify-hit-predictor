"""
Configuration management for the Spotify Hit Predictor project.
Handles environment variables, API keys, and project settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio_previews"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
for directory in [DATA_DIR, AUDIO_DIR, DATASETS_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Spotify API Configuration
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

# Audio Processing Configuration
SAMPLE_RATE = 22050  # Standard sample rate for audio processing
PREVIEW_DURATION = 30  # iTunes provides 30-second previews
HOP_LENGTH = 512  # For STFT computation
N_FFT = 2048  # FFT window size
N_MELS = 128  # Number of mel-frequency bands

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".flac": "flac",
    ".ogg": "ogg",
    ".aac": "aac",
    ".wma": "wma",
    ".aiff": "aiff",
}

# API Configuration
SPOTIFY_API_RATE_LIMIT = 50  # Requests per second
ITUNES_API_RATE_LIMIT = 20  # Requests per second
SPOTIFY_BATCH_SIZE = 50  # Number of tracks to fetch at once
AUDIO_DOWNLOAD_TIMEOUT = 30  # Seconds to wait for audio file download

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
CLASS_WEIGHTS = {0: 1.0, 1: 2.5}  # Weight hits heavier than non-hits

# Feature Configuration
MIN_AUDIO_DURATION = 10  # Minimum audio duration in seconds (exclude very short clips)
MAX_AUDIO_DURATION = 60  # Cap at preview duration
FEATURE_SCALING_METHOD = "standardscaler"  # Options: 'standardscaler', 'minmax'

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Billboard Scraping Configuration
BILLBOARD_BASE_URL = "https://www.billboard.com/charts/hot-100"
BILLBOARD_ARCHIVE_START_YEAR = 2020  # Scrape from this year onwards

# Data Balance Configuration
MAX_NON_HIT_SAMPLES = 5000  # Cap non-hit samples to avoid extreme imbalance
MIN_HIT_SAMPLES = 500  # Minimum hits to collect before training

print(f"""
Configuration Loaded:
- Project Root: {PROJECT_ROOT}
- Audio Directory: {AUDIO_DIR}
- Sample Rate: {SAMPLE_RATE} Hz
- FFT Window Size: {N_FFT}
- Hop Length: {HOP_LENGTH}
- Supported Formats: {', '.join(SUPPORTED_AUDIO_FORMATS.keys())}
""")
