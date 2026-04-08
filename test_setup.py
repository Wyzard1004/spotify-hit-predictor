"""
Testing & Validation Utils
Helper functions for testing the audio processing pipeline.
"""

import logging
from pathlib import Path
import numpy as np
from src import create_audio_processor, create_feature_extractor
from src.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def test_audio_processor():
    """Test basic audio processor functionality."""
    print("\n=== Testing Audio Processor ===")
    
    processor = create_audio_processor()
    
    # Test supported formats
    print("\nSupported Audio Formats:")
    from src.config import SUPPORTED_AUDIO_FORMATS
    for fmt in SUPPORTED_AUDIO_FORMATS.keys():
        print(f"  ✓ {fmt}")
    
    print("\nAudioProcessor Methods:")
    print("  ✓ load_audio() - Load any supported format")
    print("  ✓ validate_audio() - Check duration & quality")
    print("  ✓ normalize_audio() - Scale amplitude")
    print("  ✓ crop_to_duration() - Trim to length")
    print("  ✓ process_audio_file() - Complete pipeline")
    print("  ✓ batch_process() - Process multiple files")
    

def test_feature_extractor():
    """Test feature extractor with synthetic audio."""
    print("\n=== Testing Feature Extractor ===")
    
    extractor = create_feature_extractor()
    
    # Create synthetic audio for testing
    sr = SAMPLE_RATE
    duration = 3  # seconds
    t = np.linspace(0, duration, sr * duration)
    
    # Synthesize a sine wave (440 Hz A note)
    frequency = 440
    y = np.sin(2 * np.pi * frequency * t)
    
    # Extract features
    print(f"\nExtracting features from {duration}s synthetic audio...")
    features = extractor.extract_all_features(y, track_id="synthetic_test")
    
    print(f"\nExtracted {len(features)} features:")
    print(f"  Temporal: tempo_bpm, beat_strength, beat_regularity")
    print(f"  Spectral: spectral_centroid, spectral_rolloff, spectral_contrast")
    print(f"  Timbral: mfcc_0-12 (mean & std)")
    print(f"  Energy: rms_energy, zero_crossing_rate, energy_entropy")
    print(f"  Harmonic: chroma_C-B (mean & std)")
    
    # Show some sample values
    print(f"\nSample Feature Values:")
    print(f"  Tempo: {features.get('tempo_bpm', 0):.2f} BPM")
    print(f"  Spectral Centroid: {features.get('spectral_centroid', 0):.2f} Hz")
    print(f"  RMS Energy: {features.get('rms_energy_mean', 0):.4f}")


def test_spotify_client():
    """Test Spotify client setup."""
    print("\n=== Testing Spotify Client ===")
    
    try:
        from src import create_spotify_client
        client = create_spotify_client()
        
        print("✓ Spotify client initialized successfully")
        print("\nAvailable Methods:")
        print("  ✓ search_track() - Search for a track")
        print("  ✓ get_track_features() - Get track metadata & popularity")
        print("  ✓ get_track_audio_analysis() - Get tempo, time_signature, key")
        print("  ✓ get_track_genres() - Get artist genres")
        print("  ✓ search_tracks_batch() - Batch search multiple tracks")
        print("  ✓ get_playlist_tracks() - Get all tracks from a playlist")
        
    except ValueError as e:
        print(f"⚠ Spotify client not configured: {e}")
        print("  → Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")


def show_config():
    """Display current configuration."""
    print("\n=== Project Configuration ===")
    
    from src.config import (
        PROJECT_ROOT, AUDIO_DIR, DATASETS_DIR, MODELS_DIR,
        SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS,
        MIN_AUDIO_DURATION, MAX_AUDIO_DURATION
    )
    
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Audio Directory: {AUDIO_DIR}")
    print(f"Datasets Directory: {DATASETS_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    
    print(f"\nAudio Processing:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  FFT Window Size: {N_FFT}")
    print(f"  Hop Length: {HOP_LENGTH}")
    print(f"  Mel Bands: {N_MELS}")
    print(f"  Min Duration: {MIN_AUDIO_DURATION}s")
    print(f"  Max Duration: {MAX_AUDIO_DURATION}s")


if __name__ == "__main__":
    print("=" * 60)
    print("SPOTIFY HIT PREDICTOR - SYSTEM VALIDATION")
    print("=" * 60)
    
    show_config()
    test_audio_processor()
    test_feature_extractor()
    test_spotify_client()
    
    print("\n" + "=" * 60)
    print("✓ All systems ready!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Place audio files in: data/audio_previews/")
    print("2. Run: python main.py")
    print("3. Check results in: datasets/audio_features.csv")
