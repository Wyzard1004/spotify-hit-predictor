"""
Main Pipeline Orchestrator
Demonstrates how to use the audio processing, feature extraction, and Spotify integration modules.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict
import numpy as np

from src.audio_processor import create_audio_processor
from src.feature_extractor import create_feature_extractor
from src.spotify_client import create_spotify_client
from src.config import AUDIO_DIR, DATASETS_DIR, SAMPLE_RATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main orchestrator for the complete data processing pipeline.

    Workflow:
    1. Load audio files (multiple formats supported)
    2. Process and validate audio
    3. Extract DSP features using FFT/STFT
    4. Fetch Spotify metadata and available audio analysis
    5. Merge all data into feature matrix
    """

    def __init__(self):
        """Initialize pipeline with all components."""
        self.audio_processor = create_audio_processor(sample_rate=SAMPLE_RATE)
        self.feature_extractor = create_feature_extractor()
        
        # Spotify client is optional (requires API credentials)
        try:
            self.spotify_client = create_spotify_client()
            self.has_spotify = True
        except ValueError as e:
            logger.warning(f"Spotify integration unavailable: {e}")
            self.has_spotify = False

    def process_audio_directory(self, audio_dir: Path) -> pd.DataFrame:
        """
        Process all audio files in a directory.

        Args:
            audio_dir: Path to directory containing audio files

        Returns:
            DataFrame with extracted features for all successfully processed files
        """
        logger.info(f"Scanning directory: {audio_dir}")
        
        audio_files = []
        for audio_format in [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]:
            audio_files.extend(audio_dir.glob(f"*{audio_format}"))

        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return pd.DataFrame()

        logger.info(f"Found {len(audio_files)} audio files")

        # Process audio files in batch
        batch_result = self.audio_processor.batch_process(
            audio_files, show_progress=True, validate=True
        )

        # Extract features from processed audio
        all_features = []
        
        for (y, sr), file_path in zip(batch_result["processed"], batch_result["file_paths"]):
            track_id = file_path.stem
            features = self.feature_extractor.extract_all_features(y, track_id=track_id)
            all_features.append(features)

        df_features = pd.DataFrame(all_features)
        logger.info(f"Extracted features for {len(df_features)} tracks")

        return df_features

    def enrich_with_spotify_data(
        self, df_features: pd.DataFrame, track_metadata: List[Dict]
    ) -> pd.DataFrame:
        """
        Enrich audio features with Spotify metadata and available audio analysis.

        Args:
            df_features: DataFrame with extracted audio features
            track_metadata: List of dicts with 'track_name' and 'artist_name'

        Returns:
            DataFrame with merged audio features and Spotify data
        """
        if not self.has_spotify:
            logger.warning("Spotify client not available. Skipping Spotify enrichment.")
            return df_features

        logger.info("Fetching Spotify metadata...")
        df_spotify = self.spotify_client.search_tracks_batch(track_metadata)

        # Merge on track ID or name
        if "track_id" in df_features.columns and "track_id" in df_spotify.columns:
            df_merged = pd.merge(
                df_features, df_spotify, on="track_id", how="left", suffixes=("_audio", "_spotify")
            )
        else:
            logger.warning("Could not match audio features with Spotify data - no common identifier")
            df_merged = df_features.copy()

        logger.info(f"Merged dataset shape: {df_merged.shape}")
        return df_merged

    def save_features(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Save extracted features to CSV.

        Args:
            df: DataFrame with features
            output_path: Path to save CSV
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

    def generate_feature_report(self, df: pd.DataFrame) -> str:
        """
        Generate a summary report of extracted features.

        Args:
            df: DataFrame with features

        Returns:
            String containing feature report
        """
        report = f"""
=== FEATURE EXTRACTION REPORT ===
Total Tracks: {len(df)}
Total Features: {len(df.columns)}

Feature Categories:
- Temporal: tempo_bpm, tempo_normalized, beat_strength, beat_regularity
- Spectral: spectral_centroid, spectral_rolloff, spectral_contrast_*
- Timbral (MFCC): mfcc_*_mean, mfcc_*_std (13 coefficients)
- Energy: rms_energy_mean, rms_energy_std, zero_crossing_rate, energy_entropy
- Harmonic: chroma_* (12 note bins)
- Spotify: popularity, explicit, tempo, key, mode, genres (if available)

Data Summary:
{df.describe()}

Missing Values:
{df.isnull().sum()}
"""
        return report


# ==================== Example Usage ====================

def example_workflow():
    """
    Example workflow demonstrating the complete pipeline.
    """
    logger.info("Starting Spotify Hit Predictor pipeline...")

    # Initialize pipeline
    pipeline = DataPipeline()

    # Example 1: Process audio files from a directory
    logger.info("\n--- STEP 1: Processing Audio Files ---")
    # Uncomment to process actual audio files:
    # df_features = pipeline.process_audio_directory(AUDIO_DIR)

    # For demonstration, create dummy features
    df_features = pd.DataFrame({
        "track_id": ["track_001", "track_002"],
        "tempo_bpm": [128.5, 95.3],
        "spectral_centroid": [2150.4, 1980.2],
        "rms_energy_mean": [0.35, 0.42],
        "mfcc_0_mean": [-550.2, -480.1],
    })
    logger.info(f"Demo features shape: {df_features.shape}")

    # Example 2: Enrich with Spotify data
    logger.info("\n--- STEP 2: Enriching with Spotify Data ---")
    # Uncomment to fetch real Spotify data (requires API credentials):
    # track_metadata = [
    #     {"track_name": "Blinding Lights", "artist_name": "The Weeknd"},
    #     {"track_name": "Shape of You", "artist_name": "Ed Sheeran"},
    # ]
    # df_enriched = pipeline.enrich_with_spotify_data(df_features, track_metadata)

    df_enriched = df_features.copy()
    logger.info(f"Enriched dataset shape: {df_enriched.shape}")

    # Example 3: Save features
    logger.info("\n--- STEP 3: Saving Features ---")
    output_path = DATASETS_DIR / "audio_features_demo.csv"
    pipeline.save_features(df_enriched, output_path)

    # Example 4: Generate report
    logger.info("\n--- STEP 4: Feature Report ---")
    report = pipeline.generate_feature_report(df_enriched)
    print(report)

    logger.info("\nPipeline complete!")


if __name__ == "__main__":
    example_workflow()
