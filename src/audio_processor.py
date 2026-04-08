"""
Audio Processing Module
Handles loading, processing, and preparing audio files for feature extraction.
Supports multiple audio formats: MP3, WAV, M4A, FLAC, OGG, AAC, WMA, AIFF.
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm
import tempfile
import os
from pydub import AudioSegment

from config import (
    SAMPLE_RATE,
    PREVIEW_DURATION,
    AUDIO_DIR,
    SUPPORTED_AUDIO_FORMATS,
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
)

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles loading and processing audio files in multiple formats.
    Normalizes audio to standard sample rate and duration for consistent feature extraction.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, preview_duration: int = PREVIEW_DURATION):
        """
        Initialize AudioProcessor.

        Args:
            sample_rate: Target sample rate (Hz). Default: 22050
            preview_duration: Expected preview duration in seconds. Default: 30
        """
        self.sample_rate = sample_rate
        self.preview_duration = preview_duration
        self.logger = logging.getLogger(__name__)

    def load_audio(self, file_path: Path, mono: bool = True) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio file in any supported format.

        Handles MP3, WAV, M4A, FLAC, OGG, AAC, WMA, AIFF formats.
        Automatically converts to target sample rate.

        Args:
            file_path: Path to audio file
            mono: Convert to mono (True) or keep stereo (False). Default: True

        Returns:
            Tuple of (audio_time_series, sample_rate) or None if loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        suffix = file_path.suffix.lower()

        # Check if format is supported
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            self.logger.warning(f"Unsupported format: {suffix}. Supported: {list(SUPPORTED_AUDIO_FORMATS.keys())}")
            return None

        try:
            # For most formats, librosa handles directly
            if suffix in [".wav", ".flac", ".ogg"]:
                y, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=mono)
                return y, sr

            # For formats that librosa may struggle with, use pydub as intermediary
            elif suffix in [".mp3", ".m4a", ".aac", ".wma"]:
                return self._load_with_pydub(file_path, mono)

            # AIFF support
            elif suffix == ".aiff":
                y, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=mono)
                return y, sr

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return None

    def _load_with_pydub(self, file_path: Path, mono: bool = True) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio using pydub for better MP3/M4A/AAC support.

        Args:
            file_path: Path to audio file
            mono: Convert to mono

        Returns:
            Tuple of (audio_time_series, sample_rate) or None if loading fails
        """
        try:
            # Load with pydub
            suffix = file_path.suffix.lower()
            if suffix == ".m4a":
                format_type = "ipod"
            elif suffix == ".aac":
                format_type = "aac"
            else:
                format_type = suffix[1:].lower()

            audio = AudioSegment.from_file(str(file_path), format=format_type)

            # Convert to WAV in temporary file, then load with librosa
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                audio.export(tmp_path, format="wav")

            # Load the temporary WAV file
            y, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=mono)

            # Clean up temporary file
            os.unlink(tmp_path)

            return y, sr

        except Exception as e:
            self.logger.error(f"Error loading {file_path} with pydub: {str(e)}")
            return None

    def validate_audio(self, y: np.ndarray, sr: int) -> bool:
        """
        Validate audio for quality and duration constraints.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            True if audio meets requirements, False otherwise
        """
        # Calculate duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)

        # Check duration constraints
        if duration < MIN_AUDIO_DURATION:
            self.logger.warning(f"Audio too short: {duration:.2f}s (min: {MIN_AUDIO_DURATION}s)")
            return False

        if duration > MAX_AUDIO_DURATION * 1.2:  # Allow 20% tolerance
            self.logger.warning(f"Audio too long: {duration:.2f}s (max: {MAX_AUDIO_DURATION}s)")
            return False

        # Check for silence (very low RMS energy)
        rms = librosa.feature.rms(y=y)[0]
        if np.mean(rms) < 1e-4:
            self.logger.warning(f"Audio appears to be silent")
            return False

        return True

    def normalize_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Normalize audio amplitude to [-1, 1] range.

        Args:
            y: Audio time series
            sr: Sample rate (not used, but kept for consistency)

        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(y))
        if max_val > 0:
            return y / max_val
        return y

    def crop_to_duration(self, y: np.ndarray, sr: int, duration: int) -> np.ndarray:
        """
        Crop audio to specified duration.

        Args:
            y: Audio time series
            sr: Sample rate
            duration: Target duration in seconds

        Returns:
            Cropped audio
        """
        samples = duration * sr
        if len(y) > samples:
            y = y[:samples]
        return y

    def process_audio_file(
        self,
        file_path: Path,
        validate: bool = True,
        normalize: bool = True,
        crop: bool = True,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Complete audio processing pipeline.

        Steps:
        1. Load audio in any supported format
        2. Validate audio quality
        3. Normalize amplitude
        4. Crop to preview duration

        Args:
            file_path: Path to audio file
            validate: Validate audio quality (True)
            normalize: Normalize amplitude (True)
            crop: Crop to preview duration (True)

        Returns:
            Tuple of (processed_audio, sample_rate) or None if processing fails
        """
        # Load audio
        result = self.load_audio(file_path, mono=True)
        if result is None:
            return None

        y, sr = result

        # Validate
        if validate and not self.validate_audio(y, sr):
            return None

        # Normalize
        if normalize:
            y = self.normalize_audio(y, sr)

        # Crop to duration
        if crop:
            y = self.crop_to_duration(y, sr, self.preview_duration)

        return y, sr

    def batch_process(
        self,
        file_paths: list,
        show_progress: bool = True,
        validate: bool = True,
    ) -> dict:
        """
        Process multiple audio files in batch.

        Args:
            file_paths: List of paths to audio files
            show_progress: Show progress bar
            validate: Validate audio

        Returns:
            Dictionary with results:
            {
                "processed": [(audio, sr), ...],
                "file_paths": [Path, ...],
                "failed": [Path, ...]
            }
        """
        results = {
            "processed": [],
            "file_paths": [],
            "failed": [],
        }

        iterator = tqdm(file_paths, disable=not show_progress, desc="Processing audio files")

        for file_path in iterator:
            result = self.process_audio_file(file_path, validate=validate)
            if result is not None:
                results["processed"].append(result)
                results["file_paths"].append(Path(file_path))
            else:
                results["failed"].append(Path(file_path))

        self.logger.info(
            f"Batch processing complete: {len(results['processed'])} successful, "
            f"{len(results['failed'])} failed"
        )

        return results

    def save_processed_audio(self, y: np.ndarray, sr: int, output_path: Path) -> bool:
        """
        Save processed audio to WAV file.

        Args:
            y: Audio time series
            sr: Sample rate
            output_path: Path to save audio

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), y, sr)
            return True
        except Exception as e:
            self.logger.error(f"Error saving audio: {str(e)}")
            return False


def create_audio_processor(sample_rate: int = SAMPLE_RATE) -> AudioProcessor:
    """
    Factory function to create AudioProcessor instance.

    Args:
        sample_rate: Target sample rate in Hz

    Returns:
        AudioProcessor instance
    """
    return AudioProcessor(sample_rate=sample_rate)
