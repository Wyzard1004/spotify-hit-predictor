"""
Digital Signal Processing Feature Extraction
Extracts audio features from processed audio files using librosa and scipy.
"""

import librosa
import numpy as np
import logging
from typing import Dict, Tuple
from pathlib import Path

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts comprehensive audio features using Digital Signal Processing.

    Features extracted:
    - Temporal: Tempo, Beat Strength, Onset Rate
    - Spectral: Centroid, Spread, Rolloff, Contrast
    - Timbral: MFCC coefficients (13)
    - Energy: RMS Energy, Entropy, Zero Crossing Rate
    - Harmonic: Chroma features (12 bins)
    """

    def __init__(self, sr: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
        """
        Initialize FeatureExtractor.

        Args:
            sr: Sample rate (Hz)
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.logger = logging.getLogger(__name__)

    # ==================== Temporal Features ====================

    def extract_tempo(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract tempo information using beat tracking.

        Args:
            y: Audio time series

        Returns:
            Dictionary with tempo features
        """
        try:
            # Estimate tempo in BPM
            onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
            tempo, _ = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)

            return {
                "tempo_bpm": float(tempo) if isinstance(tempo, np.ndarray) else tempo,
                "tempo_normalized": float(tempo / 200.0),  # Normalize by typical max BPM
            }
        except Exception as e:
            self.logger.warning(f"Error extracting tempo: {str(e)}")
            return {"tempo_bpm": 0.0, "tempo_normalized": 0.0}

    def extract_beat_strength(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract beat strength using low-frequency STFT analysis.

        Args:
            y: Audio time series

        Returns:
            Dictionary with beat strength features
        """
        try:
            # Compute STFT
            D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            S = np.abs(D) ** 2

            # Extract energy in low-frequency range (bass, 0-250 Hz)
            freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            bass_idx = freq_bins < 250
            bass_energy = np.mean(S[bass_idx, :])

            # Spectral flux (rate of change of spectrum) indicates beat strength
            spectral_flux = librosa.onset.onset_strength(y=y, sr=self.sr)
            flux_mean = np.mean(spectral_flux)
            flux_std = np.std(spectral_flux)

            return {
                "bass_energy": float(bass_energy),
                "beat_strength": float(flux_mean),
                "beat_strength_std": float(flux_std),
                "beat_regularity": float(flux_std / (flux_mean + 1e-6)),  # Avoid division by zero
            }
        except Exception as e:
            self.logger.warning(f"Error extracting beat strength: {str(e)}")
            return {
                "bass_energy": 0.0,
                "beat_strength": 0.0,
                "beat_strength_std": 0.0,
                "beat_regularity": 0.0,
            }

    # ==================== Spectral Features ====================

    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features: centroid, spread, rolloff, contrast.

        Args:
            y: Audio time series

        Returns:
            Dictionary with spectral features
        """
        try:
            # Spectral Centroid (brightness, in Hz)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            centroid_mean = np.mean(spectral_centroid)
            centroid_std = np.std(spectral_centroid)

            # Spectral Rolloff (frequency below which most energy is concentrated)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            rolloff_mean = np.mean(spectral_rolloff)

            # Spectral Contrast (relative strength of different frequency bands)
            spectral_contrast = librosa.feature.spectral_contrast(
                y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            contrast_mean = np.mean(spectral_contrast, axis=1)

            return {
                "spectral_centroid": float(centroid_mean),
                "spectral_centroid_std": float(centroid_std),
                "spectral_centroid_normalized": float(centroid_mean / (self.sr / 2)),
                "spectral_rolloff": float(rolloff_mean),
                "spectral_rolloff_normalized": float(rolloff_mean / (self.sr / 2)),
                **{f"spectral_contrast_{i}": float(val) for i, val in enumerate(contrast_mean)},
            }
        except Exception as e:
            self.logger.warning(f"Error extracting spectral features: {str(e)}")
            return {}

    # ==================== Timbral Features (MFCC) ====================

    def extract_mfcc(self, y: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCC).

        MFCC captures timbral qualities of audio - what makes instruments sound different.

        Args:
            y: Audio time series
            n_mfcc: Number of MFCC coefficients to extract

        Returns:
            Dictionary with MFCC statistics
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=y, sr=self.sr, n_mfcc=n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
            )

            # Extract mean and std for each MFCC coefficient
            features = {}
            for i in range(n_mfcc):
                features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
                features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

            return features
        except Exception as e:
            self.logger.warning(f"Error extracting MFCC: {str(e)}")
            return {}

    # ==================== Energy Features ====================

    def extract_energy_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract energy-related features: RMS energy, entropy, zero crossing rate.

        Args:
            y: Audio time series

        Returns:
            Dictionary with energy features
        """
        try:
            # RMS Energy (loudness)
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)

            # Zero Crossing Rate (indicator of noisiness/high-frequency content)
            zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)

            # Energy entropy (measure of spectral distribution)
            energy = rms ** 2
            energy_normalized = energy / (np.sum(energy) + 1e-10)
            entropy = -np.sum(energy_normalized * np.log(energy_normalized + 1e-10))

            return {
                "rms_energy_mean": float(rms_mean),
                "rms_energy_std": float(rms_std),
                "zero_crossing_rate": float(zcr_mean),
                "zero_crossing_rate_std": float(zcr_std),
                "energy_entropy": float(entropy),
            }
        except Exception as e:
            self.logger.warning(f"Error extracting energy features: {str(e)}")
            return {}

    # ==================== Harmonic Features ====================

    def extract_chroma_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract Chroma features (distribution of energy across 12 musical notes).

        Args:
            y: Audio time series

        Returns:
            Dictionary with chroma features
        """
        try:
            # Constant-Q transform chroma features
            chroma = librosa.feature.chroma_cqt(y=y, sr=self.sr, hop_length=self.hop_length)

            # Extract statistics for each chroma bin
            features = {}
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

            for i, note in enumerate(note_names):
                features[f"chroma_{note}_mean"] = float(np.mean(chroma[i]))
                features[f"chroma_{note}_std"] = float(np.std(chroma[i]))

            return features
        except Exception as e:
            self.logger.warning(f"Error extracting chroma features: {str(e)}")
            return {}

    # ==================== Complete Feature Extraction ====================

    def extract_all_features(self, y: np.ndarray, track_id: str = "") -> Dict[str, float]:
        """
        Extract all audio features in one call.

        Args:
            y: Audio time series
            track_id: Optional track identifier for logging

        Returns:
            Dictionary with all extracted features
        """
        all_features = {"track_id": track_id}

        # Temporal
        all_features.update(self.extract_tempo(y))
        all_features.update(self.extract_beat_strength(y))

        # Spectral
        all_features.update(self.extract_spectral_features(y))

        # Timbral
        all_features.update(self.extract_mfcc(y, n_mfcc=13))

        # Energy
        all_features.update(self.extract_energy_features(y))

        # Harmonic
        all_features.update(self.extract_chroma_features(y))

        return all_features

    def extract_from_file(self, audio_path: Path, track_id: str = "") -> Dict[str, float]:
        """
        Extract features directly from an audio file.

        Args:
            audio_path: Path to processed audio file
            track_id: Optional track identifier

        Returns:
            Dictionary with extracted features
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
            return self.extract_all_features(y, track_id=track_id or audio_path.stem)
        except Exception as e:
            self.logger.error(f"Error extracting features from {audio_path}: {str(e)}")
            return {"track_id": track_id or audio_path.stem}


def create_feature_extractor(
    sr: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH
) -> FeatureExtractor:
    """
    Factory function to create FeatureExtractor instance.

    Args:
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT

    Returns:
        FeatureExtractor instance
    """
    return FeatureExtractor(sr=sr, n_fft=n_fft, hop_length=hop_length)
