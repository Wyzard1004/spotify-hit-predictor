# Spotify Hit Predictor - Audio Analysis & ML Pipeline

A machine learning project to predict whether a song will be a mainstream hit based purely on its audio features using Digital Signal Processing (DSP).

## Project Structure

```
spotify-hit-predictor/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration management & environment variables
│   ├── audio_processor.py          # Audio loading & processing (multi-format support)
│   ├── feature_extractor.py        # DSP feature extraction (FFT/STFT)
│   └── spotify_client.py           # Spotify API integration
├── data/
│   └── audio_previews/             # Downloaded audio files (added to .gitignore)
├── datasets/
│   ├── billboard_hits.csv          # Scrapped Billboard Hot 100 data
│   ├── spotify_metadata.csv        # Spotify metadata for tracks
│   ├── audio_features.csv          # Extracted audio features
│   └── processed_features.csv      # Cleaned & merged feature matrix
├── models/
│   ├── baseline_logistic_regression.pkl
│   ├── random_forest.pkl
│   └── adaboost.pkl
├── main.py                         # Main pipeline orchestrator
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── GAMEPLAN.md                     # Project gameplan (in .gitignore)
```

## Raw Data

Raw data files are not tracked in git. Download them manually and place in `data/raw/`:

- `hot100.csv` — [Billboard Hot 100 & more](https://www.kaggle.com/datasets/ludmin/billboard)
- `dataset.csv` — [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### 2. Configure Spotify API (Optional)

To use Spotify metadata enrichment, register for free at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard):

```bash
# Copy .env.example to .env
cp .env.example .env

# Edit .env with your Spotify credentials
nano .env
```

Set these values:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_secret
```

## Jupyter Notebooks

The project includes three comprehensive analysis notebooks in the `notebooks/` directory:

### 1. `01_data_acquisition.ipynb` - Data Collection

**Purpose:** Acquire and merge Billboard Hot 100 data with Spotify tracks dataset

**What it covers:**
- Load Billboard Hot 100 historical data (1958-present)
- Load Spotify Tracks Dataset from Kaggle
- Match tracks between Billboard and Spotify datasets
- Create binary hit/non-hit labels based on Billboard charting
- Handle missing values and duplicates
- Generate `merged_labeled_dataset.csv` for downstream analysis

**Key outputs:**
- Merged dataset with 2,100+ tracks
- Label distribution: 96.4% non-hits, 3.6% hits
- Track metadata (artist, genre, release date)

**Run time:** ~5 minutes

### 2. `02_feature_extraction.ipynb` - Audio Feature Extraction

**Purpose:** Extract 60+ audio DSP features from audio files

**What it covers:**
- Load audio previews (MP3, M4A, WAV formats)
- Extract temporal features (tempo, beat strength)
- Extract spectral features (spectral centroid, rolloff, contrast)
- Extract timbral features (MFCC coefficients)
- Extract energy features (RMS energy, zero crossing rate)
- Extract harmonic features (chroma features)
- Validate and normalize extracted features
- Generate `audio_features.csv` with all tracks and features

**Key outputs:**
- Feature matrix: 2,100+ tracks × 60+ features
- Quality validation metrics
- Feature statistics and distributions

**Run time:** ~10-30 minutes (depending on audio file count)

### 3. `03_EDA_and_Baseline.ipynb` - Exploratory Data Analysis & Baseline Model

**Purpose:** Analyze feature distributions, detect patterns, and train baseline model with PCA visualizations

**What it covers:**
- Load processed features dataset
- Generate class distribution visualization (class imbalance analysis)
- Create feature distribution plots (KDE plots by label)
- Compute feature correlation heatmap
- Perform Principal Component Analysis (PCA):
  - Scree plot showing variance explained per component
  - Biplot with feature loading vectors (red arrows)
  - Cumulative variance analysis
- t-SNE visualization for non-linear dimensionality reduction
- PCA component loadings analysis (PC1, PC2, PC3)
- Calculate effect sizes (Cohen's d) for feature importance
- Train baseline Logistic Regression model with class weighting
- Evaluate model performance (AUC, F1 score)

**Key outputs:**
- 9 comprehensive visualizations
- PCA insights (variance explained, feature relationships)
- Baseline model metrics (AUC: 0.776, F1: 0.799)
- Feature importance rankings

**Run time:** ~2 minutes (includes t-SNE computation)

### How to Run the Notebooks

```bash
# Activate virtual environment
cd /home/william/spotify-hit-predictor
source .venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/

# Or run a specific notebook
jupyter notebook notebooks/03_EDA_and_Baseline.ipynb
```

### Recommended Execution Order

1. **First:** `01_data_acquisition.ipynb` (only if acquiring new data)
2. **Second:** `02_feature_extraction.ipynb` (only if processing new audio)
3. **Third:** `03_EDA_and_Baseline.ipynb` (main analysis - run every time)

The processed datasets are already in `data/processed/features.csv`, so you can start with notebook 3 for immediate analysis.

## Module Overview

### `audio_processor.py` - Multi-Format Audio Loading

**Handles:** MP3, WAV, M4A, FLAC, OGG, AAC, WMA, AIFF

**Key Features:**
- Automatic format detection and conversion
- Sample rate normalization (default: 22050 Hz)
- Audio validation (duration, silence detection)
- Amplitude normalization
- Batch processing with progress tracking

**Usage:**
```python
from src import create_audio_processor

processor = create_audio_processor()

# Load single audio file
y, sr = processor.load_audio("song.mp3")

# Process with validation and normalization
y, sr = processor.process_audio_file("song.m4a", validate=True, normalize=True)

# Batch process multiple files
results = processor.batch_process(audio_files)
for (y, sr), file_path in zip(results["processed"], results["file_paths"]):
    print(f"Processed: {file_path}")
```

### `feature_extractor.py` - DSP Feature Extraction

**Extracts 60+ features** across multiple categories:

#### Temporal Features (4 features)
- `tempo_bpm`: Track tempo in beats per minute
- `beat_strength`: Energy in beat patterns
- `beat_regularity`: Consistency of rhythm

#### Spectral Features (6+ features)
- `spectral_centroid`: "Brightness" of audio (in Hz)
- `spectral_rolloff`: Frequency concentration
- `spectral_contrast`: Strength of different frequency bands

#### Timbral Features (26 features)
- `mfcc_0-12_mean`: Mel-Frequency Cepstral Coefficients
- `mfcc_0-12_std`: Statistics for each coefficient

#### Energy Features (5 features)
- `rms_energy_mean/std`: Overall loudness
- `zero_crossing_rate`: High-frequency content
- `energy_entropy`: Spectral distribution

#### Harmonic Features (24 features)
- `chroma_C/D/E/..._mean/std`: Distribution across 12 musical notes

**Usage:**
```python
from src import create_feature_extractor
import librosa

extractor = create_feature_extractor()

# Load audio
y, sr = librosa.load("song.mp3", sr=22050)

# Extract all features
features = extractor.extract_all_features(y, track_id="song_001")

# Or extract specific feature categories
tempo_features = extractor.extract_tempo(y)
spectral_features = extractor.extract_spectral_features(y)
mfcc_features = extractor.extract_mfcc(y, n_mfcc=13)
```

### `spotify_client.py` - Spotify API Integration

**Available Spotify Features:**
- `popularity`: 0-100 popularity score
- `explicit`: Content warning flag
- `tempo`: Official tempo from audio analysis
- `time_signature`: Beat structure
- `key`: Musical key (0=C, 1=C#, etc.)
- `mode`: Major (1) or Minor (0)
- `duration_ms`: Track length
- `genres`: Genre tags

**Usage:**
```python
from src import create_spotify_client

# Initialize (requires .env credentials)
client = create_spotify_client()

# Search for a track
track = client.search_track("Blinding Lights", "The Weeknd")

# Get track features
features = client.get_track_features(track["id"])

# Get audio analysis (tempo, time signature, etc.)
analysis = client.get_track_audio_analysis(track["id"])

# Batch search multiple tracks
track_list = [
    {"track_name": "Song 1", "artist_name": "Artist 1"},
    {"track_name": "Song 2", "artist_name": "Artist 2"},
]
df_spotify = client.search_tracks_batch(track_list)

# Get all tracks from a playlist
tracks = client.get_playlist_tracks("playlist_id")
```

## Audio Processing Features

### Multi-Format Support

The audio processor automatically handles:
- **MP3**: Via pydub + librosa conversion
- **WAV**: Native librosa support
- **M4A/AAC**: Via pydub conversion
- **FLAC/OGG**: Native librosa support

### Automatic Processing Pipeline

1. **Load**: Detect format, load with appropriate decoder
2. **Validate**: Check duration (>= 10s), detect silence, validate quality
3. **Normalize**: Scale audio amplitude to [-1, 1] range
4. **Crop**: Trim to 30-second preview duration (or specified length)

### Example: Complete Workflow

```python
from src import create_audio_processor, create_feature_extractor
from src.config import AUDIO_DIR, DATASETS_DIR
import pandas as pd

# Initialize
processor = create_audio_processor()
extractor = create_feature_extractor()

# Find all audio files
audio_files = list(AUDIO_DIR.glob("*.mp3")) + list(AUDIO_DIR.glob("*.m4a"))

# Process audio and extract features
all_features = []

for audio_file in audio_files:
    # Process audio
    result = processor.process_audio_file(audio_file, validate=True, normalize=True)
    
    if result:
        y, sr = result
        # Extract features
        features = extractor.extract_all_features(y, track_id=audio_file.stem)
        all_features.append(features)

# Save as CSV
df = pd.DataFrame(all_features)
df.to_csv(DATASETS_DIR / "audio_features.csv", index=False)
print(f"Extracted {len(df)} tracks, {len(df.columns)} features")
```

## Configuration

Edit `src/config.py` to customize:

```python
SAMPLE_RATE = 22050              # Audio sample rate (Hz)
PREVIEW_DURATION = 30            # Expected preview length (seconds)
N_FFT = 2048                     # FFT window size
HOP_LENGTH = 512                 # STFT hop length
MIN_AUDIO_DURATION = 10          # Minimum audio length
MAX_AUDIO_DURATION = 60          # Maximum audio length
```

## Exploratory Data Analysis with PCA

The project includes comprehensive EDA with dimensionality reduction and feature analysis in `notebooks/03_EDA_and_Baseline.ipynb`:

### Visualizations

1. **Class Distribution** - Shows the imbalance between hits and non-hits
2. **Feature Distributions** - KDE plots of each feature split by label
3. **Correlation Heatmap** - Identifies feature correlations (e.g., energy & loudness)
4. **PCA Scree Plot** - Displays variance explained by each principal component
5. **PCA Biplot** - 2D visualization with feature loadings as red arrows
   - **Red arrows** represent feature contribution vectors to PC1 and PC2
   - Arrow direction shows which principal component the feature influences
   - Arrow length indicates strength of contribution
   - Features pointing in similar directions are correlated
6. **t-SNE Visualization** - Non-linear dimensionality reduction for cluster detection
7. **PCA Component Loadings** - Bar charts showing feature weights in PC1, PC2, PC3
8. **Feature Importance (Cohen's d)** - Effect sizes quantifying feature discrimination power
9. **Logistic Regression Coefficients** - Learned feature weights from baseline model

### Why PCA?

- **Dimensionality Reduction**: 9 audio features → 2-3 principal components capturing ~90% of variance
- **Multicollinearity Detection**: Identifies correlated features (energy, loudness) that may compete in models
- **Visualization**: Project high-dimensional data into 2D/3D for intuitive understanding
- **Feature Engineering**: PC1, PC2, PC3 can replace original correlated features

### Running the EDA Notebook

```bash
jupyter notebook notebooks/03_EDA_and_Baseline.ipynb
```

Expected runtime: ~2 minutes (t-SNE is computationally expensive)

## Running the Pipeline

To run the complete pipeline example:

```bash
cd /home/william/spotify-hit-predictor
source .venv/bin/activate
python main.py
```

This will:
1. Process audio files from `data/audio_previews/`
2. Extract 60+ audio features using DSP
3. Optionally enrich with Spotify metadata (if API credentials configured)
4. Save features to `datasets/audio_features.csv`
5. Generate a summary report

## Audio Feature Descriptions

### Why These Features?

**Tempo & Rhythm** - Hit songs often have "grooving" characteristics
- Consistent, energetic beats at 100-130 BPM are common in pop hits

**Spectral Centroid** - Brightness of the track
- Bright, present vocals/instruments correlate with higher energy

**MFCC** - Perceptual audio qualities
- Captures what makes different voices/instruments sound unique

**Energy** - Overall loudness and intensity
- Radio-friendly hits are typically well-mastered and dynamic

**Chroma/Harmony** - Musical structure
- Certain chord progressions and key signatures are more "commercial"

## Next Steps

1. **Data Acquisition** (Benjamin Liu):
   - Scrape Billboard Hot 100 archives
   - Download audio previews from iTunes
   - Collect non-hit samples

2. **Feature Engineering**:
   - Combine audio DSP features with Spotify metadata
   - Feature scaling & normalization
   - Handle missing values

3. **Model Training** (Benjamin Liu):
   - Baseline Logistic Regression
   - Random Forest & AdaBoost ensembles
   - Hyperparameter tuning

4. **Frontend Dashboard** (William Zhang):
   - React UI for audio upload
   - Spectrogram visualization
   - Real-time prediction display

## References

- **librosa** Documentation: https://librosa.org
- **Spotify Web API**: https://developer.spotify.com/documentation/web-api
- **Digital Signal Processing**: https://en.wikipedia.org/wiki/Digital_signal_processing
- **Music Information Retrieval**: http://musicinformationretrieval.com/

## Team

- **William Zhang** - Audio Processing, Frontend
- **Benjamin Liu** - Data Acquisition, Machine Learning
