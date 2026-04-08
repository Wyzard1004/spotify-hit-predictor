"""
Spotify API Integration
Handles authentication and data fetching from Spotify Web API.
Fetches metadata, genres, audio features (where available), and track information.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import time

from config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET,
    SPOTIFY_API_RATE_LIMIT,
    SPOTIFY_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class SpotifyClient:
    """
    Client for Spotify Web API.
    Fetches track metadata, genres, available audio features, and playlist data.
    """

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Initialize Spotify client with authentication.

        Args:
            client_id: Spotify API client ID (uses env var if not provided)
            client_secret: Spotify API client secret (uses env var if not provided)

        Raises:
            ValueError: If authentication credentials are not provided
        """
        client_id = client_id or SPOTIFY_CLIENT_ID
        client_secret = client_secret or SPOTIFY_CLIENT_SECRET

        if not client_id or not client_secret:
            raise ValueError(
                "Spotify credentials not found. Set SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET in .env file"
            )

        try:
            auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("Successfully authenticated with Spotify API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Spotify: {str(e)}")
            raise

        self.logger = logging.getLogger(__name__)
        self.rate_limit_delay = 1.0 / SPOTIFY_API_RATE_LIMIT

    def _rate_limit(self):
        """Apply rate limiting to API requests."""
        time.sleep(self.rate_limit_delay)

    def search_track(self, track_name: str, artist_name: str = "") -> Optional[Dict]:
        """
        Search for a track on Spotify.

        Args:
            track_name: Name of the track
            artist_name: Name of the artist (optional)

        Returns:
            Track information dictionary or None if not found
        """
        try:
            query = f"track:{track_name}"
            if artist_name:
                query += f" artist:{artist_name}"

            self._rate_limit()
            results = self.sp.search(q=query, type="track", limit=1)

            if results["tracks"]["items"]:
                return results["tracks"]["items"][0]
            return None

        except Exception as e:
            self.logger.error(f"Error searching for track '{track_name}': {str(e)}")
            return None

    def get_track_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a track.

        NOTE: Spotify deprecated the detailed audio features endpoint, but we can still
        get some useful metadata through other endpoints.

        Args:
            track_id: Spotify track ID

        Returns:
            Dictionary with available track features
        """
        try:
            self._rate_limit()
            track = self.sp.track(track_id)

            # Extract available features from track metadata
            features = {
                "track_id": track_id,
                "name": track["name"],
                "artist": track["artists"][0]["name"] if track["artists"] else "Unknown",
                "popularity": track.get("popularity", 0),  # 0-100 popularity score
                "explicit": track.get("explicit", False),
                "duration_ms": track.get("duration_ms", 0),
                "release_date": track.get("album", {}).get("release_date", ""),
                "album": track.get("album", {}).get("name", ""),
            }

            return features

        except Exception as e:
            self.logger.error(f"Error fetching features for track {track_id}: {str(e)}")
            return None

    def get_track_audio_analysis(self, track_id: str) -> Optional[Dict]:
        """
        Get audio analysis for a track (still available endpoint).

        This provides detailed tempo, time signature, and beat information.

        Args:
            track_id: Spotify track ID

        Returns:
            Audio analysis dictionary
        """
        try:
            self._rate_limit()
            analysis = self.sp.audio_analysis(track_id)

            if analysis:
                return {
                    "track_id": track_id,
                    "tempo": analysis.get("track", {}).get("tempo", 0),
                    "time_signature": analysis.get("track", {}).get("time_signature", 0),
                    "key": analysis.get("track", {}).get("key", 0),
                    "mode": analysis.get("track", {}).get("mode", 0),  # 0=minor, 1=major
                    "beats": len(analysis.get("beats", [])),
                    "bars": len(analysis.get("bars", [])),
                    "sections": len(analysis.get("sections", [])),
                }
            return None

        except Exception as e:
            self.logger.error(f"Error fetching audio analysis for track {track_id}: {str(e)}")
            return None

    def get_track_genres(self, artist_id: str) -> List[str]:
        """
        Get genres for an artist.

        Args:
            artist_id: Spotify artist ID

        Returns:
            List of genre tags
        """
        try:
            self._rate_limit()
            artist = self.sp.artist(artist_id)
            return artist.get("genres", [])

        except Exception as e:
            self.logger.error(f"Error fetching genres for artist {artist_id}: {str(e)}")
            return []

    def search_tracks_batch(self, track_list: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Search for multiple tracks and fetch their metadata/features.

        Args:
            track_list: List of dicts with 'track_name' and 'artist_name' keys

        Returns:
            DataFrame with track metadata and available features
        """
        results = []

        iterator = tqdm(track_list, desc="Fetching Spotify metadata")

        for track_info in iterator:
            track_name = track_info.get("track_name", "")
            artist_name = track_info.get("artist_name", "")

            # Search for track
            spotify_track = self.search_track(track_name, artist_name)

            if spotify_track:
                track_id = spotify_track["id"]

                # Get track features
                features = self.get_track_features(track_id)

                # Get audio analysis
                audio_analysis = self.get_track_audio_analysis(track_id)

                # Get artist genres
                if spotify_track["artists"]:
                    artist_id = spotify_track["artists"][0]["id"]
                    genres = self.get_track_genres(artist_id)
                    if features:
                        features["genres"] = ", ".join(genres[:3])  # Top 3 genres

                # Combine all data
                if features and audio_analysis:
                    combined = {**features, **audio_analysis}
                    results.append(combined)
                elif features:
                    results.append(features)

        df = pd.DataFrame(results)
        self.logger.info(f"Successfully fetched Spotify data for {len(df)} tracks")
        return df

    def get_playlist_tracks(self, playlist_id: str, limit: int = None) -> List[Dict]:
        """
        Get all tracks from a Spotify playlist.

        Args:
            playlist_id: Spotify playlist ID
            limit: Maximum number of tracks to fetch (None = all)

        Returns:
            List of track information dicts
        """
        try:
            all_tracks = []
            offset = 0

            while True:
                self._rate_limit()
                results = self.sp.playlist_tracks(playlist_id, offset=offset, limit=50)

                if not results["items"]:
                    break

                for item in results["items"]:
                    if item["track"]:
                        all_tracks.append(item["track"])

                    if limit and len(all_tracks) >= limit:
                        return all_tracks[:limit]

                offset += 50

            self.logger.info(f"Fetched {len(all_tracks)} tracks from playlist {playlist_id}")
            return all_tracks

        except Exception as e:
            self.logger.error(f"Error fetching playlist {playlist_id}: {str(e)}")
            return []

    def search_playlists(
        self, query: str, limit: int = 10, offset: int = 0
    ) -> List[Dict]:
        """
        Search for playlists.

        Useful for finding playlists of non-hit songs or obscure tracks.

        Args:
            query: Search query
            limit: Number of results to return
            offset: Offset for pagination

        Returns:
            List of playlist information dicts
        """
        try:
            self._rate_limit()
            results = self.sp.search(q=query, type="playlist", limit=limit, offset=offset)
            playlists = results.get("playlists", {}).get("items", [])
            return playlists

        except Exception as e:
            self.logger.error(f"Error searching playlists: {str(e)}")
            return []

    def get_available_features(self) -> Dict[str, str]:
        """
        Return a dictionary of available Spotify features and their descriptions.

        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            "popularity": "Track popularity (0-100). A value of 100 is the most popular.",
            "explicit": "Whether the track contains explicit content.",
            "duration_ms": "Track duration in milliseconds.",
            "tempo": "Overall estimated tempo of a track in beats per minute (BPM).",
            "time_signature": "An estimated time signature (e.g., 4/4).",
            "key": "The pitch class notation. 0=C, 1=C♯/D♭, etc.",
            "mode": "Major (1) or minor (0).",
            "beats": "Number of beats detected in the track.",
            "bars": "Number of bars detected in the track.",
            "sections": "Number of sections detected in the track.",
        }


def create_spotify_client(
    client_id: Optional[str] = None, client_secret: Optional[str] = None
) -> SpotifyClient:
    """
    Factory function to create SpotifyClient instance.

    Args:
        client_id: Spotify API client ID
        client_secret: Spotify API client secret

    Returns:
        SpotifyClient instance
    """
    return SpotifyClient(client_id=client_id, client_secret=client_secret)
