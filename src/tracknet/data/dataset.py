from pathlib import Path
from typing import Generator, Literal, Optional, Set
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from trackml.dataset import load_event

from .schemas import Track
from .filters import TrackFilter, FilterPipeline


class BlacklistManager:
    """
    Manages blacklists for hits and particles per event.

    Attributes:
        blacklisted_hits (dict[str, Set[int]]): 
            A mapping from event_id -> set of blacklisted hit IDs.
        blacklisted_particles (dict[str, Set[int]]): 
            A mapping from event_id -> set of blacklisted particle IDs.

    Methods:
        __init__(blacklist_dir: Optional[Path]):
            Initializes the BlacklistManager and loads blacklists if a directory is provided.

        _load_blacklists(blacklist_dir: Path):
            Loads blacklisted hits and particles from CSV files in the specified directory.

        is_valid(event_id: str, hit_ids: np.ndarray, particle_id: int) -> bool:
            Checks if the given hit IDs and particle ID are valid (not blacklisted for that event).
    """

    def __init__(self, blacklist_dir: Optional[Path]):
        self.blacklisted_hits: dict[str, Set[int]] = {}
        self.blacklisted_particles: dict[str, Set[int]] = {}

        if blacklist_dir:
            self._load_blacklists(blacklist_dir)

    def _load_blacklists(self, blacklist_dir: Path):
        # Load blacklisted hits
        for file in blacklist_dir.glob("*-blacklist_hits.csv"):
            # Example filename: event000001000-blacklist_hits.csv
            event_id = file.stem.split(
                "-blacklist_hits")[0]  # e.g. "event000001000"
            if event_id not in self.blacklisted_hits:
                self.blacklisted_hits[event_id] = set()

            df = pd.read_csv(file)
            # Assume the file has a column named 'hit_id'
            self.blacklisted_hits[event_id].update(df['hit_id'].values)

        # Load blacklisted particles
        for file in blacklist_dir.glob("*-blacklist_particles.csv"):
            # Example filename: event000001000-blacklist_particles.csv
            event_id = file.stem.split(
                "-blacklist_particles")[0]  # e.g. "event000001000"
            if event_id not in self.blacklisted_particles:
                self.blacklisted_particles[event_id] = set()

            df = pd.read_csv(file)
            # Assume the file has a column named 'particle_id'
            self.blacklisted_particles[event_id].update(
                df['particle_id'].values)

    def is_valid(self, event_id: str, hit_ids: np.ndarray, particle_id: int) -> bool:
        """Check if the given hit IDs and particle ID are valid for the specified event."""
        # Fetch blacklisted sets for the event, or empty if not present
        black_hits_for_event = self.blacklisted_hits.get(event_id, set())
        black_parts_for_event = self.blacklisted_particles.get(event_id, set())

        # Return True if particle_id and hit_ids are not in blacklisted sets
        if particle_id in black_parts_for_event:
            return False
        if any(hit_id in black_hits_for_event for hit_id in hit_ids):
            return False
        return True


class TrackMLTracksDataset(IterableDataset):
    """
    Iterable dataset for TrackML tracks.

    Args:
        data_dirs (str | Path | list[str | Path]): Directory or list of directories containing the TrackML data files.
        blacklist_dir (Optional[str | Path]): Directory containing blacklist files. Default is None.
        transforms (Optional[list]): List of transformations to apply to each track. Default is None.
        filters (Optional[list[TrackFilter]]): List of TrackFilter instances to apply to each track. Default is None.
        validation_split (float): Fraction of data to use for validation. Default is 0.1.
        split (Literal["train", "validation"]): Dataset split to use ('train' or 'validation'). Default is 'train'.

    Examples:
        >>> dataset = TrackMLTracksDataset(
        ...     data_dirs=['/path/to/data1', '/path/to/data2'],
        ...     blacklist_dir='/path/to/blacklist',
        ...     transforms=[DropRepeatedLayerHits()],
        ...     filters=[MinHitsFilter(3), PtFilter(0.8)],
        ...     validation_split=0.3,
        ...     split='train'
        ... )
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        blacklist_dir: Optional[str | Path] = None,
        transforms: Optional[list] = None,
        filters: Optional[list[TrackFilter]] = None,
        validation_split: float = 0.1,
        split: Literal["train", "validation"] = 'train'
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.transforms = transforms or []
        self.filter_pipeline = FilterPipeline(filters)

        self.blacklist = BlacklistManager(
            Path(blacklist_dir) if blacklist_dir else None)

        all_files = []
        for data_dir in self.data_dirs:
            all_files.extend(sorted(data_dir.glob("event*-hits.csv")))

        n_val = int(len(all_files) * validation_split)

        if split == 'train':
            self.event_files = all_files[n_val:]
        elif split == 'validation':
            self.event_files = all_files[:n_val]
        else:
            raise ValueError(
                f"Invalid split value: {split}. Must be 'train' or 'validation'.")

    def _process_event(self, event_file: Path) -> list[Track]:
        event_id = event_file.stem.split('-')[0]
        data_dir = event_file.parent

        hits, particles, truth = load_event(
            data_dir / event_id,
            parts=['hits', 'particles', 'truth']
        )

        track_data = truth.merge(hits, on='hit_id').merge(
            particles, on='particle_id')
        tracks = []

        for particle_id, group in track_data.groupby('particle_id'):
            hit_ids = group['hit_id'].values

            if not self.blacklist.is_valid(event_id, hit_ids, particle_id):
                continue

            track = Track(
                # from trackml: The reconstructed tracks must be
                # uniquely identified only within each event.
                event_id=event_id,
                track_id=len(tracks),
                particle_id=particle_id,
                hits_xyz=group[['x', 'y', 'z']].values,
                px=group['px'].iloc[0],
                py=group['py'].iloc[0],
                pz=group['pz'].iloc[0],
                charge=group['q'].iloc[0],
                volume_ids=group['volume_id'].values,
                layer_ids=group['layer_id'].values,
                module_ids=group['module_id'].values,
                hit_ids=hit_ids
            )

            tracks.append(track)

        return tracks

    def __iter__(self) -> Generator[Track, None, None]:
        for event_file in self.event_files:
            for track in self._process_event(event_file):
                for transform in self.transforms:
                    track = transform(track)
                if self.filter_pipeline(track):
                    yield track
