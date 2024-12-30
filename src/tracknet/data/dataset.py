from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Set
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset
from trackml.dataset import load_event
from functools import cached_property


@dataclass
class Track:
    event_id: str
    track_id: int
    particle_id: int
    hits_xyz: np.ndarray
    px: np.float32
    py: np.float32
    pz: np.float32
    charge: np.int32
    volume_ids: np.ndarray
    layer_ids: np.ndarray
    module_ids: np.ndarray
    hit_ids: np.ndarray

    @cached_property
    def hits_cylindrical(self) -> np.ndarray:
        """
        Convert hit coordinates from Cartesian (x, y, z) to cylindrical (r, phi, theta).

        Returns:
            np.ndarray: A 2D array where each row contains the cylindrical coordinates
                        (r, phi, theta) corresponding to the Cartesian coordinates (x, y, z)
                        of each hit.
                        - r: Radial distance from the origin in the xy-plane.
                        - phi: Azimuthal angle in the xy-plane.
                        - theta: Polar angle from the z-axis.
        """
        x = self.hits_xyz[:, 0]
        y = self.hits_xyz[:, 1]
        z = self.hits_xyz[:, 2]

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        theta = np.arctan2(r, z)

        return np.column_stack([r, phi, theta])

    @cached_property
    def momentum_pt(self) -> float:
        """Transverse momentum pT = sqrt(px^2 + py^2)."""
        return np.sqrt(self.px**2 + self.py**2)

    def __post_init__(self):
        """
        Post-initialization method to sort hit data by radial distance.

        This method calculates the radial distance (r) of each hit from the origin
        using the x and y coordinates. It then sorts the hits and associated data
        (volume_ids, layer_ids, module_ids, hit_ids) based on this radial distance
        in ascending order.
        """
        r = np.sqrt(self.hits_xyz[:, 0]**2 + self.hits_xyz[:, 1]**2)
        sort_idx = np.argsort(r)
        self.hits_xyz = self.hits_xyz[sort_idx]
        self.volume_ids = self.volume_ids[sort_idx]
        self.layer_ids = self.layer_ids[sort_idx]
        self.module_ids = self.module_ids[sort_idx]
        self.hit_ids = self.hit_ids[sort_idx]
        _ = self.hits_cylindrical  # Calculate and cache the cylindrical coordinates

    def __repr__(self) -> str:
        """Detailed representation showing track properties and hit distribution."""
        # Count hits per volume-layer combination
        detector_info = []
        for volume, layer, module in zip(self.volume_ids, self.layer_ids, self.module_ids):
            # e.g. "vol7:[{'l2': 3, 'l4': 1}]"
            detector_info.append(
                f"vol{int(volume)}-l{int(layer)}-m{int(module)}"
            )

        hits_info = "\n\t\t".join(detector_info)

        return (f"Track(\n"
                f"\tevent_id={self.event_id},\n"
                f"\ttrack_id={self.track_id},\n"
                f"\tparticle_id={self.particle_id},\n"
                f"\thits={len(self.hits_xyz)},\n"
                f"\tpx={self.px:.2f}, py={self.py:.2f}, pz={self.pz:.2f},\n"
                f"\tpT={self.momentum_pt:.2f}, charge={self.charge},\n"
                f"\tdetector_info=[\n\t\t{hits_info}\n\t]\n"
                f")")

    def __str__(self) -> str:
        """Returns concise string representation focusing on key attributes."""
        return f"Track {self.track_id} with {len(self.hits_xyz)} hits, pT={self.momentum_pt:.2f}"


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
    Initialize the TrackMLTracksDataset.

    Args:
        data_dirs (str | Path | list[str | Path]): Directory or list of directories containing the TrackML data files.
        blacklist_dir (Optional[str | Path]): Directory containing blacklist files. Default is None.
        transforms (Optional[list]): List of transformations to apply to each track. Default is None.
        min_hits (int): Minimum number of hits required for a track to be included. Default is 3.
        min_momentum (float): Minimum transverse momentum (pT) required for a track to be included. Default is 1.0.
        validation_split (float): Fraction of data to use for validation. Default is 0.1.
        split (str): Dataset split to use ('train' or 'validation'). Default is 'train'.

    Examples:
        >>> dataset = TrackMLTracksDataset(
        ...     data_dirs=['/path/to/data1', '/path/to/data2'],
        ...     blacklist_dir='/path/to/blacklist',
        ...     transforms=[SomeTransform()],
        ...     min_hits=3,
        ...     min_momentum=0.8,
        ...     validation_split=0.3,
        ...     split='train'
        ... )
    """

    def __init__(
        self,
        data_dirs: str | Path | list[str | Path],
        blacklist_dir: Optional[str | Path] = None,
        transforms: Optional[list] = None,
        min_hits: int = 3,
        min_momentum: float = 1.0,
        validation_split: float = 0.1,
        split: Literal["train", "validation"] = 'train'
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        self.data_dirs = [Path(data_dir) for data_dir in data_dirs]
        self.transforms = transforms or []
        if min_hits <= 2:
            raise ValueError("min_hits must be greater than 2")
        self.min_hits = min_hits
        self.min_momentum = min_momentum

        self.blacklist = BlacklistManager(
            Path(blacklist_dir) if blacklist_dir else None)

        all_files = []
        for data_dir in self.data_dirs:
            all_files.extend(sorted(data_dir.glob("event*-hits.csv")))

        n_val = int(len(all_files) * validation_split)

        if split == 'train':
            self.event_files = all_files[n_val:]
        else:  # validation
            self.event_files = all_files[:n_val]

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
            if len(group) < self.min_hits:
                continue

            momentum = np.sqrt(group['px'].iloc[0]**2 + group['py'].iloc[0]**2)

            if momentum < self.min_momentum:
                continue

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

    def __iter__(self):
        for event_file in self.event_files:
            for track in self._process_event(event_file):
                for transform in self.transforms:
                    track = transform(track)
                yield track
