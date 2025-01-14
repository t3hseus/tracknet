import numpy as np
import torch
from typing import Union
from .dataset import Track


class MinMaxNormalizeXYZ:
    """
    Transform that applies min-max normalization to XYZ coordinates.
    Supports both single Track objects and arrays of coordinates.

    Args:
        min_xyz (tuple): Tuple specifying [min_x, min_y, min_z]
        max_xyz (tuple): Tuple specifying [max_x, max_y, max_z]
        epsilon (float): Small value to avoid division-by-zero

        Example:
        # Suppose you have precomputed the global min/max over your training data:
        global_min = (-300.0, -300.0, -300.0)
        global_max = (300.0,  300.0,  300.0)

        transform = MinMaxNormalizeXYZ(min_xyz=global_min, max_xyz=global_max)

        # Then in your dataset:
        transforms = [transform]
        dataset = TrackMLTracksDataset(..., transforms=transforms)
    """

    def __init__(
        self,
        min_xyz: tuple[float, float, float],
        max_xyz: tuple[float, float, float],
        epsilon: float = 1e-9
    ):
        self.min_xyz = np.asarray(min_xyz, dtype=np.float32)
        self.max_xyz = np.asarray(max_xyz, dtype=np.float32)
        self.epsilon = epsilon
        self.range_xyz = (self.max_xyz - self.min_xyz) + self.epsilon

    def __call__(self, track: Track) -> Track:
        """Apply normalization to a Track object."""
        track.hits_xyz = self.normalize_xyz(track.hits_xyz)
        return track

    @classmethod
    def identity(cls) -> 'MinMaxNormalizeXYZ':
        """Create an identity transform that performs no normalization."""
        return cls(
            min_xyz=(0.0, 0.0, 0.0),
            max_xyz=(1.0, 1.0, 1.0)
        )

    def normalize_xyz(
        self,
        xyz: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize XYZ coordinates to [0,1] range.

        Args:
            xyz: Array of shape (..., 3) containing XYZ coordinates
                 Supports both numpy arrays and torch tensors

        Returns:
            Normalized coordinates in same format as input
        """
        if isinstance(xyz, torch.Tensor):
            min_xyz = torch.from_numpy(self.min_xyz).to(xyz.device)
            range_xyz = torch.from_numpy(self.range_xyz).to(xyz.device)
            return (xyz - min_xyz) / range_xyz
        else:
            return (xyz - self.min_xyz) / self.range_xyz

    def denormalize_xyz(
        self,
        normalized_xyz: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert normalized coordinates back to detector space.

        Args:
            normalized_xyz: Array of shape (..., 3) containing normalized coordinates
                          Supports both numpy arrays and torch tensors

        Returns:
            Denormalized coordinates in detector space, same format as input
        """
        if isinstance(normalized_xyz, torch.Tensor):
            min_xyz = torch.from_numpy(self.min_xyz).to(normalized_xyz.device)
            range_xyz = torch.from_numpy(
                self.range_xyz).to(normalized_xyz.device)
            return normalized_xyz * range_xyz + min_xyz
        else:
            return normalized_xyz * self.range_xyz + self.min_xyz


class DropRepeatedLayerHits:
    """
    A transform that drops repeated hits in the same (volume_id, layer_id),
    keeping only the first (smallest-r) hit in that volume-layer pair.

    Assumptions:
      - Track.hits_xyz, volume_ids, layer_ids, module_ids, hit_ids
        are sorted by ascending r (thanks to Track.__post_init__).
      - We only check (volume_id, layer_id). If volume differs, hits are kept
        even if the layer_id is the same.

    Example usage:
        dataset = TrackMLTracksDataset(
            data_dirs=[...],
            transforms=[DropRepeatedLayerHits(), ...]
        )
    """

    def __call__(self, track):
        visited_volume_layer = set()
        keep_mask = []

        # We iterate over hits in ascending r (already guaranteed by track.__post_init__)
        for volume, layer in zip(track.volume_ids, track.layer_ids):
            volume_layer = (volume, layer)
            if volume_layer not in visited_volume_layer:
                # First time we see this volume-layer pair
                visited_volume_layer.add(volume_layer)
                keep_mask.append(True)
            else:
                # Already encountered this volume-layer pair
                keep_mask.append(False)

        keep_mask = np.array(keep_mask, dtype=bool)

        # Apply mask to all relevant arrays
        track.hits_xyz = track.hits_xyz[keep_mask]
        track.volume_ids = track.volume_ids[keep_mask]
        track.layer_ids = track.layer_ids[keep_mask]
        track.module_ids = track.module_ids[keep_mask]
        track.hit_ids = track.hit_ids[keep_mask]

        del track.hits_cylindrical  # Invalidate the cached property
        _ = track.hits_cylindrical  # Recompute the property

        return track
