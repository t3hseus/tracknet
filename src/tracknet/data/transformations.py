import numpy as np
from .dataset import Track


class MinMaxNormalizeXYZ:
    """
    Transform that applies min-max normalization to the (x, y, z) coordinates of Track.hits_xyz.
    Normalized coordinates are in [0, 1].

    Args:
        min_xyz (np.ndarray): Array of shape (3,) specifying [min_x, min_y, min_z].
        max_xyz (np.ndarray): Array of shape (3,) specifying [max_x, max_y, max_z].
        epsilon (float): Small value to avoid division-by-zero if (max - min) is zero.

    Example:
        # Suppose you have precomputed the global min/max over your training data:
        global_min = np.array([ -300.0, -300.0, -300.0 ])
        global_max = np.array([  300.0,  300.0,  300.0 ])

        transform = MinMaxNormalizeXYZ(min_xyz=global_min, max_xyz=global_max)

        # Then in your dataset:
        transforms = [transform]
        dataset = TrackMLTracksDataset(..., transforms=transforms)
    """

    def __init__(self, min_xyz: np.ndarray, max_xyz: np.ndarray, epsilon: float = 1e-9):
        assert min_xyz.shape == (3,), "min_xyz must be shape (3,)."
        assert max_xyz.shape == (3,), "max_xyz must be shape (3,)."
        self.min_xyz = min_xyz.astype(np.float32)
        self.max_xyz = max_xyz.astype(np.float32)
        self.epsilon = epsilon

    def __call__(self, track: Track) -> Track:
        # Extract the hits (N, 3)
        hits_xyz = track.hits_xyz

        # Apply min-max normalization per coordinate
        #    x_norm = (x - min_x) / (max_x - min_x)
        #    y_norm = ...
        #    z_norm = ...
        range_xyz = (self.max_xyz - self.min_xyz) + self.epsilon  # shape (3,)

        hits_xyz_normalized = (hits_xyz - self.min_xyz) / range_xyz

        # In-place update of the track hits:
        track.hits_xyz = hits_xyz_normalized

        return track


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
