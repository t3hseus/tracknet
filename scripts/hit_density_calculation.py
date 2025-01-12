import argparse
import numpy as np
import pandas as pd
import math
from typing import Tuple, Dict
from pathlib import Path
from tqdm import tqdm
from trackml.dataset import load_event


class RunningStats:
    """Keep track of running statistics for hit densities."""

    def __init__(self):
        self.n = 0
        self.voxel_sums = None
        self.total_hits = 0

    def update(self, voxel_counts: np.ndarray):
        """Update running statistics with new batch of data."""
        if self.voxel_sums is None:
            self.voxel_sums = np.zeros_like(voxel_counts, dtype=np.float32)

        self.voxel_sums += voxel_counts
        self.total_hits += np.sum(voxel_counts)
        self.n += 1

    def get_density_map(self) -> np.ndarray:
        """Return average hits per voxel."""
        if self.n == 0:
            return np.array([])
        return self.voxel_sums / self.n


def create_voxel_centers(grid_info: Dict) -> np.ndarray:
    """Create array of voxel center coordinates."""
    voxel_size = grid_info['voxel_size']
    origin = np.array(grid_info['origin'])
    shape = grid_info['shape']

    x = np.arange(shape[0]) * voxel_size + origin[0] + voxel_size/2
    y = np.arange(shape[1]) * voxel_size + origin[1] + voxel_size/2
    z = np.arange(shape[2]) * voxel_size + origin[2] + voxel_size/2

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    return centers


def create_voxel_grid(hits_df: pd.DataFrame, grid_info: Dict) -> np.ndarray:
    """Map hits to voxels using predefined grid parameters."""
    origin = grid_info['origin']
    voxel_size = grid_info['voxel_size']
    shape = grid_info['shape']

    voxel_counts = np.zeros(shape, dtype=np.int32)

    x_idx = ((hits_df['x'] - origin[0]) / voxel_size).astype(int)
    y_idx = ((hits_df['y'] - origin[1]) / voxel_size).astype(int)
    z_idx = ((hits_df['z'] - origin[2]) / voxel_size).astype(int)

    mask = (
        (x_idx >= 0) & (x_idx < shape[0]) &
        (y_idx >= 0) & (y_idx < shape[1]) &
        (z_idx >= 0) & (z_idx < shape[2])
    )

    x_idx, y_idx, z_idx = x_idx[mask], y_idx[mask], z_idx[mask]

    for x, y, z in zip(x_idx, y_idx, z_idx):
        voxel_counts[x, y, z] += 1

    return voxel_counts


def determine_grid_parameters(data_dir: Path, voxel_size: float, sample_size: int = 10) -> Dict:
    """Determine grid parameters by sampling a few events."""
    hit_files = sorted(data_dir.glob("event*-hits.csv"))
    if not hit_files:
        raise ValueError(f"No event files found in {data_dir}")

    x_min, y_min, z_min = float('inf'), float('inf'), float('inf')
    x_max, y_max, z_max = float('-inf'), float('-inf'), float('-inf')

    sampled_files = np.random.choice(hit_files, min(
        sample_size, len(hit_files)), replace=False)

    for hit_file in sampled_files:
        try:
            event_id = hit_file.stem.split('-')[0]
            hits, = load_event(data_dir / event_id, parts=['hits'])

            x_min = min(x_min, hits['x'].min())
            y_min = min(y_min, hits['y'].min())
            z_min = min(z_min, hits['z'].min())
            x_max = max(x_max, hits['x'].max())
            y_max = max(y_max, hits['y'].max())
            z_max = max(z_max, hits['z'].max())
        except Exception as e:
            print(f"Warning: Failed to load {hit_file}: {e}")
            continue

    if any(map(math.isinf, [x_min, y_min, z_min, x_max, y_max, z_max])):
        raise ValueError(
            "Failed to determine detector bounds from sampled files")

    padding = voxel_size * 0.1
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    z_min -= padding
    z_max += padding

    nx = int(np.ceil((x_max - x_min) / voxel_size))
    ny = int(np.ceil((y_max - y_min) / voxel_size))
    nz = int(np.ceil((z_max - z_min) / voxel_size))

    return {
        'origin': [x_min, y_min, z_min],
        'voxel_size': voxel_size,
        'shape': (nx, ny, nz)
    }


def process_events(data_dir: Path, voxel_size: float, batch_size: int = 10) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Process events in batches and compute density map."""
    hit_files = sorted(data_dir.glob("event*-hits.csv"))
    if not hit_files:
        raise ValueError(f"No event files found in {data_dir}")

    print("Determining grid parameters...")
    grid_info = determine_grid_parameters(data_dir, voxel_size)

    print(f"Grid shape: {grid_info['shape']}")
    print(f"Processing {len(hit_files)} events in batches of {batch_size}...")

    stats_calculator = RunningStats()

    for i in tqdm(range(0, len(hit_files), batch_size)):
        batch_files = hit_files[i:i + batch_size]

        for hit_file in batch_files:
            try:
                event_id = hit_file.stem.split('-')[0]
                hits, = load_event(data_dir / event_id, parts=['hits'])
                voxel_counts = create_voxel_grid(hits, grid_info)
                stats_calculator.update(voxel_counts)
            except Exception as e:
                print(f"Warning: Failed to process {hit_file}: {e}")
                continue

    if stats_calculator.n == 0:
        raise ValueError("No events were successfully processed")

    density_map = stats_calculator.get_density_map()
    voxel_centers = create_voxel_centers(grid_info)

    return grid_info, density_map.ravel(), voxel_centers


def main():
    parser = argparse.ArgumentParser(
        description='Calculate hit density statistics for TrackML data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing TrackML event files'
    )
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=100.0,
        help='Size of cubic voxels in mm'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of events to process in each batch'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/hit_density_stats.npz',
        help='Output file path for statistics (.npz)'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    output_path = Path(args.output)
    # Create all parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid_info, density_map, voxel_centers = process_events(
        data_dir,
        args.voxel_size,
        args.batch_size
    )

    print("\nSaving results...")
    np.savez(
        output_path,
        grid_info=grid_info,
        density_map=density_map,
        voxel_centers=voxel_centers
    )

    print("\nDensity map statistics:")
    print(f"Mean density: {np.mean(density_map):.2f} hits/voxel")
    print(f"Max density: {np.max(density_map):.2f} hits/voxel")
    print(f"Density percentiles:")
    for p in [50, 90, 95, 99]:
        print(
            f"{p}th percentile: {np.percentile(density_map[density_map > 0], p):.2f}")


if __name__ == "__main__":
    main()
