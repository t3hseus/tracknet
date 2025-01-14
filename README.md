# TrackNET: Intelligent Particle Track Building

TrackNET is a deep learning-based approach for particle track reconstruction, implemented using PyTorch Lightning. This repository contains the implementation of the StepAhead TrackNET model, recurrent neural network performing sequential track building in high-energy physics experiments.

## Model Overview

StepAhead TrackNET operates as a trainable Kalman filter, using a GRU-based architecture to predict spherical search regions for the next hits in a particle track. For each input hit sequence, the model generates two predictions:

- t1: Immediate next hit location and search radius
- t2: Location and search radius for the hit after next

This dual prediction strategy helps handle missing detector hits and improves track reconstruction robustness.

## Key Features

- Sequential track building using RNN architecture
- Dual-step prediction (t1 and t2) for robust reconstruction
- Configurable search region size through learned radius prediction
- Built-in visualization tools for track predictions
- Comprehensive metrics tracking (search area, hit efficiency)
- Integration with the TrackML particle tracking challenge

## Repository Setup

1. Clone the repository:
```bash
git clone https://github.com/t3hseus/tracknet.git
cd tracknet
```

2. Get the TrackML utility library:
```bash
cd src/
git clone https://github.com/LAL/trackml-library.git
cd ..
```

3. Create and activate the conda environment:
```bash
conda env update -f environment.yml
conda activate tracknet
```

4. Download TrackML data:
   - Visit https://www.kaggle.com/c/trackml-particle-identification/data
   - Download training and testing datasets
   - Unpack archives to `data/trackml/`:
     - `data/trackml/blacklist_training/`
     - `data/trackml/train_100_events/`
     - etc.

5. Calculate hit density statistics (required for HitDensityMetric):
```bash
python scripts/hit_density_calculation.py \
    --data-dir data/trackml/train_100_events \
    --voxel-size 100.0 \
    --batch-size 10 \
    --output outputs/hit_density_stats.npz
```

6. Configure settings (optional):
   - Edit `configs/user_settings/user_settings.yaml` for custom paths
   - Adjust model parameters in `configs/model/step_ahead.yaml`
   - Modify training parameters in `configs/train.yaml`

## Training

1. Start training:
```bash
python train.py
```

For specific GPU selection:
```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py
```

2. Monitor training:
```bash
tensorboard --logdir outputs/
```

## Results and Outputs

Training results are stored in `outputs/` with the following structure:
```
outputs/YYYY-MM-DD/HH-MM-SS/
├── logs
│   └── step_ahead_tracknet_trackml
│       └── version_0
│           ├── checkpoints/             # Top-3 model checkpoints
│           ├── events.out.tfevents.*    # TensorBoard logs
│           ├── hparams.yaml            # Hyperparameters
│           └── last_track_viz.html     # Track visualization
└── train.log                          # Training log
```

Key outputs:
- Model checkpoints: `logs/*/checkpoints/`
- Training metrics: TensorBoard logs
- Visualization of the first track in the last validation batch from the last completed epoch: `last_track_viz.html`
- Configuration: `hparams.yaml`

## References

1. Rusov, D., Goncharov, P., Zhemchugov, A. *et al.* Deep Tracking for the SPD Experiment. *Phys. Part. Nuclei Lett.* **20**, 1180–1182 (2023). https://doi.org/10.1134/S1547477123050655

2. Bakina, O., et al. "Deep Learning for Track Recognition in Pixel and Strip-Based Particle Detectors." *Journal of Instrumentation*, vol. 17, no. 12, IOP Publishing, Dec. 2022, p. P12023, doi:10.1088/1748-0221/17/12/P12023.

3. Rusov, D. *et al.* (2023). Recurrent and Graph Neural Networks for Particle Tracking at the BM@N Experiment. In: *Advances in Neural Computation, Machine Learning, and Cognitive Research VI. NEUROINFORMATICS 2022*. Studies in Computational Intelligence, vol 1064. Springer, Cham. https://doi.org/10.1007/978-3-031-19032-2_32

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tracknet2024,
  author = {Goncharov, Pavel and Rusov, Daniil},
  title = {TrackNET: Intelligent Particle Track Building},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/t3hseus/tracknet}},
}
```