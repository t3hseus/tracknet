import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import plotly
import io
import PIL.Image

from typing import Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data.schemas import BatchSample
from .model import StepAheadTrackNET, TrackPrediction
from .loss import TrackNetLoss
from .data.transformations import MinMaxNormalizeXYZ
from .metrics import SearchAreaMetric, HitEfficiencyMetric, HitDensityMetric
from .visualization import visualize_track_predictions


class TrackNETModule(pl.LightningModule):
    def __init__(
        self,
        input_features: int = 3,
        hidden_features: int = 32,
        output_features: int = 3,
        learning_rate: float = 1e-3,
        loss_alpha: float = 0.9,
        batch_first: bool = True,
        hit_density_stats_path: Optional[str] = None,
        hits_normalizer: Optional[MinMaxNormalizeXYZ] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = StepAheadTrackNET(
            input_features=input_features,
            hidden_features=hidden_features,
            output_features=output_features,
            batch_first=batch_first
        )

        # Loss
        self.loss_fn = TrackNetLoss(alpha=loss_alpha)

        # Metrics
        self.train_search_area_t1 = SearchAreaMetric('t1')
        self.train_search_area_t2 = SearchAreaMetric('t2')
        self.train_hit_efficiency_t1 = HitEfficiencyMetric('t1')
        self.train_hit_efficiency_t2 = HitEfficiencyMetric('t2')

        self.val_search_area_t1 = SearchAreaMetric('t1')
        self.val_search_area_t2 = SearchAreaMetric('t2')
        self.val_hit_efficiency_t1 = HitEfficiencyMetric('t1')
        self.val_hit_efficiency_t2 = HitEfficiencyMetric('t2')

        if hit_density_stats_path is not None:
            # this metric has long calculation time,
            # so we only compute it during validation
            self.val_hit_density_t1 = HitDensityMetric(
                density_stats_path=hit_density_stats_path,
                time_step='t1',
                normalizer=hits_normalizer
            )
            self.val_hit_density_t2 = HitDensityMetric(
                density_stats_path=hit_density_stats_path,
                time_step='t2',
                normalizer=hits_normalizer
            )

        # Save last output for logging
        # optional is needed to initialize with None
        self.last_batch: Optional[BatchSample] = None
        self.last_output: Optional[TrackPrediction] = None

    def forward(self, batch):
        return self.model(batch['inputs'], batch['input_lengths'])

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss_fn(output, batch['targets'], batch['target_mask'])

        # Update metrics
        self.train_search_area_t1.update(output, batch['target_mask'])
        self.train_search_area_t2.update(output, batch['target_mask'])
        self.train_hit_efficiency_t1.update(
            output, batch['targets'], batch['target_mask'])
        self.train_hit_efficiency_t2.update(
            output, batch['targets'], batch['target_mask'])

        # Log metrics
        batch_size = batch['inputs'].size(0)
        self.log('train_loss', loss, prog_bar=True, batch_size=batch_size)
        self.log_dict({
            "train_search_area_t1": self.train_search_area_t1.compute(),
            "train_hit_efficiency_t1": self.train_hit_efficiency_t1.compute(),
            "train_search_area_t2": self.train_search_area_t2.compute(),
            "train_hit_efficiency_t2": self.train_hit_efficiency_t2.compute()
        }, prog_bar=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.loss_fn(output, batch['targets'], batch['target_mask'])

        # Update metrics
        self.val_search_area_t1.update(output, batch['target_mask'])
        self.val_search_area_t2.update(output, batch['target_mask'])
        self.val_hit_efficiency_t1.update(
            output, batch['targets'], batch['target_mask'])
        self.val_hit_efficiency_t2.update(
            output, batch['targets'], batch['target_mask'])

        # Log metrics
        batch_size = batch['inputs'].size(0)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch_size)
        metrics_dict = {
            "val_search_area_t1": self.val_search_area_t1.compute(),
            "val_search_area_t2": self.val_search_area_t2.compute(),
            "val_hit_efficiency_t1": self.val_hit_efficiency_t1.compute(),
            "val_hit_efficiency_t2": self.val_hit_efficiency_t2.compute(),
        }

        if hasattr(self, 'val_hit_density_t1'):
            self.val_hit_density_t1.update(output, batch['target_mask'])
            self.val_hit_density_t2.update(output, batch['target_mask'])
            metrics_dict.update({
                "val_hit_density_t1": self.val_hit_density_t1.compute(),
                "val_hit_density_t2": self.val_hit_density_t2.compute()
            })

        self.log_dict(metrics_dict, prog_bar=True, batch_size=batch_size)

        # Save last batch for visualization
        if batch_idx == 0:
            # Convert tensors to CPU numpy arrays for visualization
            self.last_batch = {
                'inputs': batch['inputs'].detach().cpu().numpy(),
                'targets': batch['targets'].detach().cpu().numpy(),
                'input_lengths': batch['input_lengths'],
                'target_mask': batch['target_mask'].detach().cpu().numpy()
            }
            self.last_output = {
                k: v.detach().cpu().numpy() for k, v in output.items()
            }

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate  # type: ignore
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,
            patience=2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def on_save_checkpoint(self, checkpoint):
        """Save hyperparameters and model state."""
        checkpoint['hyperparameters'] = self.hparams

    def on_load_checkpoint(self, checkpoint):
        """Load hyperparameters and model state."""
        self.hparams.update(checkpoint['hyperparameters'])

    def on_validation_epoch_end(self):
        """Log track visualization."""
        if self.last_batch is not None and self.last_output is not None:
            fig = visualize_track_predictions(
                self.last_batch, self.last_output, track_idx=0)

            # Convert Plotly figure to image bytes
            img_bytes = fig.to_image(format="png")

            # Create a buffer from the bytes
            buffer = io.BytesIO(img_bytes)

            # Create PIL Image from buffer
            image = PIL.Image.open(buffer)

            # Convert PIL image to matplotlib figure
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis('off')

            # Log to tensorboard
            self.logger.experiment.add_figure(
                'Track Visualization',
                plt.gcf(),
                global_step=self.current_epoch
            )

            plt.close()  # Clean up the matplotlib figure

            if hasattr(self.logger, 'log_dir'):
                # save the last track visualization as an HTML file
                save_path = f"{self.logger.log_dir}/last_track_viz.html"
                plotly.offline.plot(fig, filename=save_path, auto_open=False)
