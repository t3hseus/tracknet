import hydra
import logging
import os
from argparse import ArgumentParser
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from src.tracknet.data.collate import collate_fn

# Configure argument parser
parser = ArgumentParser()
parser.add_argument("--loglevel", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Logging level")

logging.basicConfig()
logger = logging.getLogger("train")


def setup_training(cfg: DictConfig):
    """Set up training components based on config."""
    # Instantiate datasets with all components from config
    train_dataset = instantiate(cfg.dataset, split='train')
    val_dataset = instantiate(cfg.dataset, split='validation')

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    # Instantiate model from config
    model = instantiate(cfg.model)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model
    }


def train(cfg: DictConfig, components):
    """Run training loop."""
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logging.output_dir,
        name=cfg.experiment.name
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        limit_train_batches=cfg.training.limit_train_batches,
    )

    trainer.fit(
        model=components["model"],
        train_dataloaders=components["train_loader"],
        val_dataloaders=components["val_loader"],
        ckpt_path=cfg.training.resume_from
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    pl.seed_everything(cfg.training.seed)

    components = setup_training(cfg)
    train(cfg, components)


if __name__ == "__main__":
    main()  # type: ignore
