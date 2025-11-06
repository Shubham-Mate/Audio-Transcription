from ..training.config import load_config
from ..model.model import TransformerModel
from ..utils import logger
import torch
import pathlib
import glob
import re


def extract_epoch(path):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt", str(path))
    return int(match.group(1)) if match else -1


config = load_config()
DATA_SPLIT = config["training"]["split"]
HYPERPARAMETERS = config["training"]["hyperparameters"]
MODEL_PARAMETERS = config["training"]["model_parameters"]

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Loaded device: {DEVICE}")

MODEL_SAVE_DIR = pathlib.Path(__file__).parent.parent.parent / "outputs" / "model"
CHECKPOINT_DIR = (
    pathlib.Path(__file__).parent.parent.parent / "outputs" / "model_checkpoints"
)
PREPROCESSED_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "preprocessed"
)

checkpoints = sorted(glob.glob(str(CHECKPOINT_DIR / "*.pt")), key=extract_epoch)
for checkpoint in checkpoints:
    checkpoint_loaded = torch.load(checkpoint, map_location=DEVICE)
    epoch = checkpoint_loaded["epoch"]
    train_loss = checkpoint_loaded["train_loss"]
    val_loss = None
    best_val_loss = None
    if "best_val_loss" in checkpoint_loaded:
        best_val_loss = checkpoint_loaded["best_val_loss"]
    if "val_loss" in checkpoint_loaded:
        val_loss = checkpoint_loaded["val_loss"]

    print(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")
