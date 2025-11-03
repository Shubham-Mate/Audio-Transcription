from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import torch
import pathlib
import glob
import re
from tqdm import tqdm
from .dataloader import ASRDataset
from .config import load_config
from ..model.model import TransformerModel
from ..utils import logger

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
SENTENCE_CSV_PATH = PREPROCESSED_PATH / "sentence.csv"
TOKENIZER_MODEL_FILE_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "outputs"
    / "tokenizer"
    / "sentencepiece_tokenizer.model"
)

scaler = GradScaler()


def extract_epoch(path):
    match = re.search(r"checkpoint_epoch_(\d+)\.pt", str(path))
    return int(match.group(1)) if match else -1


def train_batch(model, batch, optimizer, criterion, device, train_mode=True):
    """
    Train the model on a single batch.
    batch: tuple of (mel_fbanks, mel_mask, tokens, token_mask)
    """

    mel_fbanks, mel_mask, tokens, token_mask = batch
    mel_fbanks = mel_fbanks.to(device)
    mel_mask = mel_mask.to(device)
    tokens = tokens.to(device)
    token_mask = token_mask.to(device)

    optimizer.zero_grad()

    # Assuming decoder input is tokens without last token, target is tokens without first
    decoder_input = tokens[:, :-1]
    target_tokens = tokens[:, 1:]

    # Forward pass
    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model(
            encoder_inp=mel_fbanks, decoder_inp=decoder_input, encoder_mask=mel_mask
        )  # adapt if you also pass masks

        # Compute loss (flatten batch and sequence dims)
        batch_size, seq_len, vocab_size = logits.size()
        loss = criterion(
            logits.view(batch_size * seq_len, vocab_size),
            target_tokens.reshape(batch_size * seq_len),
        )

    if train_mode:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return loss.item()


asr_dataset = ASRDataset(
    csv_file_path=SENTENCE_CSV_PATH,
    preprocess_file_path=PREPROCESSED_PATH,
    tokenizer_file_path=TOKENIZER_MODEL_FILE_PATH,
)

train_dataset, val_dataset, test_dataset = random_split(
    asr_dataset,
    [DATA_SPLIT["train"], DATA_SPLIT["val"], DATA_SPLIT["test"]],
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=HYPERPARAMETERS["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=HYPERPARAMETERS["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=HYPERPARAMETERS["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

# Setup model, loss function, optimizer for training
model = TransformerModel(
    vocab_size=20000,
    inp_dims=80,
    out_dims=MODEL_PARAMETERS["intermediate_dims"],
    num_heads=MODEL_PARAMETERS["num_heads"],
    num_blocks=MODEL_PARAMETERS["num_blocks"],
).to(DEVICE)

criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])

epoch = 0
best_val_loss = float("inf")
if len(glob.glob(str(CHECKPOINT_DIR / "*.pt"))) == 0:
    epoch = 0
else:
    last_checkpoint_path = max(
        glob.glob(str(CHECKPOINT_DIR / "*.pt")), key=extract_epoch
    )
    checkpoint = torch.load(last_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1
    if "best_val_loss" in checkpoint:
        best_val_loss = checkpoint["best_val_loss"]

while epoch < HYPERPARAMETERS["epochs"] + 1:
    # --------------------------------- Training ----------------------------------------- #

    model.train()
    running_loss = 0.0
    train_bar = tqdm(
        train_dataloader, desc=f"Epoch {epoch}/{HYPERPARAMETERS['epochs']}", leave=False
    )

    for batch_idx, batch in enumerate(train_bar, start=1):
        loss = train_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
        )

        running_loss += loss
        if batch_idx % HYPERPARAMETERS.get("log_interval", 10) == 0:
            avg_loss_sofar = running_loss / batch_idx
            train_bar.set_postfix(loss=f"{avg_loss_sofar}")

    epoch_loss = running_loss / len(train_dataloader)
    # print(f"Epoch {epoch}/{HYPERPARAMETERS['epochs']} — Avg Loss: {epoch_loss}")

    # --------------------------------- Validation ----------------------------------------- #

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch}/{HYPERPARAMETERS['epochs']} (Validation)",
            leave=False,
        )
        for batch in val_bar:
            val_loss = train_batch(  # reuse function but w/o backward pass
                model=model,
                batch=batch,
                optimizer=optimizer,  # skip optimizer step
                criterion=criterion,
                device=DEVICE,
                train_mode=False,
            )
            val_running_loss += val_loss

    epoch_val_loss = val_running_loss / len(val_dataloader)

    print(
        f"Epoch {epoch}/{HYPERPARAMETERS['epochs']} — "
        f"Train Loss: {epoch_loss} | Val Loss: {epoch_val_loss}"
    )

    # -------------------------------Saving best model--------------------------------------- #

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_DIR / "best_model.pt")

    # ---------------------------------- Checkpointing -------------------------------------- #

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_loss,
            "val_loss": epoch_val_loss,
            "best_val_loss": best_val_loss,
        },
        str(CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"),
    )

    # --------------------------------------------------------------------------------------- #

    epoch += 1
