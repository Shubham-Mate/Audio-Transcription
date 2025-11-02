from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torch
import pathlib
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


def train_batch(model, batch, optimizer, criterion, device):
    """
    Train the model on a single batch.
    batch: tuple of (mel_fbanks, mel_mask, tokens, token_mask)
    """
    scaler = GradScaler()

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
    with autocast():
        logits = model(
            encoder_inp=mel_fbanks, decoder_inp=decoder_input, encoder_mask=mel_mask
        )  # adapt if you also pass masks

        # Compute loss (flatten batch and sequence dims)
        batch_size, seq_len, vocab_size = logits.size()
        loss = criterion(
            logits.view(batch_size * seq_len, vocab_size),
            target_tokens.reshape(batch_size * seq_len),
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

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

for epoch in range(1, HYPERPARAMETERS["epochs"] + 1):
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
    print(f"Epoch {epoch}/{HYPERPARAMETERS['epochs']} — Avg Loss: {epoch_loss}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epoch_loss,
        },
        str(CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"),
    )
