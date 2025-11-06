import torch
import pathlib
from ..training.config import load_config
from ..model.model import TransformerModel
from ..training.dataloader import ASRDataset
from ..training.tokenizer import Tokenizer
from ..utils import logger

PREPROCESSED_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "preprocessed"
)
SENTENCE_CSV_PATH = PREPROCESSED_PATH / "sentence.csv"
MODEL_SAVE_DIR = pathlib.Path(__file__).parent.parent.parent / "outputs" / "model"
TOKENIZER_MODEL_FILE_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "outputs"
    / "tokenizer"
    / "sentencepiece_tokenizer.model"
)

config = load_config()
DATA_SPLIT = config["training"]["split"]
HYPERPARAMETERS = config["training"]["hyperparameters"]
MODEL_PARAMETERS = config["training"]["model_parameters"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = Tokenizer(model_file_path=TOKENIZER_MODEL_FILE_PATH)

model = TransformerModel(
    vocab_size=20000,
    inp_dims=80,
    out_dims=MODEL_PARAMETERS["intermediate_dims"],
    num_heads=MODEL_PARAMETERS["num_heads"],
    num_blocks=MODEL_PARAMETERS["num_blocks"],
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_SAVE_DIR / "best_model.pt", map_location=DEVICE))

model.to(DEVICE)

logger.info("Loaded the model")
model.eval()

asr_dataset = ASRDataset(
    csv_file_path=SENTENCE_CSV_PATH,
    preprocess_file_path=PREPROCESSED_PATH,
    tokenizer_file_path=TOKENIZER_MODEL_FILE_PATH,
)
logger.info("Initialiased the dataset")
mel, mel_mask, tok, tok_mask = asr_dataset[-1]

mel = mel.unsqueeze(0).to(DEVICE)
mel_mask = mel_mask.unsqueeze(0).to(DEVICE)
tok = tok.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    out = model.generate(
        encoder_inp=mel, encoder_mask=mel_mask, start_token=1, end_token=2
    )

    out_2 = model(mel, tok, mel_mask)
    out_2 = torch.argmax(out_2, dim=-1)
logger.info("Obtained the output")
print(tok)
print(out_2[0])

print(tokenizer.detokenize(tok.cpu().numpy().tolist()))
print(tokenizer.detokenize(out_2[0].cpu().numpy().tolist()))
