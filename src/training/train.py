from torch.utils.data import dataloader
import pathlib
from .dataloader import ASRDataset

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

asr_dataset = ASRDataset(
    csv_file_path=SENTENCE_CSV_PATH,
    preprocess_file_path=PREPROCESSED_PATH,
    tokenizer_file_path=TOKENIZER_MODEL_FILE_PATH,
)

point = asr_dataset[5]
print(point[0].size(), point[1].size(), point[2].size(), point[3].size())

for i in range(4):
    print(point[i])
