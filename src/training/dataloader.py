from torch.utils.data import Dataset
import torch
import polars
import numpy as np
import pathlib
from .tokenizer import Tokenizer
from .config import load_config


class ASRDataset(Dataset):
    def __init__(
        self, csv_file_path, preprocess_file_path, tokenizer_file_path
    ) -> None:
        super().__init__()

        self.df = polars.read_csv(csv_file_path, separator="\t", quote_char=None)
        self.preprocess_file_path = pathlib.Path(preprocess_file_path)
        self.tokenizer = Tokenizer(tokenizer_file_path)
        self.config = load_config()["data"]

    def pad_mel_filter_bank(self, mel, max_len: int, pad_value: float = 0.0):
        T, dim = mel.size()
        if T >= max_len:
            padded = mel[:max_len, :]
            mask = torch.zeros(max_len, dtype=torch.bool, device=mel.device)
        else:
            pad_amount = max_len - T
            pad_tensor = mel.new_full((pad_amount, dim), pad_value)
            padded = torch.cat([mel, pad_tensor], dim=0)
            mask = torch.cat(
                [
                    torch.zeros(T, dtype=torch.bool, device=mel.device),
                    torch.ones(pad_amount, dtype=torch.bool, device=mel.device),
                ],
                dim=0,
            )
        return padded, mask

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        path = self.preprocess_file_path / self.df.item(idx, "path")
        sentence = self.df.item(idx, "sentence")
        mel_filter_banks = torch.Tensor(np.load(str(path)).T)

        tokenized, token_mask = self.tokenizer.tokenize(
            sentence, max_len=self.config["text"]["max_length"]
        )
        tokenized, token_mask = torch.Tensor(tokenized), torch.Tensor(token_mask)
        padded_mel_filter_bank, mel_mask = self.pad_mel_filter_bank(
            mel_filter_banks, self.config["audio"]["max_length"]
        )

        return (
            padded_mel_filter_bank,
            mel_mask,
            tokenized.long(),
            token_mask.long(),
        )
