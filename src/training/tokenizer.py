import sentencepiece
import pathlib
from .config import load_config

config = load_config()


class Tokenizer:
    def __init__(self, model_file_path, pad_token: int = 0) -> None:
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(model_file_path), add_eos=True, add_bos=True
        )
        self.pad_token = pad_token

    def add_pad_tokens(self, tokenized, max_len):
        length = len(tokenized)

        if len(tokenized) < max_len:
            pad_len = max_len - length
            tokenized = tokenized + [self.pad_token] * (pad_len)
            mask = [0] * length + [1] * pad_len
        else:
            tokenized = tokenized[:max_len]
            tokenized[-1] = 0
            mask = [0] * max_len

        return (tokenized, mask)

    def tokenize(self, text: str, max_len: int = config["data"]["text"]["max_length"]):
        tokenized = self.tokenizer.encode_as_ids(text)
        padded_tokenized = self.add_pad_tokens(tokenized, max_len=max_len)

        return padded_tokenized
