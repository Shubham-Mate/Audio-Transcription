import pathlib
import sentencepiece as spm
from ..utils import logger

OUTPUT_FILE_NAME = "sentencepiece_tokenizer"
OUTPUT_FILE_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "outputs"
    / "tokenizer"
    / OUTPUT_FILE_NAME
)

CORPUS_FILE_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "preprocessed" / "corpus.txt"
)

logger.info("Starting training....")
spm.SentencePieceTrainer.Train(
    input=str(CORPUS_FILE_PATH),
    model_prefix=str(OUTPUT_FILE_PATH),
    vocab_size=20000,
    model_type="bpe",
    input_sentence_size=500000,
    shuffle_input_sentence=True,
)
logger.info("Successfully completed training")
