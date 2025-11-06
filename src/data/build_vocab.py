import sentencepiece as spm
from ..utils import logger
from ..utils.paths import TOKENIZER_OUTPUT_FILE_PATH, CORPUS_FILE_PATH


logger.info("Starting training....")
spm.SentencePieceTrainer.Train(
    input=str(CORPUS_FILE_PATH),
    model_prefix=str(TOKENIZER_OUTPUT_FILE_PATH),
    vocab_size=20000,
    model_type="bpe",
    input_sentence_size=500000,
    shuffle_input_sentence=True,
)
logger.info("Successfully completed training")
