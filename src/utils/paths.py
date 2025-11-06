import pathlib

SRC_DIR = pathlib.Path(__file__).parent.parent
GLOBAL_DIRECTORY = SRC_DIR.parent
OUTPUT_DIR = GLOBAL_DIRECTORY / "outputs"
DATA_DIR = GLOBAL_DIRECTORY / "data"

TOKENIZER_OUTPUT_FILE_NAME = "sentencepiece_tokenizer"
TOKENIZER_OUTPUT_FILE_PATH = OUTPUT_DIR / "tokenizer" / TOKENIZER_OUTPUT_FILE_NAME

PREPROCESSED_PATH = DATA_DIR / "preprocessed"
CORPUS_FILE_PATH = PREPROCESSED_PATH / "corpus.txt"

CONFIG_PATH = SRC_DIR / "config.yaml"

COMMON_VOICE_DIR = DATA_DIR / "raw" / "common_voice"

MODEL_SAVE_DIR = OUTPUT_DIR / "model"
CHECKPOINT_DIR = OUTPUT_DIR / "model_checkpoints"

SENTENCE_OUTPUT_FILE = "sentence.csv"
SENTENCE_CSV_PATH = PREPROCESSED_PATH / SENTENCE_OUTPUT_FILE

TOKENIZER_MODEL_FILE_PATH = (
    OUTPUT_DIR / "tokenizer" / f"{TOKENIZER_OUTPUT_FILE_NAME}.model"
)

LIBRISPEECH_DIR = DATA_DIR / "raw" / "librispeech"
COMMON_VOICE_DIR = DATA_DIR / "raw" / "common_voice"
