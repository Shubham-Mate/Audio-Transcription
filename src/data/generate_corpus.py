import pathlib
import glob
import polars as pl
from ..utils import logger

# Generate data file paths
common_voice_file_paths = [
    (
        pathlib.Path(__file__).parent.parent.parent
        / "data"
        / "raw"
        / "common_voice"
        / f"en_train_{i}"
        / "validated.tsv"
    )
    for i in range(4)
]

wiki_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "raw"
    / "wiki-and-book-corpus"
)
working_file_paths = glob.glob(str(wiki_path / "*.parquet"))

output_file_path = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "preprocessed" / "corpus.txt"
)

# Load files
common_voice_corpus = [
    pl.read_csv(
        common_voice_file_path,
        separator="\t",
        quote_char=None,
    )
    for common_voice_file_path in common_voice_file_paths
]

wiki_corpus = [pl.read_parquet(file_path) for file_path in working_file_paths]

logger.info("Successfully loaded files")

# Write to output corpus file

# First clear the file
open(output_file_path, "w").close()

with open(output_file_path, "a") as file_descriptor:
    for i, parquet_file in enumerate(wiki_corpus):
        file_descriptor.writelines(parquet_file["train"].to_list())
        logger.info(f"Successfully written wiki-corpus {i}")

    for single_corpus in common_voice_corpus:
        single_corpus.with_columns(
            pl.col("sentence").str.to_lowercase().alias("sentence")
        )
        logger.info("Successfully written common voice corpus")
        file_descriptor.writelines(single_corpus["sentence"].to_list())

logger.info("Successfully generated corpus")
