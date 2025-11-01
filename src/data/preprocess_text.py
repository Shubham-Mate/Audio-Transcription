from ..utils import logger
import polars
import glob
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_csv_from_txt_file(txt_file_path):
    reg_expr = r"^(\d+(?:-\d+)+)\s+(.*)$"
    with open(txt_file_path, "r") as file_descriptor:
        file_content = file_descriptor.read().splitlines()
        file_paths = [
            f"librispeech/{re.match(reg_expr, line).group(1)}.npy"
            for line in file_content
        ]
        sentences = [
            f"{re.match(reg_expr, line).group(2)}".lower() for line in file_content
        ]

    new_df = polars.DataFrame({"path": file_paths, "sentence": sentences})

    return (txt_file_path, new_df)


def preprocess_tsv(tsv_file_path):
    tsv_file_path = pathlib.Path(tsv_file_path)
    df = polars.read_csv(tsv_file_path, separator="\t", quote_char=None)
    df = df.with_columns(
        [
            (
                "common_voice"
                + "/"
                + polars.lit(tsv_file_path.parent.name)
                + "/"
                + polars.col("path").str.replace(".mp3$", ".npy")
            ).alias("path"),
            polars.col("sentence").str.to_lowercase().alias("sentence"),
        ]
    ).select(["path", "sentence"])
    return (tsv_file_path, df)


NUM_WORKERS = 8

# Generate aggregated csv for common voice
COMMON_VOICE_DIR = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "raw" / "common_voice"
)

validated_tsv_paths = glob.glob(
    str(COMMON_VOICE_DIR / "**" / "validated.tsv"), recursive=True
)

common_voice_dfs = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [
        executor.submit(preprocess_tsv, validated_tsv_paths[i])
        for i in range(len(validated_tsv_paths))
    ]

    for future in as_completed(futures):
        logger.info(f"Successfully Processed: {future.result()[0]}")
        common_voice_dfs.append(future.result()[1])


# Generate aggregated csv for librispeech
LIBRISPEECH_DIR = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "raw" / "librispeech"
)

librispeech_txt_paths = glob.glob(
    str(LIBRISPEECH_DIR / "**" / "**" / "*.txt"), recursive=True
)
librispeech_dfs = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [
        executor.submit(generate_csv_from_txt_file, librispeech_txt_paths[i])
        for i in range(len(librispeech_txt_paths))
    ]

    for future in as_completed(futures):
        logger.info(f"Successfully Processed: {future.result()[0]}")
        librispeech_dfs.append(future.result()[1])

aggregated_dfs = common_voice_dfs + librispeech_dfs
aggregated_dfs = polars.concat(aggregated_dfs)

OUTPUT_DIR = pathlib.Path(__file__).parent.parent.parent / "data" / "preprocessed"
OUTPUT_FILE = "sentence.csv"

aggregated_dfs.write_csv(OUTPUT_DIR / OUTPUT_FILE, separator="\t")
