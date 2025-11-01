import pathlib
import torchaudio
import torchaudio.transforms as T
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob
from ..utils import logger


def preprocess_audio(
    audio_path: str,
    target_sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
):
    """
    Load an audio file, resample, compute log-mel spectrogram, normalize,
    and save to a .npy file for reuse.
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)  # shape: [channels, samples]

    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sr = target_sample_rate

    # Compute Mel spectrogram
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )(waveform)  # [1, n_mels, time]

    # Convert to log scale
    mel_db = T.AmplitudeToDB()(mel_spectrogram)  # [1, n_mels, time]

    # Normalize (per utterance)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)

    return mel_norm


def preprocess_and_save(input_path, output_path):
    mel = preprocess_audio(input_path)
    np.save(output_path, mel.squeeze(0).numpy())

    return output_path


def generate_mel_filter_banks(input_dir, output_dir, num_workers=8, format="mp3"):
    input_dir = pathlib.Path(input_dir)
    output_dir = pathlib.Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    input_paths = glob.glob(str(input_dir / f"*.{format}"), recursive=True)
    output_paths = [
        str(output_dir / os.path.basename(input_path).replace(format, "npy"))
        for input_path in input_paths
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(preprocess_and_save, input_paths[i], output_paths[i])
            for i in range(len(input_paths))
        ]

        for future in as_completed(futures):
            logger.info(f"Successfully Processed: {future.result()}")


# Generate Mel Filter Banks for Common voice dataset
COMMON_VOICE_DIR = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "raw" / "common_voice"
)
COMMON_VOICE_SUBDIRS = [f"en_train_{i}" for i in range(4)]
OUTPUT_DIR = (
    pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "preprocessed"
    / "common_voice"
)

for i in range(4):
    generate_mel_filter_banks(
        input_dir=COMMON_VOICE_DIR / COMMON_VOICE_SUBDIRS[i] / "clips",
        output_dir=OUTPUT_DIR / COMMON_VOICE_SUBDIRS[i],
        num_workers=8,
        format="mp3",
    )

# Generate Mel Filter Banks for LibriSpeech dataset
LIBRISPEECH_DIR = (
    pathlib.Path(__file__).parent.parent.parent / "data" / "raw" / "librispeech"
)
LIBRISPEECH_OUTPUT_DIR = (
    pathlib.Path(__file__).parent.parent.parent
    / "data"
    / "preprocessed"
    / "librispeech"
)

generate_mel_filter_banks(
    input_dir=LIBRISPEECH_DIR / "**" / "**",
    output_dir=LIBRISPEECH_OUTPUT_DIR,
    num_workers=8,
    format="flac",
)
