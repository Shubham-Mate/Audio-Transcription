import torchaudio
import torchaudio.transforms as T
import numpy as np
from vad import EnergyVAD
import argparse
import pathlib
import sys

np.set_printoptions(threshold=sys.maxsize)


def parse_argruments():
    parser = argparse.ArgumentParser(
        prog="Speech Transcriptor",
        description="Given an audio file, generates transcription",
    )
    parser.add_argument(
        "--file",
        type=pathlib.Path,  # Converts string input into a Path object automatically
        required=True,
        help="Path to the input file",
    )

    args = parser.parse_args()
    input_file_path = args.file
    return input_file_path


input_file_path = parse_argruments()

TARGET_SAMPLE_RATE = 16000
FRAME_SIZE_IN_SECS = 1

# Setup VAD
vad = EnergyVAD(
    sample_rate=TARGET_SAMPLE_RATE,
    frame_length=FRAME_SIZE_IN_SECS * 1000,
    frame_shift=FRAME_SIZE_IN_SECS * 1000,
    energy_threshold=0.05,
    pre_emphasis=0.95,
)

waveform, sr = torchaudio.load(input_file_path)  # shape: [channels, samples]
if sr != TARGET_SAMPLE_RATE:
    resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
    waveform = resampler(waveform)
    sr = TARGET_SAMPLE_RATE

print(sr)
voice_activity = vad(waveform.numpy())

print(voice_activity.shape)
print(voice_activity)
print(np.sum(voice_activity))
