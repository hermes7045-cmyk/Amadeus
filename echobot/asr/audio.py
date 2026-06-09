from __future__ import annotations

import io
import sys
import wave
from array import array


def read_wav_bytes(audio_bytes: bytes, target_sample_rate: int) -> list[float]:
    if not audio_bytes:
        raise ValueError("ASR audio body must not be empty")

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        if frame_count <= 0:
            return []
        raw_frames = wav_file.readframes(frame_count)

    samples = decode_pcm_frames(
        raw_frames,
        channel_count=channel_count,
        sample_width=sample_width,
    )
    if sample_rate != target_sample_rate:
        samples = resample_samples(
            samples,
            input_sample_rate=sample_rate,
            output_sample_rate=target_sample_rate,
        )
    return samples


def write_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    if sample_rate <= 0:
        raise ValueError("ASR sample_rate must be positive")

    pcm_samples = array("h")
    for sample in samples:
        clamped = max(-1.0, min(1.0, float(sample)))
        if clamped >= 1.0:
            pcm_samples.append(32767)
            continue
        pcm_samples.append(int(clamped * 32768.0))

    if sys.byteorder != "little":
        pcm_samples.byteswap()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_samples.tobytes())

    return buffer.getvalue()


def decode_pcm_frames(
    raw_frames: bytes,
    *,
    channel_count: int,
    sample_width: int,
) -> list[float]:
    if channel_count <= 0:
        raise ValueError("WAV file must have at least one channel")

    if sample_width == 1:
        mono_samples = [(sample - 128) / 128.0 for sample in raw_frames]
    elif sample_width == 2:
        pcm_samples = array("h")
        pcm_samples.frombytes(raw_frames)
        if sys.byteorder != "little":
            pcm_samples.byteswap()
        mono_samples = [sample / 32768.0 for sample in pcm_samples]
    elif sample_width == 4:
        pcm_samples = array("i")
        pcm_samples.frombytes(raw_frames)
        if sys.byteorder != "little":
            pcm_samples.byteswap()
        mono_samples = [sample / 2147483648.0 for sample in pcm_samples]
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channel_count == 1:
        return mono_samples

    downmixed: list[float] = []
    for index in range(0, len(mono_samples), channel_count):
        frame = mono_samples[index:index + channel_count]
        if not frame:
            continue
        downmixed.append(sum(frame) / len(frame))
    return downmixed


def resample_samples(
    samples: list[float],
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> list[float]:
    if not samples or input_sample_rate == output_sample_rate:
        return samples
    if len(samples) == 1:
        return samples[:]

    output_length = max(1, round(len(samples) * output_sample_rate / input_sample_rate))
    if output_length == 1:
        return [samples[0]]

    position_scale = (len(samples) - 1) / (output_length - 1)
    resampled: list[float] = []
    for output_index in range(output_length):
        position = output_index * position_scale
        left_index = int(position)
        right_index = min(left_index + 1, len(samples) - 1)
        fraction = position - left_index
        value = (
            samples[left_index] * (1.0 - fraction)
            + samples[right_index] * fraction
        )
        resampled.append(value)
    return resampled


def pcm16le_bytes_to_floats(audio_bytes: bytes) -> list[float]:
    if not audio_bytes:
        return []

    trimmed_length = len(audio_bytes) - (len(audio_bytes) % 2)
    if trimmed_length <= 0:
        return []

    pcm_samples = array("h")
    pcm_samples.frombytes(audio_bytes[:trimmed_length])
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    return [sample / 32768.0 for sample in pcm_samples]
