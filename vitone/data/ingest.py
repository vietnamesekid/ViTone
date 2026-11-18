"""Ingest: register raw media as Lhotse recordings and measure them.

Ingest never changes content. It does not resample, downmix or normalize
loudness. Consumers resample on load to the rate they need, and the export
stage applies model-specific processing.

Input routing:
  * Lossless audio that libsndfile reads (WAV, FLAC, AIFF, ...) is referenced in place.
  * Anything else (AAC/M4A, MP3, Opus, ALAC, video containers, ...) is decoded once
    with PyAV (bundled FFmpeg). Sample rate and channel layout are taken from the
    decoded frames, not the container header, because HE-AAC headers report the core
    rate and mono. Storage depends on the decoded samples:
      - lossy codecs and 16-bit integer:  FLAC 16-bit (a lossy codec's noise floor is
        far above 16-bit quantization noise)
      - wider integer (ALAC 24, PCM 24/32): FLAC 24-bit (beyond any converter's real
        dynamic range)
      - floating point:                   RF64 float32/float64, bit-exact

Integer storage cannot represent samples above full scale, which lossy decoders
produce. In that case one linear gain is applied so the extreme sample lands on
full scale. It is recorded in `gain` and derived from the data, not tuned.
Integer-native lossless sources always fit, so they are never scaled.

Files that are not audio (cover art, show notes) raise NotAudioError, so
callers can skip them.

Measurements, used to route and filter sources:
  * peak, gain
  * channel_correlation: ~1.0 for a mixed-down stereo master, low for one mic per channel
  * bandwidth_hz: NeMo SDP EstimateBandwidth (mean power spectrum, Blackman-Harris,
    n_fft=512, 10 ms hop, -50 dB below the peak), computed over the whole file.
    Resolution is sr/512, and the window's main lobe can push the estimate up to
    4 bins above a sharp cutoff.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from lhotse import AudioSource, Recording

VERSION = 3
LOSSLESS_CONTAINERS = {"WAV", "WAVEX", "FLAC", "AIFF", "W64", "RF64", "CAF"}
LOSSLESS_SUBTYPES = ("PCM", "FLOAT", "DOUBLE", "ALAC")
LOSSLESS_CODECS = ("pcm_", "flac", "alac", "wavpack", "tta", "ape", "mlp", "truehd")
FULL_SCALE = {"PCM_16": 1 - 2**-15, "PCM_24": 1 - 2**-23}  # largest positive value; negative limit is -1
BLOCK = 1 << 16  # frames per block when streaming lossless files
BANDWIDTH_THRESHOLD_DB = -50.0  # NeMo SDP EstimateBandwidth default

# 4-term Blackman-Harris coefficients.
_BH = (0.35875, 0.48829, 0.14128, 0.01168)


class NotAudioError(ValueError):
    """The file has no decodable audio stream."""


@dataclass(frozen=True)
class Probe:
    in_place: bool  # libsndfile reads it losslessly; no decoded copy needed
    lossless: bool  # the codec is lossless
    codec: str
    bit_rate: int | None
    sample_rate: int
    channels: int
    sample_format: str  # FFmpeg sample format of decoded frames (e.g. "fltp", "s16p", "s32p")

    @property
    def storage(self) -> tuple[str, str, str]:
        """(libsndfile format, subtype, file suffix) for the decoded copy."""
        fmt = self.sample_format.rstrip("p")
        if not self.lossless or fmt in {"u8", "s16"}:
            return "FLAC", "PCM_16", ".flac"
        if fmt in {"s32", "s64"}:
            return "FLAC", "PCM_24", ".flac"
        return "RF64", "DOUBLE" if fmt == "dbl" else "FLOAT", ".wav"


@dataclass(frozen=True)
class IngestResult:
    recording_id: str
    source_path: str
    source_size: int
    source_mtime_ns: int
    source_sha256: str
    codec: str
    lossless: bool
    sample_rate: int
    channels: int
    bit_rate: int | None
    num_samples: int
    duration: float
    audio_path: str  # what Lhotse reads: the source itself, or the decoded copy
    peak: float
    gain: float  # linear gain applied to the stored copy (1.0 = untouched)
    channel_correlation: float | None
    bandwidth_hz: float
    version: int

    def to_dict(self) -> dict:
        return asdict(self)


def recording_id(source: str, relative_path: Path) -> str:
    """Stable id from the path relative to the source root. Keeps Unicode letters
    (Vietnamese diacritics) and drops only the extension."""
    stem = unicodedata.normalize("NFC", relative_path.with_suffix("").as_posix()).lower()
    return f"{source}-" + re.sub(r"\W+", "_", stem).strip("_")


def ingest(path: Path, rec_id: str, out_dir: Path, cached: dict | None = None) -> IngestResult:
    """Ingest one file. `cached` is a previous result for this id. It is reused when the
    ingest version matches, the stored audio exists, and the source is unchanged:
    same size and mtime, or same SHA-256 if only the mtime moved."""
    st = path.stat()
    digest = None
    if cached and cached.get("version") == VERSION:
        in_place = cached["audio_path"] == cached["source_path"]
        hit = {
            **cached,
            "source_path": str(path),  # the source tree may have moved
            "source_mtime_ns": st.st_mtime_ns,
            "audio_path": str(path) if in_place else cached["audio_path"],
        }
        if Path(hit["audio_path"]).exists():
            if (cached["source_size"], cached["source_mtime_ns"]) == (st.st_size, st.st_mtime_ns):
                return IngestResult(**hit)
            digest = _sha256(path)
            if digest == cached["source_sha256"]:
                return IngestResult(**hit)

    probe = _probe(path)
    stats = _Stats(probe.sample_rate, probe.channels)
    for block in _blocks(path, probe):
        stats.update(block)

    gain = 1.0
    audio_path = path
    if not probe.in_place:
        _, subtype, suffix = probe.storage
        if subtype in FULL_SCALE:
            gain = min(1.0, FULL_SCALE[subtype] / max(stats.max, 1e-12), 1.0 / max(-stats.min, 1e-12))
        audio_path = out_dir / f"{rec_id}{suffix}"
        _write(path, audio_path, probe, gain)

    return IngestResult(
        recording_id=rec_id,
        source_path=str(path),
        source_size=st.st_size,
        source_mtime_ns=st.st_mtime_ns,
        source_sha256=digest or _sha256(path),
        codec=probe.codec,
        lossless=probe.lossless,
        sample_rate=probe.sample_rate,
        channels=probe.channels,
        bit_rate=probe.bit_rate,
        num_samples=stats.frames,
        duration=round(stats.frames / probe.sample_rate, 6),
        audio_path=str(audio_path),
        peak=round(max(stats.max, -stats.min), 6),
        gain=gain,
        channel_correlation=stats.correlation(),
        bandwidth_hz=stats.bandwidth(BANDWIDTH_THRESHOLD_DB),
        version=VERSION,
    )


def to_recording(result: IngestResult | dict) -> Recording:
    """Build the Lhotse recording from ingest results, without reopening the audio."""
    r = result if isinstance(result, dict) else result.to_dict()
    return Recording(
        id=r["recording_id"],
        sources=[AudioSource(type="file", channels=list(range(r["channels"])), source=r["audio_path"])],
        sampling_rate=r["sample_rate"],
        num_samples=r["num_samples"],
        duration=r["num_samples"] / r["sample_rate"],
    )


def _probe(path: Path) -> Probe:
    try:
        info = sf.info(str(path))
        if info.format in LOSSLESS_CONTAINERS and info.subtype.startswith(LOSSLESS_SUBTYPES):
            codec = f"{info.format}/{info.subtype}"
            return Probe(True, True, codec, None, info.samplerate, info.channels, info.subtype)
    except sf.LibsndfileError:
        pass
    try:
        container = av.open(str(path))
    except av.error.InvalidDataError as e:
        raise NotAudioError(f"{path}: not a media file") from e
    with container:
        if not container.streams.audio:
            raise NotAudioError(f"{path}: no audio stream")
        stream = container.streams.audio[0]
        first = next(container.decode(stream), None)
        if first is None:
            raise NotAudioError(f"{path}: no decodable audio")
        ctx = stream.codec_context
        return Probe(
            in_place=False,
            lossless=ctx.name.startswith(LOSSLESS_CODECS),
            codec=ctx.name,
            bit_rate=ctx.bit_rate or None,  # codec-level only; container bit rate includes video
            sample_rate=first.sample_rate,
            channels=first.layout.nb_channels,
            sample_format=first.format.name,
        )


def _blocks(path: Path, probe: Probe) -> Iterator[np.ndarray]:
    """Yield float blocks shaped (frames, channels) at the native rate. Decoded media is
    converted to float64, which holds every integer and float sample format exactly."""
    if probe.in_place:
        yield from sf.blocks(str(path), blocksize=BLOCK, dtype="float32", always_2d=True)
        return
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        to_float = av.AudioResampler(format="dblp")  # sample format only; rate and layout untouched
        for frame in container.decode(stream):
            if (frame.sample_rate, frame.layout.nb_channels) != (probe.sample_rate, probe.channels):
                raise ValueError(f"{path}: sample rate or channel layout changes mid-stream")
            for f in to_float.resample(frame):
                yield f.to_ndarray().T
        for f in to_float.resample(None):
            yield f.to_ndarray().T


def _write(src: Path, dst: Path, probe: Probe, gain: float) -> None:
    fmt, subtype, _ = probe.storage
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    with sf.SoundFile(str(tmp), "w", probe.sample_rate, probe.channels, format=fmt, subtype=subtype) as out:
        for block in _blocks(src, probe):
            out.write(block * gain if gain != 1.0 else block)
    tmp.replace(dst)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class _Stats:
    """Streaming extremes, inter-channel correlation and mean power spectrum."""

    N_FFT = 512

    def __init__(self, sample_rate: int, channels: int):
        self.sr = sample_rate
        self.hop = max(1, sample_rate // 100)  # 10 ms
        n = np.arange(self.N_FFT)
        self.window = sum((-1) ** k * a * np.cos(2 * np.pi * k * n / (self.N_FFT - 1)) for k, a in enumerate(_BH))
        self.frames = 0
        self.max = 0.0
        self.min = 0.0
        self.xx = np.zeros((channels, channels))  # channel Gram matrix
        self.power = np.zeros(self.N_FFT // 2 + 1)
        self.n_spec = 0
        self.tail = np.zeros(0)
        self.skip = 0  # samples to drop before the next frame (hop > N_FFT at high rates)

    def update(self, block: np.ndarray) -> None:
        self.frames += len(block)
        if len(block):
            self.max = max(self.max, float(block.max()))
            self.min = min(self.min, float(block.min()))
        b = block.astype(np.float64)
        self.xx += b.T @ b
        mono = np.concatenate([self.tail, block.mean(axis=1)])
        if self.skip:
            dropped = min(self.skip, len(mono))
            mono, self.skip = mono[dropped:], self.skip - dropped
        if len(mono) >= self.N_FFT:
            frames = np.lib.stride_tricks.sliding_window_view(mono, self.N_FFT)[:: self.hop]
            self.power += (np.abs(np.fft.rfft(frames * self.window, axis=1)) ** 2).sum(axis=0)
            self.n_spec += len(frames)
            next_start = len(frames) * self.hop
            self.skip = max(0, next_start - len(mono))
            mono = mono[next_start:]
        self.tail = mono

    def correlation(self) -> float | None:
        """Mean pairwise cosine similarity between channels (None for mono or silent channels)."""
        c = self.xx.shape[0]
        if c < 2:
            return None
        d = np.sqrt(np.diag(self.xx))
        if not d.all():
            return None
        corr = self.xx / np.outer(d, d)
        return round(float(corr[np.triu_indices(c, 1)].mean()), 4)

    def bandwidth(self, threshold_db: float) -> float:
        if not self.n_spec or not self.power.any():
            return 0.0
        db = 10 * np.log10(np.maximum(self.power / self.power.max(), 1e-30))
        top = np.nonzero(db > threshold_db)[0][-1]
        return round(float(top * self.sr / self.N_FFT), 1)
