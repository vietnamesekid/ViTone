from __future__ import annotations

import os
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf
from lhotse import RecordingSet

from vitone.data.config import Config, SourceConfig
from vitone.data.ingest import FULL_SCALE, ingest, recording_id, to_recording
from vitone.data.pipeline import read_jsonl, run_ingest

SR = 44100


def tone(seconds: float = 2.0, channels: int = 2, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    x = amp * np.sin(2 * np.pi * 440 * t)
    return np.repeat(x[:, None], channels, axis=1).astype(np.float32)


def encode(x: np.ndarray, path: Path, codec: str, sr: int = SR, sample_format: str | None = None) -> Path:
    """Encode (frames, channels) float audio with PyAV/FFmpeg."""
    layout = "stereo" if x.shape[1] == 2 else "mono"
    with av.open(str(path), "w") as out:
        stream = out.add_stream(codec, rate=sr, layout=layout)
        ctx = stream.codec_context
        if sample_format:
            ctx.format = sample_format
        # Re-chunk to the codec's fixed frame size (e.g. 960 for Opus) and flush at the end.
        resampler = av.AudioResampler(format=ctx.format.name, layout=layout, rate=sr, frame_size=ctx.frame_size or None)
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(x.T), format="fltp", layout=layout)
        frame.sample_rate = sr
        for f in [*resampler.resample(frame), *resampler.resample(None)]:
            out.mux(stream.encode(f))
        out.mux(stream.encode(None))
    return path


def white_noise(cutoff: float, sr: int, seconds: float = 5.0) -> np.ndarray:
    n = int(sr * seconds)
    spec = np.fft.rfft(np.random.default_rng(0).standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / sr) > cutoff] = 0
    x = np.fft.irfft(spec, n)
    return (0.1 * x / np.abs(x).max()).astype(np.float32)


@pytest.mark.parametrize("fmt,subtype", [("wav", "PCM_16"), ("flac", "PCM_24"), ("wav", "FLOAT")])
def test_lossless_is_referenced_in_place(tmp_path, fmt, subtype):
    src = tmp_path / f"a.{fmt}"
    sf.write(src, tone(), SR, subtype=subtype)
    r = ingest(src, "a", tmp_path / "out")
    assert r.lossless and r.audio_path == str(src) and r.gain == 1.0
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "name,codec,sample_format,subtype,lossless",
    [
        ("a.m4a", "aac", None, "PCM_16", False),
        ("a.mp3", "libmp3lame", None, "PCM_16", False),
        ("a.ogg", "libopus", None, "PCM_16", False),
        ("a.mp4", "aac", None, "PCM_16", False),
        ("a.m4a", "alac", "s16p", "PCM_16", True),
        ("a.m4a", "alac", "s32p", "PCM_24", True),
        ("a.mkv", "pcm_s16le", None, "PCM_16", True),
        ("a.mkv", "pcm_f32le", None, "FLOAT", True),
        ("a.mkv", "pcm_f64le", None, "DOUBLE", True),
    ],
)
def test_other_formats_are_decoded(tmp_path, name, codec, sample_format, subtype, lossless):
    sr = 48000 if codec == "libopus" else SR  # Opus only supports 48 kHz
    src = encode(tone(sr=sr), tmp_path / name, codec, sr, sample_format)
    r = ingest(src, "a", tmp_path / "out")
    info = sf.info(r.audio_path)
    assert info.subtype == subtype and r.lossless == lossless
    assert (info.samplerate, info.channels) == (sr, 2)  # native rate and layout kept
    assert r.num_samples == info.frames and abs(r.duration - 2.0) < 0.1  # codec priming/padding only


def test_lossy_overs_are_scaled_not_clipped(tmp_path):
    # A full-scale square wave makes the AAC decoder overshoot 0 dBFS (Gibbs ringing).
    x = np.sign(tone(amp=1.0)).astype(np.float32) * 0.999
    r = ingest(encode(x, tmp_path / "a.m4a", "aac"), "a", tmp_path / "out")
    assert r.peak > 1.0 and r.gain < 1.0
    y, _ = sf.read(r.audio_path)
    assert y.max() <= FULL_SCALE["PCM_16"] and y.min() >= -1.0  # int16 range is asymmetric
    assert max(y.max() / FULL_SCALE["PCM_16"], -y.min()) == pytest.approx(1.0, abs=2**-15)  # extreme at full scale


@pytest.mark.parametrize("codec,dtype", [("pcm_f32le", "float32"), ("pcm_f64le", "float64")])
def test_float_source_is_stored_bit_exact(tmp_path, codec, dtype):
    x = tone(amp=1.5)  # above 0 dBFS: representable only in float
    r = ingest(encode(x, tmp_path / "a.mkv", codec), "a", tmp_path / "out")
    y, _ = sf.read(r.audio_path, dtype=dtype)
    assert r.gain == 1.0 and np.array_equal(y, x.astype(dtype))


def test_integer_source_with_negative_full_scale_is_bit_exact(tmp_path):
    x = tone()
    x[100] = -1.0  # becomes -32768, the one value with no positive counterpart
    src = encode(x, tmp_path / "a.mkv", "pcm_s16le")
    with av.open(str(src)) as c:  # reference: the integers FFmpeg stored
        ref = np.concatenate([f.to_ndarray().reshape(-1, 2) for f in c.decode(audio=0)])
    r = ingest(src, "a", tmp_path / "out")
    y, _ = sf.read(r.audio_path, dtype="int16")
    assert r.gain == 1.0 and ref.min() == -32768
    np.testing.assert_array_equal(y, ref)


def test_channel_correlation(tmp_path):
    rng = np.random.default_rng(0)
    same = np.repeat(rng.standard_normal((SR, 1)), 2, axis=1) * 0.1
    indep = rng.standard_normal((SR, 2)) * 0.1
    sf.write(tmp_path / "same.wav", same, SR)
    sf.write(tmp_path / "indep.wav", indep, SR)
    sf.write(tmp_path / "mono.wav", same[:, 0], SR)
    assert ingest(tmp_path / "same.wav", "s", tmp_path).channel_correlation == pytest.approx(1.0, abs=1e-3)
    assert abs(ingest(tmp_path / "indep.wav", "i", tmp_path).channel_correlation) < 0.05
    assert ingest(tmp_path / "mono.wav", "m", tmp_path).channel_correlation is None


@pytest.mark.parametrize("sr,cutoff", [(SR, 4000), (SR, 8000), (SR, 16000), (16000, 4000), (96000, 20000)])
def test_bandwidth_finds_lowpass_cutoff(tmp_path, sr, cutoff):
    # 96 kHz exercises a 10 ms hop longer than the FFT window.
    sf.write(tmp_path / "n.wav", white_noise(cutoff, sr), sr, subtype="FLOAT")
    bw = ingest(tmp_path / "n.wav", "n", tmp_path).bandwidth_hz
    # Blackman-Harris main lobe is +/-4 bins wide, so the edge can smear up to 4 bins.
    assert abs(bw - cutoff) <= 4 * sr / 512


def test_cache_reuse_and_invalidation(tmp_path):
    # Every decode writes a new file (atomic rename), so a changed inode means "re-decoded".
    src = encode(tone(), tmp_path / "a.m4a", "aac")
    first = ingest(src, "a", tmp_path / "out")
    inode = Path(first.audio_path).stat().st_ino

    assert ingest(src, "a", tmp_path / "out", cached=first.to_dict()) == first

    os.utime(src, ns=(first.source_mtime_ns + 10**9,) * 2)  # touched, same bytes
    touched = ingest(src, "a", tmp_path / "out", cached=first.to_dict())
    assert touched.source_mtime_ns == first.source_mtime_ns + 10**9
    assert Path(first.audio_path).stat().st_ino == inode

    moved = tmp_path / "moved" / "a.m4a"
    moved.parent.mkdir()
    src.rename(moved)
    assert ingest(moved, "a", tmp_path / "out", cached=touched.to_dict()).source_path == str(moved)
    assert Path(first.audio_path).stat().st_ino == inode

    ingest(moved, "a", tmp_path / "out", cached={**first.to_dict(), "version": first.version - 1})
    assert Path(first.audio_path).stat().st_ino != inode  # stale version is recomputed


def test_cache_follows_moved_in_place_source(tmp_path):
    sf.write(tmp_path / "a.wav", tone(), SR)
    first = ingest(tmp_path / "a.wav", "a", tmp_path / "out")
    (tmp_path / "b").mkdir()
    moved = (tmp_path / "a.wav").rename(tmp_path / "b" / "a.wav")
    again = ingest(moved, "a", tmp_path / "out", cached=first.to_dict())
    assert again.audio_path == again.source_path == str(moved)


def test_recording_slices_match_source(tmp_path):
    x = tone(seconds=3.0)
    sf.write(tmp_path / "a.flac", x, SR, subtype="PCM_24")
    rec = to_recording(ingest(tmp_path / "a.flac", "a", tmp_path))
    assert rec == RecordingSet.from_recordings([rec])["a"]
    np.testing.assert_allclose(rec.load_audio(offset=1.0, duration=0.5), x[SR : SR + SR // 2].T, atol=1e-6)
    assert rec.resample(16000).load_audio(offset=1.0, duration=0.5).shape == (2, 8000)


def test_recording_id_keeps_vietnamese_and_subdirs():
    assert recording_id("s", Path("Mùa 2/Tập 1 - Khách mời.m4a")) == "s-mùa_2_tập_1_khách_mời"
    assert recording_id("s", Path("Tập 1.m4a")) != recording_id("s", Path("Tạp 1.m4a"))


def test_run_ingest(tmp_path):
    raw = tmp_path / "raw"
    (raw / "season 1").mkdir(parents=True)
    sf.write(raw / "season 1" / "ep 1.wav", tone(), SR)
    encode(tone(), raw / "ep 2.m4a", "aac")
    (raw / ".DS_Store").write_bytes(b"\0")
    (raw / "._ep 2.m4a").write_bytes(b"\0")  # AppleDouble metadata
    (raw / "cover.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    (raw / "notes.txt").write_text("show notes")

    out = tmp_path / "out"
    rows = run_ingest(SourceConfig("s", raw), out)
    assert [r["recording_id"] for r in rows] == ["s-ep_2", "s-season_1_ep_1"]
    assert all(Path(r["audio_path"]).is_absolute() for r in rows)
    assert read_jsonl(out / "manifests" / "ingest.jsonl") == rows
    recordings = RecordingSet.from_file(out / "manifests" / "recordings.jsonl.gz")
    assert sorted(recordings.ids) == ["s-ep_2", "s-season_1_ep_1"]


def test_run_ingest_rejects_id_clashes(tmp_path):
    sf.write(tmp_path / "Ep 1.wav", tone(), SR)
    sf.write(tmp_path / "ep_1.flac", tone(), SR)
    with pytest.raises(ValueError, match="same recording id"):
        run_ingest(SourceConfig("s", tmp_path), tmp_path / "out")


def test_config_paths_are_relative_to_the_config_file(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    cfg_file = tmp_path / "configs" / "c.yaml"
    cfg_file.write_text("sources:\n  a: {path: ../raw, glob: '*.m4a'}\n")
    monkeypatch.chdir(tmp_path / "configs")  # any working directory
    src = Config.load(Path("c.yaml")).sources["a"]
    assert src.root == (tmp_path / "raw").resolve()
    assert (src.name, src.glob, src.language) == ("a", "*.m4a", "vi")
