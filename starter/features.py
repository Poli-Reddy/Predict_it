"""
Audio helpers for the AMD assignment.

What you get here: loading, framing, a deliberately basic VAD, a pitch tracker,
a few spectral descriptors, and a narrowband tone detector.

What you do NOT get: features that solve the problem. The VAD in particular is
about as naive as it gets -- it thresholds energy and will happily call a
packet-loss gap "silence". Improving it is most of the job.

Causality: use `checkpoint_slices()` (or slice yourself) so that features for
the checkpoint at time T only ever see audio in [0, T]. The scorer replays your
probabilities as a live dialer; a live dialer cannot hear the future.
"""
import numpy as np
import soundfile as sf
from scipy import signal as sps

SR = 8000                                    # kit audio is 8 kHz mono
CHECKPOINTS_MS = list(range(500, 6001, 250))  # 500, 750, ... 6000
FRAME_MS = 20
HOP_MS = 10


# ----------------------------------------------------------------- loading

def load_wav(path, sr=SR):
    """Read a wav as mono float32 in [-1, 1], resampled to `sr` if needed."""
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        n = int(round(len(y) * sr / file_sr))
        y = sps.resample(y, n).astype("float32")
    return y


def checkpoint_slices(y, sr=SR, checkpoints_ms=None):
    """Yield (checkpoint_ms, audio_up_to_that_moment).

    The slice is empty-safe: if the file is shorter than the checkpoint you get
    whatever audio exists. Deciding what to do with a nearly-empty slice is
    part of the problem.
    """
    for ms in (checkpoints_ms or CHECKPOINTS_MS):
        yield ms, y[: int(sr * ms / 1000)]


# ----------------------------------------------------------------- framing

def frames(y, sr=SR, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    """Split into overlapping frames -> shape (n_frames, frame_len)."""
    flen = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(y) < flen:
        return np.zeros((0, flen), dtype="float32")
    n = 1 + (len(y) - flen) // hop
    idx = np.arange(flen)[None, :] + hop * np.arange(n)[:, None]
    return y[idx]


def rms_envelope(y, sr=SR):
    """Per-frame RMS, in dBFS. Silence floors at -100 dB rather than -inf."""
    F = frames(y, sr)
    if len(F) == 0:
        return np.zeros(0, dtype="float32")
    r = np.sqrt((F ** 2).mean(axis=1) + 1e-12)
    return (20 * np.log10(np.maximum(r, 1e-5))).astype("float32")


def zero_crossing_rate(y, sr=SR):
    F = frames(y, sr)
    if len(F) == 0:
        return np.zeros(0, dtype="float32")
    return (np.abs(np.diff(np.signbit(F), axis=1)).sum(axis=1) / F.shape[1]).astype("float32")


# ----------------------------------------------------------------- naive VAD

def simple_vad(y, sr=SR, margin_db=8.0):
    """Energy-only VAD. Baseline quality on purpose.

    Estimates a noise floor from the quietest 10% of frames and calls a frame
    'speech' if it sits `margin_db` above that. No hysteresis, no minimum
    duration, no spectral test.

    Known failure modes, all of which appear in the data:
      * comfort noise / DTX -> the floor is not silence, so the margin is wrong
      * packet-loss gaps    -> read as pauses that were never pauses
      * AGC pumping         -> the floor drifts within one call

    Returns a bool array, one entry per frame.
    """
    env = rms_envelope(y, sr)
    if len(env) == 0:
        return np.zeros(0, dtype=bool)
    floor = np.percentile(env, 10)
    return env > (floor + margin_db)


def speech_segments(is_speech, hop_ms=HOP_MS):
    """Turn a per-frame bool mask into [(start_s, end_s), ...]."""
    out = []
    start = None
    for i, v in enumerate(is_speech):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start * hop_ms / 1000.0, i * hop_ms / 1000.0))
            start = None
    if start is not None:
        out.append((start * hop_ms / 1000.0, len(is_speech) * hop_ms / 1000.0))
    return out


# ----------------------------------------------------------------- pitch

def pitch_track(y, sr=SR, fmin=60.0, fmax=400.0):
    """Autocorrelation pitch tracker. Returns Hz per frame, 0.0 = unvoiced.

    This is an algorithm, not a pretrained model, so it is allowed.
    """
    F = frames(y, sr)
    if len(F) == 0:
        return np.zeros(0, dtype="float32")
    lo, hi = int(sr / fmax), int(sr / fmin)
    out = np.zeros(len(F), dtype="float32")
    for i, fr in enumerate(F):
        fr = fr - fr.mean()
        if np.sqrt((fr ** 2).mean()) < 1e-4:
            continue
        ac = np.correlate(fr, fr, mode="full")[len(fr) - 1:]
        if len(ac) <= hi or ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + lo
        if ac[k] / ac[0] > 0.3:          # weak voicing test
            out[i] = sr / k
    return out


# ----------------------------------------------------------------- spectral

def spectral_stats(y, sr=SR):
    """Mean spectral centroid, flatness and rolloff over the slice.

    Flatness is the one worth thinking about: tones and recorded greetings sit
    in a different place from live speech on a noisy line.
    """
    if len(y) < 256:
        return {"centroid": 0.0, "flatness": 0.0, "rolloff85": 0.0}
    f, t, Z = sps.stft(y, fs=sr, nperseg=256, noverlap=128)
    mag = np.abs(Z) + 1e-10
    power = mag ** 2
    total = power.sum(axis=0)
    centroid = float((f[:, None] * power).sum(axis=0).mean() / max(total.mean(), 1e-10))
    gmean = np.exp(np.log(mag).mean(axis=0))
    amean = mag.mean(axis=0)
    flatness = float((gmean / np.maximum(amean, 1e-10)).mean())
    csum = np.cumsum(power, axis=0) / np.maximum(total, 1e-10)
    idx = (csum >= 0.85).argmax(axis=0)
    rolloff = float(f[idx].mean())
    return {"centroid": centroid, "flatness": flatness, "rolloff85": rolloff}


def tone_score(y, sr=SR, fmin=350.0, fmax=1600.0):
    """How much does this slice look like a sustained narrowband tone?

    Voicemail beeps and IVR tones are narrowband and steady; speech is not.
    Returns roughly 0..1 -- peak bin power over total, averaged across frames
    where a single bin dominates.
    """
    if len(y) < 512:
        return 0.0
    f, t, Z = sps.stft(y, fs=sr, nperseg=512, noverlap=256)
    band = (f >= fmin) & (f <= fmax)
    if not band.any():
        return 0.0
    power = (np.abs(Z) ** 2)
    tot = power.sum(axis=0) + 1e-12
    peak = power[band].max(axis=0)
    return float(np.mean(peak / tot))


# ----------------------------------------------------------------- convenience

def turn_shape(y, sr=SR):
    """A few timing descriptors of the slice, from the naive VAD.

    These are the obvious starting point -- and on their own they are not
    enough. Look at where they mislead you.
    """
    seg = speech_segments(simple_vad(y, sr))
    dur = len(y) / sr
    if not seg:
        return {"onset_s": dur, "first_speech_s": 0.0, "longest_pause_s": dur,
                "speech_ratio": 0.0, "n_segments": 0}
    onset = seg[0][0]
    first = seg[0][1] - seg[0][0]
    pauses = [seg[i + 1][0] - seg[i][1] for i in range(len(seg) - 1)]
    trailing = dur - seg[-1][1]
    speech = sum(e - s for s, e in seg)
    return {
        "onset_s": onset,
        "first_speech_s": first,
        "longest_pause_s": max(pauses + [trailing]) if (pauses or trailing) else 0.0,
        "speech_ratio": speech / dur if dur else 0.0,
        "n_segments": len(seg),
    }
