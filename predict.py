#!/usr/bin/env python3
"""
AMD (Answering Machine Detection) - Production Inference Script

Features engineered based on dataset analysis:
- TIER 1: Longest pause, voicing rate, tone score, pitch variability, speech density  
- TIER 2: Pause variance, spectral features, energy dynamics

Key insight: Machines talk continuously without pauses. Humans wait for responses.
"""
import argparse
import csv
import os
import sys
import numpy as np

# Import path handling
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "starter"), os.path.join(_HERE, os.pardir)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from features import (
    SR, CHECKPOINTS_MS, load_wav, checkpoint_slices,
    rms_envelope, zero_crossing_rate, pitch_track,
    spectral_stats, tone_score
)

MODEL_PATH = os.path.join(_HERE, "model.npz")


class _LinearClf:
    """Logistic regression for numpy.savez (no pickle)."""

    def __init__(self, w, b, scaler_mean, scaler_scale):
        self.w = np.asarray(w, dtype="float64")
        self.b = b
        self.scaler_mean = np.asarray(scaler_mean, dtype="float64")
        self.scaler_scale = np.asarray(scaler_scale, dtype="float64")

    def predict_proba(self, X):
        # Standardize
        X_scaled = (np.asarray(X, dtype="float64") - self.scaler_mean) / self.scaler_scale
        # Logistic
        z = X_scaled @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def speech_segments(is_speech, hop_ms=10):
    """Convert VAD mask to [(start_s, end_s), ...]."""
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


def improved_vad(y, sr=SR, energy_margin_db=10.0, min_speech_frames=3):
    """
    Improved VAD handling packet loss and comfort noise.
    Uses energy + ZCR with hysteresis.
    """
    env = rms_envelope(y, sr)
    zcr = zero_crossing_rate(y, sr)
    
    if len(env) == 0:
        return np.zeros(0, dtype=bool)
    
    # Adaptive floor (15th percentile handles comfort noise)
    floor = np.percentile(env, 15)
    
    # Energy-based detection
    energy_speech = env > (floor + energy_margin_db)
    
    # ZCR refinement (speech has moderate ZCR)
    zcr_speech = (zcr > 0.02) & (zcr < 0.25)
    
    # Combine
    is_speech = energy_speech | zcr_speech
    
    # Fill short gaps (packet loss)
    max_gap = 5  # ~50ms
    speech_list = list(is_speech)
    for i in range(len(speech_list) - max_gap):
        if speech_list[i] and not any(speech_list[i+1:i+max_gap+1]) and (i+max_gap+1 < len(speech_list)) and speech_list[i+max_gap+1]:
            for j in range(i+1, i+max_gap+1):
                speech_list[j] = True
    
    is_speech = np.array(speech_list, dtype=bool)
    
    # Remove short bursts (noise)
    speech_regions = []
    start = None
    for i, v in enumerate(is_speech):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_speech_frames:
                speech_regions.append((start, i))
            start = None
    if start is not None and len(is_speech) - start >= min_speech_frames:
        speech_regions.append((start, len(is_speech)))
    
    cleaned = np.zeros(len(is_speech), dtype=bool)
    for start, end in speech_regions:
        cleaned[start:end] = True
    
    return cleaned


def extract_features(slice_audio, elapsed_s):
    """
    Feature extraction for one checkpoint.
    CRITICAL: Only uses audio from [0, elapsed_s] - causality enforced.
    
    Returns 20-dimensional feature vector.
    """
    if len(slice_audio) < SR * 0.1:  # < 100ms
        return np.zeros(20, dtype='float32')
    
    # VAD
    vad = improved_vad(slice_audio, SR)
    segments = speech_segments(vad)
    
    # Duration
    dur = len(slice_audio) / SR
    
    # === Pause features (TIER 1) ===
    if len(segments) > 1:
        pauses = [segments[i+1][0] - segments[i][1] for i in range(len(segments)-1)]
        longest_pause = max(pauses)
        mean_pause = np.mean(pauses)
        pause_variance = np.var(pauses)
    else:
        longest_pause = dur - segments[0][1] if segments else dur
        mean_pause = 0.0
        pause_variance = 0.0
    
    # === Speech timing ===
    if segments:
        total_speech = sum(e - s for s, e in segments)
        speech_ratio = total_speech / dur
        onset = segments[0][0]
        first_speech_dur = segments[0][1] - segments[0][0]
    else:
        total_speech = 0.0
        speech_ratio = 0.0
        onset = dur
        first_speech_dur = 0.0
    
    # === Pitch features (TIER 1) ===
    pitch = pitch_track(slice_audio, SR)
    voiced_frames = pitch[pitch > 0]
    voicing_rate = len(voiced_frames) / len(pitch) if len(pitch) > 0 else 0.0
    
    if len(voiced_frames) > 2:
        pitch_mean = float(np.mean(voiced_frames))
        pitch_std = float(np.std(voiced_frames))
        pitch_range = float(np.max(voiced_frames) - np.min(voiced_frames))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
    
    # === Tone detection (TIER 1) ===
    tone = tone_score(slice_audio, SR)
    
    # === Spectral features (TIER 2) ===
    spectral = spectral_stats(slice_audio, SR)
    spectral_flatness = spectral['flatness']
    spectral_centroid = spectral['centroid']
    spectral_rolloff = spectral['rolloff85']
    
    # === Energy features ===
    rms = rms_envelope(slice_audio, SR)
    rms_mean = float(np.mean(rms)) if len(rms) > 0 else -100.0
    rms_std = float(np.std(rms)) if len(rms) > 0 else 0.0
    
    # === ZCR ===
    zcr = zero_crossing_rate(slice_audio, SR)
    zcr_mean = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
    
    # === Segment count ===
    n_segments = len(segments)
    
    # === Temporal context ===
    time_factor = min(elapsed_s / 2.5, 1.0)
    
    # Feature vector (20 features - MUST match training!)
    features = np.array([
        longest_pause,           # 0
        voicing_rate,            # 1
        tone,                    # 2
        pitch_std,               # 3
        speech_ratio,            # 4
        pause_variance,          # 5
        mean_pause,              # 6
        spectral_flatness,       # 7
        pitch_range,             # 8
        onset,                   # 9
        first_speech_dur,        # 10
        n_segments,              # 11
        rms_mean,                # 12
        rms_std,                 # 13
        zcr_mean,                # 14
        spectral_centroid,       # 15
        spectral_rolloff,        # 16
        pitch_mean,              # 17
        elapsed_s,               # 18
        time_factor,             # 19
    ], dtype='float32')
    
    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features


class Model:
    """Model wrapper with loading and inference."""

    def __init__(self, clf=None):
        self.clf = clf

    @classmethod
    def load(cls, path=MODEL_PATH):
        if os.path.exists(path):
            d = np.load(path)
            return cls(_LinearClf(d["w"], float(d["b"]), d["scaler_mean"], d["scaler_scale"]))
        return cls(None)

    def p_machine(self, feats):
        if self.clf is None:
            return 0.5  # Fallback
        return float(self.clf.predict_proba(feats.reshape(1, -1))[0][1])


def call_list(data_dir):
    """Load call list from labels.csv (without reading label column)."""
    with open(os.path.join(data_dir, "labels.csv"), newline="") as fh:
        seen = {}
        for r in csv.DictReader(fh):
            seen.setdefault(r["call_id"].strip(), (r.get("audio_file") or "").strip())
    return sorted(seen.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    model = Model.load()
    rows = []
    calls = call_list(args.data_dir)
    
    for n, (call_id, rel) in enumerate(calls, 1):
        path = os.path.join(args.data_dir, rel) if rel else ""
        if not path or not os.path.exists(path):
            # Missing call - emit neutral
            rows += [(call_id, ms, 0.5) for ms in CHECKPOINTS_MS]
            continue
        
        y = load_wav(path, SR)
        
        # Collect raw probabilities first
        raw_probs = []
        for ms, sl in checkpoint_slices(y, SR):
            p = model.p_machine(extract_features(sl, ms / 1000.0))
            raw_probs.append((ms, float(min(max(p, 0.0), 1.0))))
        
        # Apply temporal smoothing (EMA) to stabilize predictions
        # Machines talk continuously, so prob should increase or stay high
        # Humans create pauses, so prob should dip and recover
        smoothed_probs = []
        alpha = 0.4  # Smoothing factor (0.3-0.5 works well)
        prev_p = 0.5  # Start neutral
        
        for i, (ms, raw_p) in enumerate(raw_probs):
            # EMA: smooth_p = alpha * raw_p + (1 - alpha) * prev_p
            smooth_p = alpha * raw_p + (1.0 - alpha) * prev_p
            
            # Enforce monotonic increase tendency on machines
            # If we've been high, don't drop too much (machines stay on)
            if prev_p > 0.7 and smooth_p < prev_p - 0.15:
                smooth_p = prev_p - 0.15
            
            smoothed_probs.append((call_id, ms, round(smooth_p, 5)))
            prev_p = smooth_p
        
        rows.extend(smoothed_probs)
        
        if n % 50 == 0:
            print(f"  ...{n}/{len(calls)} calls", file=sys.stderr, flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["call_id", "checkpoint_ms", "p_machine"])
        w.writerows(rows)
    print(f"wrote {args.out}  ({len(calls)} calls, {len(rows)} rows)")


if __name__ == "__main__":
    main()
