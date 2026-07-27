#!/usr/bin/env python3
"""
Training script for AMD model.
Extracts features from all checkpoints and trains classifier.
"""
import os
import sys
import csv
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'starter'))
from features import (
    SR, CHECKPOINTS_MS, load_wav, checkpoint_slices,
    rms_envelope, zero_crossing_rate, pitch_track,
    spectral_stats, tone_score, simple_vad, speech_segments
)

def improved_vad(y, sr=SR, energy_margin_db=10.0, min_speech_frames=3):
    """
    Improved VAD with hysteresis to handle packet loss.
    Uses energy and zero-crossing rate.
    """
    env = rms_envelope(y, sr)
    zcr = zero_crossing_rate(y, sr)
    
    if len(env) == 0:
        return np.zeros(0, dtype=bool)
    
    # Adaptive floor (use 15th percentile to handle comfort noise)
    floor = np.percentile(env, 15)
    
    # Energy-based speech detection with larger margin
    energy_speech = env > (floor + energy_margin_db)
    
    # ZCR-based refinement (speech has moderate ZCR, silence is low, noise is high)
    zcr_speech = (zcr > 0.02) & (zcr < 0.25)
    
    # Combine with OR (either condition)
    is_speech = energy_speech | zcr_speech
    
    # Hysteresis: fill short gaps (packet loss)
    max_gap = 5  # frames (~50ms)
    speech_list = list(is_speech)
    for i in range(len(speech_list) - max_gap):
        if speech_list[i] and not any(speech_list[i+1:i+max_gap+1]) and (i+max_gap+1 < len(speech_list)) and speech_list[i+max_gap+1]:
            for j in range(i+1, i+max_gap+1):
                speech_list[j] = True
    
    is_speech = np.array(speech_list, dtype=bool)
    
    # Remove very short speech bursts (noise)
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

def extract_features(slice_audio, elapsed_s, sr=SR):
    """
    Extract comprehensive features for one checkpoint.
    Based on dataset analysis insights.
    """
    if len(slice_audio) < sr * 0.1:  # Less than 100ms
        # Too short to analyze
        return np.zeros(20, dtype='float32')
    
    # Improved VAD
    vad = improved_vad(slice_audio, sr)
    segments = speech_segments(vad)
    
    # === TIER 1 FEATURES (from analysis) ===
    
    # 1. Longest pause (55% discrimination)
    dur = len(slice_audio) / sr
    if len(segments) > 1:
        pauses = [segments[i+1][0] - segments[i][1] for i in range(len(segments)-1)]
        longest_pause = max(pauses) if pauses else 0.0
        mean_pause = np.mean(pauses) if pauses else 0.0
        pause_variance = np.var(pauses) if pauses else 0.0
    else:
        longest_pause = dur - segments[0][1] if segments else dur
        mean_pause = 0.0
        pause_variance = 0.0
    
    # 2. Speech density (16% discrimination)
    if segments:
        total_speech = sum(e - s for s, e in segments)
        speech_ratio = total_speech / dur if dur > 0 else 0.0
        onset = segments[0][0]
        first_speech_dur = segments[0][1] - segments[0][0]
    else:
        speech_ratio = 0.0
        onset = dur
        first_speech_dur = 0.0
    
    # 3. Voicing rate (53% discrimination!)
    pitch = pitch_track(slice_audio, sr)
    voiced_frames = pitch[pitch > 0]
    voicing_rate = len(voiced_frames) / len(pitch) if len(pitch) > 0 else 0.0
    
    # 4. Pitch variability (31% discrimination)
    if len(voiced_frames) > 2:
        pitch_mean = float(np.mean(voiced_frames))
        pitch_std = float(np.std(voiced_frames))
        pitch_range = float(np.max(voiced_frames) - np.min(voiced_frames))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
    
    # 5. Tone score (38% discrimination)
    tone = tone_score(slice_audio, sr)
    
    # === TIER 2 FEATURES ===
    
    # Spectral features
    spectral = spectral_stats(slice_audio, sr)
    spectral_flatness = spectral['flatness']
    spectral_centroid = spectral['centroid']
    spectral_rolloff = spectral['rolloff85']
    
    # Energy features (with caution on generalization)
    rms = rms_envelope(slice_audio, sr)
    if len(rms) > 0:
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
    else:
        rms_mean = -100.0
        rms_std = 0.0
    
    # Zero crossing rate
    zcr = zero_crossing_rate(slice_audio, sr)
    zcr_mean = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
    
    # Segment features
    n_segments = len(segments)
    
    # Temporal context
    time_factor = min(elapsed_s / 2.5, 1.0)  # Confidence ramp
    
    # Feature vector (20 features)
    features = np.array([
        # TIER 1 - Strongest discriminators
        longest_pause,           # 0
        voicing_rate,            # 1
        tone,                    # 2
        pitch_std,               # 3
        speech_ratio,            # 4
        
        # TIER 2 - Supporting features
        pause_variance,          # 5
        mean_pause,              # 6
        spectral_flatness,       # 7
        pitch_range,             # 8
        onset,                   # 9
        
        # Additional features
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

def build_training_data():
    """Build training matrix from all checkpoints."""
    data_dir = Path(__file__).parent / 'amd_data' / 'handout'
    labels_file = data_dir / 'labels.csv'
    
    with open(labels_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    X = []
    y = []
    
    print(f"Building training data from {len(rows)} calls...")
    
    for i, row in enumerate(rows, 1):
        call_id = row['call_id']
        audio_file = data_dir / row['audio_file']
        label = 1 if row['label'] == 'machine' else 0
        
        if not audio_file.exists():
            continue
        
        try:
            audio = load_wav(str(audio_file), SR)
            
            for ms, slice_audio in checkpoint_slices(audio, SR):
                feats = extract_features(slice_audio, ms / 1000.0, SR)
                X.append(feats)
                y.append(label)
        
        except Exception as e:
            print(f"Error processing {call_id}: {e}")
            continue
        
        if i % 25 == 0:
            print(f"  Processed {i}/{len(rows)} calls...")
    
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    print(f"\nTraining matrix: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class 0 (human): {np.sum(y==0)} samples")
    print(f"  Class 1 (machine): {np.sum(y==1)} samples")
    
    return X, y

def train_and_save():
    """Train model and save to disk."""
    print("="*60)
    print("TRAINING AMD MODEL")
    print("="*60)
    
    X, y = build_training_data()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n" + "="*60)
    print("TRAINING CLASSIFIERS")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    }
    
    best_model = None
    best_score = -1
    best_name = None
    
    for name, model in models.items():
        print(f"\n{name}:")
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc', n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"  ROC-AUC: {mean_score:.4f} ± {std_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (ROC-AUC: {best_score:.4f})")
    
    # Train on full dataset
    print("\nTraining final model on full dataset...")
    best_model.fit(X_scaled, y)
    
    # For Logistic Regression, save weights
    if isinstance(best_model, LogisticRegression):
        weights = best_model.coef_[0]
        bias = best_model.intercept_[0]
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_
        
        # Save
        output_file = Path(__file__).parent / 'model.npz'
        np.savez(output_file, 
                 w=weights, 
                 b=bias,
                 scaler_mean=scaler_mean,
                 scaler_scale=scaler_scale)
        print(f"Saved model to {output_file}")
    else:
        # For tree-based models, we need different strategy
        # For now, use Logistic Regression as fallback
        print("Tree-based model - retraining Logistic Regression for deployment...")
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_scaled, y)
        
        weights = lr.coef_[0]
        bias = lr.intercept_[0]
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_
        
        output_file = Path(__file__).parent / 'model.npz'
        np.savez(output_file, 
                 w=weights, 
                 b=bias,
                 scaler_mean=scaler_mean,
                 scaler_scale=scaler_scale)
        print(f"Saved Logistic Regression model to {output_file}")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    train_and_save()
