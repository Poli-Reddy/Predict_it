#!/usr/bin/env python3
"""
Dataset exploration script for AMD assignment.
Goal: Understand acoustic differences between human and machine calls.
"""
import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path

# Add starter to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'starter'))
from features import (
    load_wav, SR, simple_vad, speech_segments,
    rms_envelope, zero_crossing_rate, pitch_track,
    spectral_stats, tone_score, turn_shape
)

def analyze_audio(audio_path):
    """Extract comprehensive features from one audio file."""
    y = load_wav(audio_path, SR)
    
    # Basic properties
    duration = len(y) / SR
    
    # Energy features
    rms = rms_envelope(y, SR)
    zcr = zero_crossing_rate(y, SR)
    
    # VAD and speech segments
    vad = simple_vad(y, SR)
    segments = speech_segments(vad)
    
    # Speech timing
    if segments:
        onset = segments[0][0]
        first_speech_dur = segments[0][1] - segments[0][0]
        last_segment_end = segments[-1][1]
        
        # Pauses
        pauses = [segments[i+1][0] - segments[i][1] for i in range(len(segments)-1)]
        longest_pause = max(pauses) if pauses else 0.0
        mean_pause = np.mean(pauses) if pauses else 0.0
        pause_variance = np.var(pauses) if pauses else 0.0
        
        # Speech characteristics
        total_speech = sum(e - s for s, e in segments)
        speech_ratio = total_speech / duration
        n_segments = len(segments)
        
        # Longest speech segment
        longest_speech = max(e - s for s, e in segments)
    else:
        onset = duration
        first_speech_dur = 0.0
        last_segment_end = 0.0
        longest_pause = duration
        mean_pause = 0.0
        pause_variance = 0.0
        total_speech = 0.0
        speech_ratio = 0.0
        n_segments = 0
        longest_speech = 0.0
    
    # Pitch
    pitch = pitch_track(y, SR)
    voiced_frames = pitch[pitch > 0]
    if len(voiced_frames) > 0:
        pitch_mean = float(np.mean(voiced_frames))
        pitch_std = float(np.std(voiced_frames))
        pitch_range = float(np.max(voiced_frames) - np.min(voiced_frames))
        voicing_rate = len(voiced_frames) / len(pitch) if len(pitch) > 0 else 0.0
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
        voicing_rate = 0.0
    
    # Spectral
    spectral = spectral_stats(y, SR)
    
    # Tone detection
    tone = tone_score(y, SR)
    
    # Energy statistics
    rms_mean = float(np.mean(rms)) if len(rms) > 0 else -100.0
    rms_std = float(np.std(rms)) if len(rms) > 0 else 0.0
    rms_range = float(np.max(rms) - np.min(rms)) if len(rms) > 0 else 0.0
    
    # ZCR statistics
    zcr_mean = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
    zcr_std = float(np.std(zcr)) if len(zcr) > 0 else 0.0
    
    return {
        'duration': duration,
        'onset_s': onset,
        'first_speech_s': first_speech_dur,
        'longest_speech_s': longest_speech,
        'longest_pause_s': longest_pause,
        'mean_pause_s': mean_pause,
        'pause_variance': pause_variance,
        'speech_ratio': speech_ratio,
        'n_segments': n_segments,
        'last_segment_end_s': last_segment_end,
        'pitch_mean': pitch_mean,
        'pitch_std': pitch_std,
        'pitch_range': pitch_range,
        'voicing_rate': voicing_rate,
        'rms_mean_db': rms_mean,
        'rms_std_db': rms_std,
        'rms_range_db': rms_range,
        'zcr_mean': zcr_mean,
        'zcr_std': zcr_std,
        'spectral_centroid': spectral['centroid'],
        'spectral_flatness': spectral['flatness'],
        'spectral_rolloff': spectral['rolloff85'],
        'tone_score': tone,
    }

def main():
    data_dir = Path(__file__).parent.parent / 'amd_data' / 'handout'
    labels_file = data_dir / 'labels.csv'
    
    # Load labels
    with open(labels_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Analyzing {len(rows)} audio files...")
    
    results = []
    for i, row in enumerate(rows, 1):
        call_id = row['call_id']
        audio_file = data_dir / row['audio_file']
        label = row['label']
        subtype = row.get('subtype', '')
        
        if not audio_file.exists():
            print(f"Warning: {audio_file} not found")
            continue
        
        try:
            features = analyze_audio(str(audio_file))
            features['call_id'] = call_id
            features['label'] = label
            features['subtype'] = subtype
            results.append(features)
            
            if i % 25 == 0:
                print(f"  Processed {i}/{len(rows)} files...")
        except Exception as e:
            print(f"Error processing {call_id}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = Path(__file__).parent / 'dataset_features.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved features to {output_file}")
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for label in ['human', 'machine']:
        subset = df[df['label'] == label]
        print(f"\n{label.upper()} (n={len(subset)})")
        print("-" * 60)
        
        key_features = [
            'onset_s', 'first_speech_s', 'longest_speech_s',
            'longest_pause_s', 'speech_ratio', 'n_segments',
            'pitch_mean', 'pitch_std', 'voicing_rate',
            'rms_mean_db', 'spectral_flatness', 'tone_score'
        ]
        
        for feat in key_features:
            if feat in subset.columns:
                mean_val = subset[feat].mean()
                std_val = subset[feat].std()
                print(f"  {feat:20s}: {mean_val:8.3f} ± {std_val:6.3f}")
    
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    humans = df[df['label'] == 'human']
    machines = df[df['label'] == 'machine']
    
    comparisons = [
        ('onset_s', 'Speech Onset'),
        ('first_speech_s', 'First Utterance Duration'),
        ('longest_pause_s', 'Longest Pause'),
        ('speech_ratio', 'Speech Density'),
        ('pitch_std', 'Pitch Variability'),
        ('tone_score', 'Tone-like Quality'),
    ]
    
    for feat, name in comparisons:
        if feat in df.columns:
            h_mean = humans[feat].mean()
            m_mean = machines[feat].mean()
            diff_pct = abs(h_mean - m_mean) / max(abs(h_mean), abs(m_mean), 1e-9) * 100
            print(f"\n{name}:")
            print(f"  Human:   {h_mean:.3f}")
            print(f"  Machine: {m_mean:.3f}")
            print(f"  Diff:    {diff_pct:.1f}%")

if __name__ == '__main__':
    main()
