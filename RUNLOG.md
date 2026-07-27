# Experiment Run Log - AMD Assignment

All experiments track: detection latency (ms), false-hangup rate (%), decisions made.

---

## Run 00 - Baseline (Provided Heuristic)
**Date**: Initial  
**Change**: Using provided baseline.py  
**Features**: first_speech, longest_pause, speech_ratio  
**Method**: Hand-tuned heuristic weights  

**Result**: FAIL - Cannot meet 5% FHR constraint
- False hangup rate: 41.28% (budget 5%)
- Machine miss rate: 53.06%  
- Score: N/A (no valid threshold)

**Analysis**: Baseline uses first_speech_duration which has only 1.9% discrimination. Wrong feature!

---

## Run 01 - Improved Features + Logistic Regression
**Date**: First model attempt  
**Change**: 
- Implemented improved VAD (handles packet loss)
- Added TIER 1 features from dataset analysis:
  - Longest pause (55% discrimination)
  - Voicing rate (53% discrimination)
  - Tone score (38% discrimination)
  - Pitch variability (31% discrimination)
  - Speech density (16% discrimination)
- 20 total features
- Logistic Regression with balanced class weights

**Training**: 
- 5083 samples (3956 human, 1127 machine)
- ROC-AUC: 0.9584 ± 0.0190 (5-fold CV)

**Result**: SUCCESS ✅
- **Score: 2020 ms**
- False hangup rate: 4.65% (within budget!)
- Machine miss rate: 18.37%
- Operating threshold: 0.980

**By Subtype**:
- Voicemail: 1568 ms (9.1% miss) ✅  
- Screener: 6000 ms (100% miss) ❌ 
- Human: 4.7% false hangup ✅

**Analysis**: 
- Voicemails detected well (strong signal)
- Screeners missed completely (need investigation)
- Overall within budget and much better than baseline

**Decision**: KEEP this approach. Next: investigate screener failures.

---

## Dataset Analysis
**Date**: Exploration phase  
**Activity**: Analyzed all 221 calls  
**Key Findings**:
- Longest pause: 55% difference (human 0.53s vs machine 0.24s) ⭐
- Voicing rate: 53% difference (human 0.27 vs machine 0.56) ⭐
- Tone score: 38% difference ⭐
- Pitch variance: 31% difference ⭐
- First speech duration: Only 1.9% difference ❌ (baseline uses this!)

**Decision**: Focus on pause patterns, voicing continuity, prosody, tone detection

---

## Run 02 - Probability Smoothing (EMA) + Dynamic Threshold
**Date**: Latency optimization  
**Change**:
- Added Exponential Moving Average (EMA) smoothing to raw probabilities
  - Alpha = 0.4 (40% raw weight, 60% prior weight)
  - Enforces monotonic non-decay for high probabilities (p > 0.7)
  - Rationale: Machines ramble continuously (prob should stay high), humans create pauses (prob dips)
- Removed hardcoded 0.980 threshold
- Let score.py find optimal threshold for smoothed probabilities

**Model**: Same 20 features + balanced logistic regression (unchanged from Run 01)

**Result**: MAJOR IMPROVEMENT ✅✅
- **Handout Score: 1689 ms** (was 2020 ms) → **16.4% faster** 🎯
- False hangup rate: 4.65% (maintained)
- Machine miss rate: 2.04% (improved!)
- Operating threshold: 0.750 (down from 0.980 - more aggressive)

**By Subtype (Handout)**:
- Voicemail: 1330 ms (0% miss) ✅✅ (improved from 1568ms)
- Screener: 4850 ms (20% miss) ⚠️ (down from 100%)
- Human: 4.7% false hangup ✅

**Generalization (Selfcheck - Hidden Test)**:
- **Hidden Score: 2125 ms** (excellent)
- Machine miss rate: **0% (was 100%!)** ✅✅✅ 
- False hangup rate: 0% (better than budget)
- Operating threshold: 0.905

**By Subtype (Hidden)**:
- Voicemail: 2125 ms (0% miss) ✅✅
- Human: 0% false hangup ✅

**Analysis**:
- EMA smoothing massively improved generalization (0% hidden miss!)
- Lower threshold better balances early detection vs false alarms
- Machines' continuous speech pattern now better captured by temporal smoothing
- Screeners still problematic (20% miss) but overall score excellent

**Key Insight**: The problem wasn't the features - it was the probability calibration! 
Machines talk continuously so probabilities should stay elevated. EMA + monotonic enforcement captured this perfectly.

**Decision**: EXCELLENT. This approach ready for submission. Optional: screener-specific detection could improve to 0% miss.

---

