# Research Observations - AMD Dataset Analysis

Generated from: `analysis/explore_dataset.py`  
Date: Dataset exploration phase  
Samples: 221 calls (172 human, 49 machine)

## **Critical Discoveries**

### 🔥 **Observation 1: LONGEST PAUSE is the strongest discriminator**
**Evidence**: 
- Human longest pause: 0.533s ± 0.889s
- Machine longest pause: 0.240s ± 0.380s
- **Difference: 55.0%** (highest of all features!)

**Insight**: Machines don't wait for responses. Humans pause expecting replies.

**Confidence**: ⭐⭐⭐⭐⭐ VERY HIGH  
**Priority**: TIER 1 - Implement immediately

---

### 🔥 **Observation 2: PITCH VARIABILITY distinguishes prosody**
**Evidence**:
- Human pitch std: 77.757 Hz ± 14.338
- Machine pitch std: 53.537 Hz ± 13.565
- **Difference: 31.1%**

**Insight**: Humans vary pitch naturally (questions, emphasis). Machines read monotonously.

**Confidence**: ⭐⭐⭐⭐ HIGH  
**Priority**: TIER 1

---

### 🔥 **Observation 3: TONE SCORE reveals recorded greetings**
**Evidence**:
- Human tone score: 0.055 ± 0.035
- Machine tone score: 0.089 ± 0.020
- **Difference: 37.9%**

**Insight**: Voicemail beeps and IVR tones are narrowband. Live speech is broadband.

**Confidence**: ⭐⭐⭐⭐ HIGH  
**Priority**: TIER 1

---

### ⚡ **Observation 4: SPEECH DENSITY separates classes**
**Evidence**:
- Human speech ratio: 0.680 ± 0.291
- Machine speech ratio: 0.812 ± 0.128  
- **Difference: 16.3%**

**Insight**: Machines talk wall-to-wall. Humans leave gaps.

**Note**: Lower variance in machines (0.128 vs 0.291) = more consistent behavior.

**Confidence**: ⭐⭐⭐⭐ HIGH  
**Priority**: TIER 1

---

### ⚡ **Observation 5: VOICING RATE differs dramatically**
**Evidence**:
- Human voicing rate: 0.265 ± 0.165
- Machine voicing rate: 0.560 ± 0.093
- **Difference: 52.6%** (huge!)

**Insight**: Machines sustain voicing continuously. Humans have more unvoiced frames (breath, consonants).

**Confidence**: ⭐⭐⭐⭐⭐ VERY HIGH  
**Priority**: TIER 1

---

### ⚡ **Observation 6: RMS ENERGY higher in machines**
**Evidence**:
- Human RMS: -63.4 dBFS ± 7.4
- Machine RMS: -42.7 dBFS ± 6.2
- **Difference: ~20 dB** (massive!)

**Insight**: Recorded greetings have consistent, higher energy. Live humans vary more.

**Warning**: This could be AGC/recording artifact. Must validate it generalizes.

**Confidence**: ⭐⭐⭐ MEDIUM (generalization concern)  
**Priority**: TIER 2 - Use with caution

---

### 📊 **Observation 7: Onset time shows small difference**
**Evidence**:
- Human onset: 0.922s ± 0.457
- Machine onset: 0.804s ± 0.280
- **Difference: 12.7%**

**Insight**: Machines start slightly faster, but overlap is high.

**Confidence**: ⭐⭐ LOW  
**Priority**: TIER 3 - Weak alone, useful in combination

---

### 📊 **Observation 8: SPECTRAL FLATNESS varies**
**Evidence**:
- Human: 0.411 ± 0.195
- Machine: 0.354 ± 0.054

**Insight**: Machines have more tonal (less flat) spectra. Consistent with recorded audio.

**Note**: Lower variance in machines again (0.054 vs 0.195).

**Confidence**: ⭐⭐⭐ MEDIUM  
**Priority**: TIER 2

---

### 📊 **Observation 9: First speech duration is NOT discriminative**
**Evidence**:
- Human: 1.991s
- Machine: 2.029s
- **Difference: 1.9%** (negligible!)

**Insight**: Baseline heuristic uses this - **it's weak!**

**Confidence**: ⭐⭐⭐⭐ HIGH (that it doesn't work)  
**Priority**: REJECT this feature

---

### 🚨 **Observation 10: Number of segments similar**
**Evidence**:
- Human: 5.8 segments ± 7.3
- Machine: 5.4 segments ± 5.2

**Insight**: Segment count alone doesn't discriminate. But PAUSE DISTRIBUTION does!

**Confidence**: ⭐⭐⭐ MEDIUM  
**Priority**: Use pause statistics, not just count

---

## **Feature Ranking**

### **TIER 1 - Implement First** (Strong discriminators)
1. ⭐ Longest pause (55% diff)
2. ⭐ Voicing rate (53% diff)  
3. ⭐ Tone score (38% diff)
4. ⭐ Pitch variability (31% diff)
5. ⭐ Speech density (16% diff)

### **TIER 2 - Secondary Features**
6. Spectral flatness (consistency)
7. RMS energy (with generalization caution)
8. Pause variance (not just longest)
9. Energy progression over time

### **TIER 3 - Weak/Combined Use**
10. Onset time (weak alone)
11. First speech duration (REJECT - doesn't work)
12. Segment count (use distribution instead)

---

## **Critical Insights for Feature Engineering**

### ✅ **What Works**
- **Temporal patterns**: Pauses, speech continuity
- **Prosody**: Pitch variation, voicing consistency
- **Spectral**: Tone detection, flatness
- **Energy dynamics**: Voicing rate, sustained speech

### ❌ **What Doesn't Work** (Baseline Failures)
- First utterance duration (baseline uses this - wrong!)
- Simple onset time
- Raw segment count

### 🎯 **Next Steps**
1. Build robust VAD (current one fails on packet loss)
2. Extract temporal evolution features (not just point values)
3. Add pause distribution statistics
4. Implement prosody features (pitch trends)
5. Detect AGC robustness (energy normalization)

---

## **Hidden Test Risks**

### ⚠️ **Generalization Concerns**
1. **RMS energy difference** may not generalize (different recording setups)
2. **Pitch mean** varies by speaker (focus on variance, not absolute)
3. **Onset time** depends on network latency

### ✅ **Expected to Generalize**
1. Longest pause (behavioral, not acoustic)
2. Voicing rate (continuous vs. interrupted)
3. Tone score (beeps are beeps everywhere)
4. Pitch variance (prosody is universal)
5. Speech density (machines talk continuously)

---

## **Hypothesis Generation**

Based on observations, I predict:

**H1**: Combining longest_pause + voicing_rate will achieve >80% separation  
**H2**: Tone score will catch silent machines (beeps) that baseline misses  
**H3**: Pitch variance will reduce false hangups (questions have rising pitch)  
**H4**: Speech density evolution (not just final value) will detect screeners  
**H5**: Improved VAD will fix packet-loss-as-pause errors  

These will be tested systematically in experiments.
