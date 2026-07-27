# Project Understanding - Zetta AMD Assignment

## Problem Statement
Detect whether a HUMAN or MACHINE answered a phone call using only the first few seconds of audio from the answering party.

## Input
- 8-second audio clips (8 kHz mono, mu-law encoded)
- Real outbound dialer recordings with packet loss, silence suppression, AGC
- Far-end only (no local audio)

## Output
- CSV with three columns: `call_id`, `checkpoint_ms`, `p_machine`
- Probability p_machine ∈ [0, 1] at each checkpoint
- Checkpoints: 500, 750, 1000, ..., 6000 ms (every 250ms)

## Evaluation Metric
**Primary**: Mean detection latency (ms) on machine calls at false-hangup rate ≤ 5%

**Scoring Details**:
- Threshold auto-tuned to find fastest operating point within FHR budget
- Missed machines penalized at 6000ms
- Cannot game by never firing (still eat deadline penalty)

## Constraints

### Hard Rules (Disqualification if Violated)
1. **Causality**: At checkpoint T, only use audio [0, T]. NEVER future audio.
2. **Libraries**: Only numpy, scipy, scikit-learn, pandas, librosa, soundfile
3. **No Pretrained Models**: No Whisper, wav2vec, Silero, webrtcvad, HuggingFace
4. **No External Data**: Only provided kit data
5. **CPU Only**: No GPU, no cloud training
6. **Reproducibility**: Must run unmodified on unseen directories

### Grading Breakdown
- 30% - Score improvement above AI-agent baseline
- 25% - RUNLOG.md quality (must commit as you go, include failures)
- 20% - Human judgment (patterns discovered, not handed to you)
- 15% - Code quality and generalization
- 10% - NOTES.md honesty (AI usage, failure modes)

## Dataset

### Training Set
- 221 real calls (8.00s each, padded)
- 172 humans (77.8%)
- 49 machines (22.2%)
  - 43 voicemail
  - 5 screeners
  - 1 silent/other

### Known Challenges
- Severe class imbalance
- Packet loss gaps (misread as pauses by naive VAD)
- Comfort noise / DTX confuses energy thresholds
- AGC pumping causes energy drift
- Some clips extremely quiet
- NOT all machines speak (beeps, tones, silence in hidden set)

### Hidden Test Set
- Different voices, carriers, greetings
- Silent/beeping machines (not in training)
- Memorizing specific recordings = failure

## Success Criteria
1. False hangup rate ≤ 5%
2. Minimize mean detection latency on machines
3. Generalize to unseen answering systems
4. Pass selfcheck.py (8 hidden calls)
5. Clean, documented, production-quality code

## Deliverables (Required)
1. `predict.py` - inference script
2. `predictions.csv` - handout predictions
3. `RUNLOG.md` - experiment log (committed incrementally)
4. `NOTES.md` - max 10 sentences (AI usage, limitations)
5. `SUMMARY.html` - self-contained report
6. Private GitHub repo with `kamal-zetta` as collaborator

## Key Insights
- This is an **engineering competition**, not just ML
- 70% of grade is NOT the score
- Systematic experimentation > random feature engineering
- Evidence-driven features > blind AI suggestions
- Generalization > memorization
- Honest failure documentation > clean success narrative

## Time Budget (2.5 hours)
- 0:00-0:20 → Manual exploration, pattern discovery
- 0:20-0:40 → Feature engineering foundation
- 0:40-1:10 → Core features (VAD, temporal, prosody)
- 1:10-1:40 → Model training, first scoring runs
- 1:40-2:00 → Error analysis, feature iteration
- 2:00-2:15 → Final refinements
- 2:15-2:25 → Documentation
- 2:25-2:30 → Validation, submission
