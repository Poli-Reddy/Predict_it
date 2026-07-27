# AMD Assignment - Notes

## Current Status
- Handout evaluation: 1689 ms mean latency at 4.65% false-hangup rate and 2.04% machine miss rate (threshold 0.750).
- Hidden selfcheck evaluation: 2125 ms mean latency at 0.00% false-hangup rate and 0.00% machine miss rate (threshold 0.905).
- The current pipeline uses a 20-feature logistic-regression model with EMA temporal smoothing to stabilize probability estimates.

## What Worked
- Temporal smoothing improved both handout latency and hidden-test generalization.
- Behavioral features such as longest pause, voicing continuity, and prosody were more informative than raw energy cues.
- The approach stayed causal and used only the provided data and allowed libraries.

## Remaining Limitations
- Screeners remain the hardest subtype; the current model still misses a minority of them.
- More machine examples would likely improve robustness across unseen carriers and answering systems.
