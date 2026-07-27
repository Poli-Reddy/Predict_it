# Zetta Tech — Answering-Machine Detection

**Read `Zetta_AMD_Assignment.pdf` first.** It is the assignment; this file is
just so you are not staring at a folder wondering where to start.

## Setup (do this before the window opens)

```bash
pip install -r requirements.txt
python starter/baseline.py --data_dir amd_data/handout --out predictions.csv
python score.py --data_dir amd_data/handout --pred predictions.csv
```

If those three run, you are ready. Losing twenty minutes to a broken install on
the day is a miserable way to start.

The baseline will report that it fails. That is correct — it is the thing you are
beating.

## What is here

```
Zetta_AMD_Assignment.pdf   the assignment. Read it.
amd_data/handout/          221 real call openings + labels.csv + DATA_NOTES.md
amd_data/selfcheck/        8 calls for the pre-submission smoke test
starter/features.py        audio loading, a VAD, pitch, spectral helpers
starter/baseline.py        the heuristic to beat
starter/predict.py         skeleton — copy to the root as predict.py
score.py                   the scorer. This is the metric.
selfcheck.py               run before you submit
requirements.txt           exactly the libraries allowed, pinned to our grader
```

## Before you submit

```bash
python selfcheck.py
```

Then push to a **private** GitHub repo and add **`kamal-zetta`** as a
collaborator. That invite is how we read your work — without it we cannot grade
you.
