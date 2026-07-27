#!/usr/bin/env python3
"""
The given baseline. Beat it.

Rule of thumb it encodes: a machine talks for a long time without waiting for
you; a human says something short and then waits. So at each checkpoint, look at
how much continuous speech there has been and how long the longest pause is, and
turn that into a probability with a hand-tuned squashing function.

    python starter/baseline.py --data_dir amd_data/handout --out predictions.csv
    python score.py --data_dir amd_data/handout --pred predictions.csv

It is honest about being a heuristic. It has no idea what a packet-loss gap is,
it cannot tell a damaged line from a quiet one, and it has never heard a call screener.
Your first job is to find out where it breaks -- listen to the calls it gets
wrong before you write any features.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import SR, CHECKPOINTS_MS, load_wav, checkpoint_slices, turn_shape  # noqa: E402


def p_machine_from_shape(sh, elapsed_s):
    """Hand-tuned heuristic -> pseudo-probability in [0, 1]."""
    # Long unbroken first utterance looks like a greeting being read out.
    long_talk = min(sh["first_speech_s"] / 3.0, 1.0)
    # A long pause looks like someone waiting for a reply, i.e. a human.
    waiting = min(sh["longest_pause_s"] / 1.2, 1.0)
    # Wall-to-wall speech also looks machine-like.
    dense = min(sh["speech_ratio"] / 0.9, 1.0)

    p = 0.55 * long_talk + 0.30 * dense - 0.45 * waiting
    # Early on we have almost no evidence, so stay near the fence.
    confidence = min(elapsed_s / 2.5, 1.0)
    p = 0.5 + (p - 0.0) * confidence
    return float(min(max(p, 0.0), 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    labels_path = os.path.join(args.data_dir, "labels.csv")
    with open(labels_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # One row per call, so we do not depend on labels.csv listing checkpoints.
    seen = {}
    for r in rows:
        seen.setdefault(r["call_id"].strip(), r.get("audio_file", "").strip())

    out_rows = []
    for i, (call_id, rel) in enumerate(sorted(seen.items()), 1):
        path = os.path.join(args.data_dir, rel) if rel else None
        if not path or not os.path.exists(path):
            # Never skip a call: emit a neutral column instead.
            for ms in CHECKPOINTS_MS:
                out_rows.append((call_id, ms, 0.5))
            continue
        y = load_wav(path, SR)
        for ms, sl in checkpoint_slices(y, SR):
            sh = turn_shape(sl, SR)
            out_rows.append((call_id, ms, round(p_machine_from_shape(sh, ms / 1000.0), 4)))
        if i % 25 == 0:
            print(f"  ...{i} calls", file=sys.stderr)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["call_id", "checkpoint_ms", "p_machine"])
        w.writerows(out_rows)
    print(f"wrote {args.out}  ({len(seen)} calls, {len(out_rows)} rows)")


if __name__ == "__main__":
    main()
