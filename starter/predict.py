#!/usr/bin/env python3
"""
Skeleton for your submission. Copy this to the kit root as predict.py and make
it yours.

    python predict.py --data_dir amd_data/handout --out predictions.csv

It already runs and already produces a valid file -- it just is not a model yet.
`extract_features` returns four obvious numbers and `Model` is a constant. The
TODOs are your two hours.

Two things the grader enforces, so build with them in mind:

1. Causality. The row for checkpoint T may only depend on audio in [0, T].
   `checkpoint_slices` hands you exactly that; if you compute anything from the
   whole file and reuse it across checkpoints, you have broken the rule and we
   will see it when we read the code.

2. It must run unmodified on a directory it has never seen, with the same
   layout and a labels.csv whose `label` column you must not read. Train
   separately, save your model to disk, and load it here.
"""
import argparse
import csv
import os
import sys

import numpy as np

# You are told to copy this file to the kit root, which puts it one level above
# features.py. Look in both places so it imports either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "starter"), os.path.join(_HERE, os.pardir)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from features import (  # noqa: E402
    SR, CHECKPOINTS_MS, load_wav, checkpoint_slices,
    turn_shape, spectral_stats, tone_score, pitch_track,
)

MODEL_PATH = os.path.join(_HERE, "model.npz")


class _LinearClf:
    """Minimal predict_proba for weights saved with numpy.savez (no pickle)."""

    def __init__(self, w, b):
        self.w, self.b = np.asarray(w, dtype="float64"), b

    def predict_proba(self, X):
        z = np.asarray(X, dtype="float64") @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def extract_features(slice_audio, elapsed_s):
    """Features for ONE checkpoint, from audio in [0, elapsed_s] only.

    TODO: this is where the work is. Some directions that pay off:
      * turn structure -- how long was the first utterance, is anyone waiting
      * a VAD that survives packet loss and comfort noise (the one in
        features.py does not; see its docstring)
      * pitch behaviour -- a question rises, a recorded greeting does not
      * tones -- beeps and ringback are narrowband and steady
      * how the numbers above CHANGE as the call progresses, not just their
        value at this instant
    """
    sh = turn_shape(slice_audio, SR)
    sp = spectral_stats(slice_audio, SR)
    return np.array([
        sh["first_speech_s"],
        sh["longest_pause_s"],
        sh["speech_ratio"],
        elapsed_s,
        # TODO add your own. Keep the order identical between train and predict.
    ], dtype="float32")


class Model:
    """Wrap whatever you train. Must expose p_machine(features) -> float.

    Persistence: save plain arrays with `numpy.savez`, not pickle. Loading a
    pickle executes whatever code is inside it, and your submission is graded in
    a sandbox on our machine alongside thirty-nine others -- so we do not unpickle
    anything. A logistic regression is just weights and an intercept; if you train
    something heavier, store its parameters as arrays and do the forward pass
    yourself, as `_LinearClf` does below.
    """

    def __init__(self, clf=None):
        self.clf = clf

    @classmethod
    def load(cls, path=MODEL_PATH):
        if os.path.exists(path):
            d = np.load(path)          # plain arrays, no code execution on load
            return cls(_LinearClf(d["w"], float(d["b"])))
        return cls(None)               # falls back to the constant below

    def p_machine(self, feats):
        if self.clf is None:
            # TODO: replace. 0.5 everywhere means the scorer can never find a
            # useful threshold, so this scores as badly as it possibly could.
            return 0.5
        return float(self.clf.predict_proba(feats.reshape(1, -1))[0][1])


def call_list(data_dir):
    """Every call_id in labels.csv, with its audio path. The `label` column is
    not read here and must not be used by your model."""
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
            # Emit a full column anyway: a missing call must not break the run.
            rows += [(call_id, ms, 0.5) for ms in CHECKPOINTS_MS]
            continue
        y = load_wav(path, SR)
        for ms, sl in checkpoint_slices(y, SR):
            p = model.p_machine(extract_features(sl, ms / 1000.0))
            rows.append((call_id, ms, round(min(max(p, 0.0), 1.0), 5)))
        if n % 50 == 0:
            print(f"  ...{n}/{len(calls)} calls", file=sys.stderr, flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["call_id", "checkpoint_ms", "p_machine"])
        w.writerows(rows)
    print(f"wrote {args.out}  ({len(calls)} calls, {len(rows)} rows)")


if __name__ == "__main__":
    main()
