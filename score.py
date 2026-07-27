#!/usr/bin/env python3
"""
Scorer for the Zetta AMD (answering-machine detection) assignment.

Simulates a live dialer: it walks your per-checkpoint probabilities in time
order and hangs up the moment p_machine crosses a threshold. The threshold is
tuned for you -- we pick the operating point that is FASTEST while keeping the
false-hangup rate (hanging up on a real human) at or below the allowed budget.

    python score.py --data_dir amd_data/handout --pred predictions.csv

Headline metric:  mean detection latency (ms) on machine calls
                  at a false-hangup rate <= 5%.  Lower is better.

A machine you never detect is counted at the DEADLINE, so misses cost you
latency. There is no way to look good by simply never firing.
"""
import argparse
import csv
import sys
from collections import defaultdict

DEADLINE_MS = 6000     # after this the dialer has already handed the call to an agent
MAX_FALSE_HANGUP = 0.05


def load_labels(data_dir):
    path = f"{data_dir}/labels.csv"
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except FileNotFoundError:
        sys.exit(f"error: no labels.csv in {data_dir}")
    if not rows:
        sys.exit(f"error: {path} is empty")

    labels = {}
    for r in rows:
        lab = (r.get("label") or "").strip().lower()
        if lab not in ("human", "machine"):
            sys.exit(f"error: bad label {lab!r} for call {r.get('call_id')!r} "
                     "(expected 'human' or 'machine')")
        labels[r["call_id"].strip()] = {
            "label": lab,
            # subtype is optional metadata (voicemail / screener / silent / human)
            "subtype": (r.get("subtype") or "").strip().lower(),
        }
    return labels


def load_predictions(path, known_ids):
    """predictions.csv: call_id,checkpoint_ms,p_machine"""
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"call_id", "checkpoint_ms", "p_machine"} - set(reader.fieldnames or [])
            if missing:
                sys.exit(f"error: {path} is missing column(s): {', '.join(sorted(missing))}")
            preds = defaultdict(list)
            for i, r in enumerate(reader, start=2):
                cid = (r["call_id"] or "").strip()
                if cid not in known_ids:
                    continue  # ignore rows for calls we are not scoring
                try:
                    ms = float(r["checkpoint_ms"])
                    p = float(r["p_machine"])
                except (TypeError, ValueError):
                    sys.exit(f"error: {path} line {i}: checkpoint_ms and p_machine must be numeric")
                if not (0.0 <= p <= 1.0):
                    sys.exit(f"error: {path} line {i}: p_machine={p} outside [0,1]")
                # A checkpoint is a moment in the call, so it cannot be negative.
                # Without this the metric is trivially breakable: fire at
                # -100000 ms and the mean latency goes as low as you like.
                if not (0.0 <= ms <= DEADLINE_MS):
                    sys.exit(f"error: {path} line {i}: checkpoint_ms={ms:g} outside "
                             f"[0, {DEADLINE_MS}]. Checkpoints are moments in the call.")
                preds[cid].append((ms, p))
    except FileNotFoundError:
        sys.exit(f"error: predictions file not found: {path}")

    for cid in preds:
        preds[cid].sort(key=lambda t: t[0])   # time order matters: we replay it live
    return preds


def decide(checkpoints, threshold):
    """Earliest checkpoint whose probability crosses the threshold.

    Returns the firing time in ms, or None if the call is never flagged.
    """
    for ms, p in checkpoints:
        if p >= threshold:
            return ms
    return None


def evaluate(labels, preds, threshold):
    humans = machines = 0
    false_hangups = 0
    latencies = []
    misses = 0

    for cid, meta in labels.items():
        fire_at = decide(preds.get(cid, []), threshold)
        if meta["label"] == "human":
            humans += 1
            if fire_at is not None:
                false_hangups += 1          # we hung up on a real customer
        else:
            machines += 1
            if fire_at is None:
                misses += 1
                latencies.append(DEADLINE_MS)   # never caught: full penalty
            else:
                latencies.append(min(fire_at, DEADLINE_MS))

    return {
        "threshold": threshold,
        "false_hangup_rate": (false_hangups / humans) if humans else 0.0,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else float("inf"),
        "machine_miss_rate": (misses / machines) if machines else 0.0,
        "n_human": humans,
        "n_machine": machines,
    }


def best_operating_point(labels, preds):
    """Fastest threshold that still respects the false-hangup budget."""
    grid = [i / 200.0 for i in range(201)]        # 0.000 .. 1.000 step 0.005
    feasible = []
    for t in grid:
        res = evaluate(labels, preds, t)
        if res["false_hangup_rate"] <= MAX_FALSE_HANGUP:
            feasible.append(res)
    if not feasible:
        # Cannot satisfy the budget at any threshold -- report the least-bad.
        allres = [evaluate(labels, preds, t) for t in grid]
        return min(allres, key=lambda r: r["false_hangup_rate"]), False
    return min(feasible, key=lambda r: r["mean_latency_ms"]), True


def by_subtype(labels, preds, threshold):
    groups = defaultdict(dict)
    for cid, meta in labels.items():
        st = meta["subtype"] or "unspecified"
        groups[st][cid] = meta
    out = {}
    for st, sub in sorted(groups.items()):
        out[st] = evaluate(sub, preds, threshold)
    return out


def main():
    ap = argparse.ArgumentParser(description="Score AMD predictions like a live dialer.")
    ap.add_argument("--data_dir", required=True, help="folder containing labels.csv")
    ap.add_argument("--pred", required=True, help="your predictions.csv")
    ap.add_argument("--verbose", action="store_true", help="break the score down by subtype")
    args = ap.parse_args()

    labels = load_labels(args.data_dir)
    preds = load_predictions(args.pred, set(labels))

    scored = sum(1 for cid in labels if preds.get(cid))
    if scored == 0:
        sys.exit("error: predictions.csv has no rows matching any call_id in labels.csv")

    res, feasible = best_operating_point(labels, preds)

    print(f"calls scored            {scored} of {len(labels)}"
          f"   ({res['n_human']} human, {res['n_machine']} machine)")
    if scored < len(labels):
        print(f"  note: {len(labels)-scored} call(s) had no predictions and were treated as never-fired")
    print(f"operating threshold     {res['threshold']:.3f}")
    print(f"false-hangup rate       {res['false_hangup_rate']*100:.2f}%   (budget {MAX_FALSE_HANGUP*100:.0f}%)")
    print(f"machine miss rate       {res['machine_miss_rate']*100:.2f}%")
    print()
    if feasible:
        print(f"SCORE  mean detection latency  {res['mean_latency_ms']:.0f} ms   (lower is better)")
    else:
        print("SCORE  n/a -- no threshold keeps the false-hangup rate within budget.")
        print("       Your probabilities do not separate humans from machines yet.")

    if args.verbose:
        print("\nby subtype (at the same threshold):")
        for st, r in by_subtype(labels, preds, res["threshold"]).items():
            lat = f"{r['mean_latency_ms']:>6.0f} ms" if r["n_machine"] else "     n/a"
            print(f"  {st:<14} latency {lat}   "
                  f"false-hangup {r['false_hangup_rate']*100:>5.1f}%   "
                  f"miss {r['machine_miss_rate']*100:>5.1f}%   "
                  f"(n={r['n_human']+r['n_machine']}, {r['n_machine']} machine)")


if __name__ == "__main__":
    main()
