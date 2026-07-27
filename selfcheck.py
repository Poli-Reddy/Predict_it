#!/usr/bin/env python3
"""
Run this before you submit.

It executes YOUR predict.py against eight real calls that are NOT in your
handout, checks the output is something our grader can read, and prints your
score on those eight. If this fails on your machine it will fail on ours.

    python selfcheck.py

Eight calls is not a score you should trust -- it is a smoke test. Its job is to
prove your code runs on a folder it has never seen and emits a readable
predictions.csv. Do not tune against it.
"""
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "amd_data", "selfcheck")
OUT = os.path.join(HERE, "_selfcheck_predictions.csv")
TIME_LIMIT_S = 300          # generous: this is only 8 calls

CHECKPOINTS = list(range(500, 6001, 250))
FAIL = "FAIL"
OK = "ok  "


def fail(msg):
    print(f"\n{FAIL}  {msg}")
    print("\nselfcheck did not pass. Fix this before submitting.")
    sys.exit(1)


def main():
    print("selfcheck -- 8 real calls your model has not been trained on\n")

    if not os.path.isdir(SAMPLE_DIR):
        fail(f"missing sample data at {SAMPLE_DIR}. Re-download the kit.")

    predict = None
    for cand in ("predict.py", os.path.join("starter", "predict.py")):
        if os.path.exists(os.path.join(HERE, cand)):
            predict = cand
            break
    if predict is None:
        fail("no predict.py found in the kit root. It must be named predict.py "
             "and accept --data_dir and --out.")
    print(f"{OK}found {predict}")

    if os.path.exists(OUT):
        os.remove(OUT)

    cmd = [sys.executable, predict, "--data_dir", SAMPLE_DIR, "--out", OUT]
    print(f"{OK}running: {' '.join(cmd[1:])}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              timeout=TIME_LIMIT_S)
    except subprocess.TimeoutExpired:
        fail(f"predict.py did not finish within {TIME_LIMIT_S}s on 8 calls. "
             "It will time out on the full hidden set.")
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(proc.stdout[-1500:])
        print(proc.stderr[-2500:], file=sys.stderr)
        fail(f"predict.py exited with code {proc.returncode}.")
    print(f"{OK}completed in {elapsed:.1f}s")

    if not os.path.exists(OUT):
        fail(f"predict.py did not write {os.path.basename(OUT)} at the --out path.")

    with open(OUT, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        need = {"call_id", "checkpoint_ms", "p_machine"}
        if not need.issubset(cols):
            fail(f"predictions must have columns {sorted(need)}; got {sorted(cols)}")
        rows = list(reader)
    if not rows:
        fail("predictions file is empty.")
    print(f"{OK}schema fine, {len(rows)} rows")

    with open(os.path.join(SAMPLE_DIR, "labels.csv"), newline="") as fh:
        expected = {r["call_id"].strip() for r in csv.DictReader(fh)}

    got, bad_p = {}, 0
    for r in rows:
        cid = (r["call_id"] or "").strip()
        try:
            p = float(r["p_machine"])
            ms = float(r["checkpoint_ms"])
        except (TypeError, ValueError):
            fail("checkpoint_ms and p_machine must both be numeric on every row.")
        if not (0.0 <= p <= 1.0):
            bad_p += 1
        got.setdefault(cid, set()).add(ms)
    if bad_p:
        fail(f"{bad_p} row(s) have p_machine outside [0,1].")
    print(f"{OK}all probabilities within [0,1]")

    missing = expected - set(got)
    if missing:
        fail(f"no predictions for {len(missing)} call(s): {sorted(missing)}")
    thin = {c: sorted(v)[:3] for c, v in got.items() if len(v) < len(CHECKPOINTS)}
    if thin:
        print(f"      note: {len(thin)} call(s) have fewer than {len(CHECKPOINTS)} checkpoints. "
              "Missing checkpoints simply never fire, which costs you latency.")
    print(f"{OK}all {len(expected)} calls covered")

    print("\n--- score on these 8 calls (do not tune to it) ---")
    sc = subprocess.run([sys.executable, os.path.join(HERE, "score.py"),
                         "--data_dir", SAMPLE_DIR, "--pred", OUT, "--verbose"],
                        capture_output=True, text=True)
    print(sc.stdout.strip() or sc.stderr.strip())

    print("\nselfcheck passed. Your submission should run on our machine.")


if __name__ == "__main__":
    main()
