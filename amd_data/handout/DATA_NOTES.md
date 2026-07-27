# About this data

221 real call openings from Zetta's outbound dialer, one per file, 8 kHz mono,
8 seconds each starting the moment the call connected.

| | |
|---|---|
| calls | 221 |
| machine | 49 |
| human | 172 |

`labels.csv` has a `subtype` column -- `voicemail`, `screener`, `human`. It is
metadata to help you look at your errors. **You will not get it at test time**,
so do not build it into your model.

**Not every machine speaks.** Some answered calls are a beep, a tone, hold music,
or near-silence with no greeting at all. There are none in this folder and there
are some in the hidden set, so a model that can only spot a machine by hearing it
talk will miss them outright -- and a miss costs the full 6000 ms.

## Things worth knowing

**This is the far end only.** Our own side of the call has been removed, so what
you hear is whoever or whatever answered. That is what a dialer's detector gets.

**The audio is genuinely damaged in places.** Narrowband, mu-law, packet loss,
silence suppression, AGC pumping. Some clips are quiet enough that a naive
energy threshold reports the whole thing as speech. That is not a bug in the
data, it is the problem.

**Every clip is exactly 8.00 seconds.** Length tells you nothing -- we padded it
out on purpose so it cannot.

**The call ids are random.** They carry no information about the label, the
source, or the order the calls were made in.

## What is NOT in here

The hidden test set. It contains answering systems whose voices do not appear in
this folder at all -- different carriers, different greetings. A model that
learns to recognise these particular recordings will score badly on it. That is
the point of the exercise.
