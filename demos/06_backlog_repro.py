#!/usr/bin/env python
"""demos/06_backlog_repro.py — Reproduziert Live-Duplikate durch Stau.

Der Offline-Benchmark (05) hatte "keine Duplikate durchs Stitching"
gemeldet — aber er reproduziert NICHT den Live-Rückstand: Offline ist
win_end immer die exakte Audiozeit, live wurde die WANDUHRZEIT benutzt
(time.monotonic() - stream_t0). Bei Rückstand (Whisper langsamer als der
Takt) läuft die Wanduhr dem Audioinhalt des Rings davon; schwankt der
Rückstand, bekommt dasselbe Wort von verschiedenen Fenstern VERSCHIEDENE
Slot-Labels → beide Slots emittieren es → ganze Sätze doppelt. Genau
"wenn es sich staut".

Diese Demo füttert PyramidStitcher mit IDENTISCHEN Fensterinhalten:
  A) win_end = Audiozeit + schwankender Rückstand  (alter Live-Zustand)
  B) win_end = exakte Audiozeit                    (Fix / Benchmark-Welt)
Erwartung: A dupliziert Wörter, B nicht.

Aufruf: .venv/bin/python demos/06_backlog_repro.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyramid_stitch import PyramidStitcher  # noqa: E402

UPDATE, CONTEXT = 3.0, 24.0
STREAM_END = 120.0


def make_words() -> list[tuple[float, float]]:
    """Synthetische Wörter zu exakten Audiozeiten (Mitte, Ende)."""
    out = []
    t = 0.3
    i = 0
    while t < STREAM_END - 1.0:
        out.append((t, t + 0.3))
        t += 0.7
        i += 1
    return out


def run(lag_fn) -> tuple[str, int]:
    st = PyramidStitcher(context_s=CONTEXT, update_s=UPDATE)
    words = make_words()
    w = 0
    end = UPDATE
    while end <= STREAM_END:
        w += 1
        lag = lag_fn(w)
        win_start = end - CONTEXT
        vis = [(f"w{i:03d}", s - win_start, e - win_start)
               for i, (s, e) in enumerate(words)
               if win_start <= (s + e) / 2 < end]
        st.add_window(vis, win_end=end + lag, win_len=CONTEXT)
        end += UPDATE
    st.flush()
    text = st.committed_text
    dups = sum(c - 1 for c in Counter(text.split()).values() if c > 1)
    return text, dups


if __name__ == "__main__":
    cases = [
        ("schwankend 3.5/5.5 s", lambda w: 3.5 if w % 3 else 5.5),
        ("stark schwankend 2/10 s", lambda w: 2.0 if w % 2 else 10.0),
        ("stark schwankend 2/13 s", lambda w: 2.0 if w % 2 else 13.0),
        ("gleichmäßig 5 s (kein Schwanken)", lambda w: 5.0),
    ]
    text_fix, dups_fix = run(lambda w: 0.0)
    print(f"Referenz (exakte Audiozeit): {dups_fix} doppelte Wörter\n")
    failed = False
    for label, fn in cases:
        text_bug, dups_bug = run(fn)
        mark = "← DUP!" if dups_bug > dups_fix else "ok"
        if dups_bug > dups_fix:
            failed = True
        print(f"A) {label:34s}: {dups_bug:3d} doppelte Wörter {mark}")
    if failed:
        print("\n→ Bestätigt: schwankender Stau + Wanduhr-win_end dupliziert;")
        print("  der Live-Code MUSS win_end aus der Sample-Zahl des Chunks")
        print("  ableiten, nie aus der Wanduhr.")
        raise SystemExit(0)
    print("\n→ KEINE Duplikate reproduziert — Hypothese prüfen!")
    raise SystemExit(1)
