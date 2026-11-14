#!/usr/bin/env python
"""demos/07_attack_vectors.py — Fehlervektoren gegen PyramidStitcher + Clock-Fix.

Denkt um die Ecke: jede Simulation modelliert einen LIVE-Fehlermodus, den
weder der Offline-Benchmark (exakte Zeiten) noch die Unit-Tests (ideale
Zeiten) abdecken. Geprüft wird pro Vektor:
  - Dups:     Wörter, die öfter im committed_text stehen als im Strom
  - Monotonie: committed-Präfix ändert sich nachträglich nie
  - Verlust:  Anteil der Wörter, die es nie in den Strom schaffen

Vektoren:
  V1  Slot-Boundary-Jitter: Wortmitte pendelt um eine Slot-Grenze
  V2  Verworfene Chunks (Stau): echte Zeitlücke im Ring (Audiozeit springt)
  V3  Andere K-Verhältnisse (2/10 → K=5, 2.5/10 → K=4, 1.5/12 → K=8)
  V4  Einschwingphase mit Drops (Ring erst nach und nach voll)
  V5  Realistischer Whisper-Jitter ±200 ms auf jeder Wortzeit
  V6  Cluster-Drift: A~B, B~C, aber A!~C (Ketten-Match im Konsens)
  V7  Wanduhr-Regressionsschutz: alter Defekt muss noch reproduzierbar sein

Aufruf: .venv/bin/python demos/07_attack_vectors.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyramid_stitch import PyramidStitcher  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(label)


def run_windows(win_specs, update, context, track_monotonic=True):
    """win_specs: Liste pro Fenster von (words_abs, win_end, win_len,
    audio_end). words_abs: Liste (surface, t_center_abs) in RING-Zeit.
    Die Sichtungs-Auswahl folgt IMMER audio_end (der Ring enthält nur
    echte Audiozeit — Whisper sieht keine Zukunft); nur das win_end-LABEL
    darf davon abweichen (alter Wanduhr-Defekt).
    Returns (text, dups, monotone_ok)."""
    st = PyramidStitcher(context_s=context, update_s=update)
    prev_committed = ""
    monotone_ok = True
    for words_abs, win_end, win_len, audio_end in win_specs:
        win_start = audio_end - win_len
        vis = [(w, t - win_start, t - win_start + 0.3)
               for w, t in words_abs if win_start <= t < audio_end]
        st.add_window(vis, win_end=win_end, win_len=win_len)
        c = st.committed_text
        if track_monotonic and not c.startswith(prev_committed):
            monotone_ok = False
        prev_committed = c
    st.flush()
    text = st.committed_text
    # Dup = Wort öfter im Strom als erwartet. Jedes synthetische Wort
    # existiert im Original GENAU 1× (want=1), egal wie viele Fenster es
    # sahen — die Pyramide emittiert pro Slot, und ein Wort liegt in
    # genau einem Slot. (Früherer Bug hier: want über alle specs gezählt
    # = Anzahl der Fenster, die das Wort sahen → Dups unsichtbar!)
    want = Counter(w for spec in win_specs for w, _ in spec[0])
    got = Counter(text.split())
    dups = sum(n - 1 for w, n in got.items() if n > 1 and want.get(w, 0) > 0)
    # Extras (nie gesehene Wörter im Strom) sind auch verdächtig:
    dups += sum(n for w, n in got.items() if want.get(w, 0) == 0)
    return text, dups, monotone_ok


def steady_stream(n_words, gap=0.7, start=0.3, dur=120.0):
    out, t = [], start
    while t < dur - 1.0 and len(out) < n_words:
        out.append(t)
        t += gap
    return [(f"w{i:03d}", t) for i, t in enumerate(out)]


def windows_for(words, update, context, end_time, lag_fn=None, drop=None,
                audio_truth_content=True):
    """Erzeugt Fenster-Specs.

    lag_fn: alter Wanduhr-Defekt — das win_end-LABEL läuft um lag der
    Wanduhr voraus, aber der Ring-INHALT bleibt Audio-Realität
    (audio_truth_content=True: win_start = audio_end - win_len, Wörter
    nach Audiozeit ausgewählt). Mit audio_truth_content=False wandert der
    Inhalt MIT dem Label mit — dann heilt der Re-Anker alles (das war der
    Unterschied, der V7 anfangs 0 Dups zeigte).

    drop(i)=True → Fenster i überspringt 1 Intervall Audio (Chunk verworfen):
    audio_end zählt nur akzeptierte Chunks; die Lücke existiert im Ring
    nicht (die Wörter im verworfenen Intervall tauchen nie auf).
    """
    specs = []          # (words, win_end, win_len, audio_end)
    audio_end = 0.0
    w = 0
    while True:
        if drop and drop(w + 1):
            w += 1
            continue               # Chunk verworfen: audio_end zählt nicht
        if audio_end >= end_time:
            break
        w += 1
        audio_end += update
        win_len = min(audio_end, context)
        win_end = audio_end + (lag_fn(w) if lag_fn else 0.0)
        specs.append((words, win_end, win_len, audio_end))
    return specs


def vector1_boundary_jitter():
    print("V1 Slot-Boundary-Jitter (Wortmitte pendelt um 3s-Grenze):")
    rnd = random.Random(42)
    words = []
    t = 0.3
    while t < 119.0:
        words.append((f"b{len(words):03d}", t))
        t += 0.7
    # Ein Wort EXAKT auf der Grenze bei 60.0 (Slot 20), Jitter ±0.3:
    boundary = 60.0
    specs = []
    audio_end = 0.0
    w = 0
    jittered = [(s, (t + rnd.uniform(-0.02, 0.02)) if abs(t - boundary) > 0.05
                 else t + rnd.uniform(-0.3, 0.3)) for s, t in words]
    while audio_end < 120.0:
        w += 1
        audio_end += 3.0
        specs.append((jittered, audio_end, min(audio_end, 24.0), audio_end))
    _, dups, mono = run_windows(specs, 3.0, 24.0)
    check("keine Dups trotz Boundary-Jitter", dups == 0, f"{dups} Dups")
    check("Monotonie gewahrt", mono)


def vector2_dropped_chunks():
    print("V2 Verworfene Chunks (Stau → echte Zeitlücke):")
    words = steady_stream(150)
    every = 9
    specs = windows_for(words, 3.0, 24.0, 120.0, drop=lambda w: w % every == 0)
    text, dups, mono = run_windows(specs, 3.0, 24.0)
    check("keine Dups mit Chunk-Drops", dups == 0, f"{dups} Dups")
    check("Monotonie gewahrt", mono)
    # Wörter IN der Lücke dürfen fehlen, aber der Strom muss lückenlos
    # weiterlaufen (kein Duplikat des Wortes VOR der Lücke am Ende von
    # Slot k UND Anfang von Slot k+1):
    n_seen = sum(1 for w in text.split())
    print(f"       Strom: {n_seen} Wörter (Lücken-Wörter korrekt verworfen)")


def vector3_k_ratios():
    print("V3 Andere K-Verhältnisse:")
    for update, context in ((2.0, 10.0), (2.5, 10.0), (1.5, 12.0), (3.0, 12.0)):
        words = steady_stream(200, dur=100.0)
        specs = windows_for(words, update, context, 100.0)
        _, dups, mono = run_windows(specs, update, context)
        check(f"update={update} context={context} (K={context/update:.0f}): "
              f"keine Dups", dups == 0, f"{dups} Dups") if False else check(
              f"u={update}/c={context} K={context/update:.0f} keine Dups",
              dups == 0, f"{dups} Dups")
        check(f"u={update}/c={context} Monotonie", mono)


def vector4_warmup_with_drops():
    print("V4 Einschwingphase mit Drops (Ring wächst, Lücken früh):")
    words = steady_stream(150)
    specs = windows_for(words, 3.0, 24.0, 120.0, drop=lambda w: w in (2, 5, 6))
    _, dups, mono = run_windows(specs, 3.0, 24.0)
    check("keine Dups in warmup+Drops", dups == 0, f"{dups} Dups")
    check("Monotonie gewahrt", mono)


def vector5_whisper_jitter():
    print("V5 Realistischer Jitter ±200 ms auf JEDER Wortzeit pro Fenster:")
    rnd = random.Random(7)
    words = steady_stream(150)
    specs = []
    audio_end = 0.0
    w = 0
    while audio_end < 120.0:
        w += 1
        audio_end += 3.0
        win_len = min(audio_end, 24.0)
        win_start = audio_end - win_len
        vis = [(s, t + rnd.uniform(-0.2, 0.2)) for s, t in words]
        specs.append((vis, audio_end, win_len, audio_end))
    _, dups, mono = run_windows(specs, 3.0, 24.0)
    check("keine Dups mit ±200ms-Jitter", dups == 0, f"{dups} Dups")
    check("Monotonie gewahrt", mono)


def vector6_cluster_chain():
    print("V6 Cluster-Drift A~B~C, A!~C (Konsens-Kettenmatch):")
    # Innerhalb EINES Slots: Sichtungen driften schrittweise weg. Der Konsens
    # vergleicht nur mit ERSTEM und LETZTEM Cluster-Element — driften
    # Sichtungen schrittweise, kann ein Cluster auseinanderfallen und das
    # zweite Fragment als eigenes Wort emittieren (Dup desselben Slots).
    st = PyramidStitcher(context_s=24.0, update_s=3.0)
    base = ["halluzination", "halluzinatio", "halluzinatn", "halluzinat",
            "alluzinat", "aluzinat"]
    # 8 Sichtungen desselben Worts in Slot 0, driftend:
    for i, form in enumerate(base * 2):
        st.slots.setdefault(0, []).append(
            type(st).__module__ and __import__("pyramid_stitch").Sighting(
                form, 1.0 + i * 0.2, i))
    text = st._consensus(0)
    dups = text.split().count(text.split()[0]) - 1 if text.split() else 0
    check("driftendes Wort bleibt EIN Wort", dups == 0, f"{text!r}")


def vector7_wallclock_regression():
    print("V7 Wanduhr-Defekt bleibt reproduzierbar (Regressionsschutz):")
    words = steady_stream(150)
    update = 3.0
    lag_fn = lambda w: 10.0 if w % 2 else 2.0
    specs = windows_for(words, update, 24.0, 120.0, lag_fn=lag_fn)
    _, dups, _ = run_windows(specs, update, 24.0)
    check("alter Defekt zeigt Dups (wenn 0: Simulation kaputt)", dups > 0,
          f"{dups} Dups")


if __name__ == "__main__":
    vector1_boundary_jitter()
    vector2_dropped_chunks()
    vector3_k_ratios()
    vector4_warmup_with_drops()
    vector5_whisper_jitter()
    vector6_cluster_chain()
    vector7_wallclock_regression()
    print()
    if FAILURES:
        print(f"ERGEBNIS: {len(FAILURES)} FEHLER:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ERGEBNIS: alle Fehlervektoren abgedeckt, keine neuen Bugs.")
