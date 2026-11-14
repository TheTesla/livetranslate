#!/usr/bin/env python
"""demos/05_bench_vortrag.py — Reproduziert den Live-Fehler offline.

Der Bericht: 'Teilweise waren ganze Sätze doppelt. Vielleicht auch, wenn es
sich staut.' Diese Simulation füttert beide Stitcher mit Fenster-Hypothesen
eines echten 33-Minuten-Vortrags (vortrag.m4a → vortrag_16k.wav), genau wie
im Live-Betrieb: Fenster à context Sekunden, alle update Sekunden ein neues.
Erkennt Duplikate quantitativ (4-gram-Wiederholungen) und vergleicht WER
gegen die Voll-Audio-Referenz (transkript_whisper_largev3turbo_int8_beam5.txt).

Aufruf:
  .venv/bin/python demos/05_bench_vortrag.py [--model large-v3-turbo]
  Optionen: --update 3 --context 24 --limit-seconds N (für schnelle Läufe)
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import re
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

for _lib in ("cublas", "cudnn"):
    for _so in sorted(glob.glob(str(_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / _lib / "lib" / "lib*.so*"))):
        try:
            ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from jiwer import wer  # noqa: E402

from text_stitch import TextStitcher  # noqa: E402
from pyramid_stitch import PyramidStitcher  # noqa: E402

SAMPLE_RATE = 16000


def norm_tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s\-]", "", text.lower(), flags=re.UNICODE).split()


def duplicate_stats(text: str, n: int = 6, top: int = 8) -> list[tuple[str, int]]:
    """Häufigste n-gram-Wiederholungen (Duplikat-Indikator)."""
    toks = norm_tokens(text)
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    # 6-gram, das mehr als 1× vorkommt, ist verdächtig (außer sehr kurze):
    return [(" ".join(g), c) for g, c in grams.most_common(top) if c > 1]


def load_reference(path: Path) -> str:
    """Referenztranskript (Segment-Zeilen) → Text ohne Zeitstempel."""
    lines = path.read_text(encoding="utf-8").splitlines()[1:]  # 1. Zeile = language
    segs = []
    for ln in lines:
        m = re.match(r"\[[\d.]+s -> [\d.]+s\]\s*(.*)", ln)
        if m and m.group(1).strip():
            segs.append(m.group(1).strip())
    return " ".join(segs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(_ROOT / "testaudio" / "vortrag_16k.wav"))
    ap.add_argument("--reference", default=str(Path("/home/stefan/testing/STTtest") / "transkript_whisper_largev3turbo_int8_beam5.txt"))
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--update", type=float, default=3.0)
    ap.add_argument("--context", type=float, default=24.0)
    ap.add_argument("--limit-seconds", type=float, default=0.0,
                    help="nur die ersten N Sekunden verarbeiten (0 = alles)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert sr == SAMPLE_RATE, f"samplerate {sr} != {SAMPLE_RATE}"
    dur = len(audio) / SAMPLE_RATE
    if args.limit_seconds > 0:
        dur = min(dur, args.limit_seconds)
        audio = audio[:int(dur * SAMPLE_RATE)]
    n_windows = int(np.ceil(dur / args.update))
    print(f"Audio: {dur:.1f}s → {n_windows} Fenster "
          f"(update={args.update}s, context={args.context}s)")

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=args.device,
                         compute_type="int8_float16" if args.device == "cuda" else "int8")

    old_st = TextStitcher()
    py_st = PyramidStitcher(context_s=args.context, update_s=args.update)
    last_window = ""
    t_start_all = time.monotonic()

    for w in range(1, n_windows + 1):
        t_end = min(w * args.update, dur)
        t_start = max(0.0, t_end - args.context)
        window = audio[int(t_start * SAMPLE_RATE):int(t_end * SAMPLE_RATE)]
        win_len = len(window) / SAMPLE_RATE

        segs, _ = model.transcribe(window, language="de", beam_size=5,
                                   condition_on_previous_text=False,
                                   word_timestamps=True)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        last_window = text

        old_st.add_chunk(text.split())
        words_ts = [(wd.word.strip(), wd.start, wd.end)
                    for s in segs for wd in (s.words or [])]
        py_st.add_window(words_ts, win_end=t_end, win_len=win_len)

        if w % 100 == 0:
            print(f"  … Fenster {w}/{n_windows} ({time.monotonic()-t_start_all:.0f}s)")

    old_st.flush()
    py_st.flush()
    old_text = old_st.stream_text
    py_text = py_st.committed_text

    ref = load_reference(Path(args.reference))
    print("\n=== ERGEBNISSE ===")
    print(f"Referenz (Voll-Audio, {len(norm_tokens(ref))} Wörter)")

    print(f"\n--- PyramidStitcher ({len(norm_tokens(py_text))} Wörter) ---")
    dup_py = duplicate_stats(py_text)
    print("WER:", f"{wer(ref, py_text)*100:.1f} %")
    print("Top-Wiederholungen (6-gram > 1×):")
    for g, c in dup_py:
        print(f"  {c}× {g!r}")
    if not dup_py:
        print("  (keine)")

    print(f"\n--- TextStitcher ({len(norm_tokens(old_text))} Wörter) ---")
    dup_old = duplicate_stats(old_text)
    print("WER:", f"{wer(ref, old_text)*100:.1f} %")
    print("Top-Wiederholungen (6-gram > 1×):")
    for g, c in dup_old:
        print(f"  {c}× {g!r}")
    if not dup_old:
        print("  (keine)")

    print(f"\n--- Baseline letztes Fenster ({len(norm_tokens(last_window))} Wörter) ---")
    print("WER:", f"{wer(ref, last_window)*100:.1f} %")

    out = _ROOT / "bench_vortrag_out"
    out.mkdir(exist_ok=True)
    (out / "pyramid.txt").write_text(py_text, encoding="utf-8")
    (out / "iterative.txt").write_text(old_text, encoding="utf-8")
    (out / "reference.txt").write_text(ref, encoding="utf-8")
    print(f"\nAusgaben: {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
