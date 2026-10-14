#!/usr/bin/env python
"""demos/04_stitch_offline.py — Offline-Nachweis der Stitcher.

Erzeugt (einmalig) eine Referenz-Audio-Datei per espeak-ng, transkribiert
sie (a) komplett als Referenz und (b) wie live: Fenster à context_interval
Sekunden, alle update_interval Sekunden ein neues Fenster. Die Fenster-
Hypothesen gehen in BEIDE Stitcher:

  - PyramidStitcher (Zeitanker, word_timestamps=True): Slots schließen sich,
    sobald ihre Abdeckung vollständig ist; emittierter Text ist unveränderlich
  - TextStitcher (text_stitch.py, iteratives Alignment gegen den Strom)

Ausgabe: WER-Vergleich beider Verfahren + Baseline "nur letztes Fenster".

Aufruf:
  .venv/bin/python demos/04_stitch_offline.py
  .venv/bin/python demos/04_stitch_offline.py --model large-v3-turbo
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import subprocess
import sys
import time
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
import soxr  # noqa: E402
from jiwer import wer  # noqa: E402

from text_stitch import TextStitcher  # noqa: E402
from pyramid_stitch import PyramidStitcher  # noqa: E402

SAMPLE_RATE = 16000

REF_TEXT = (
    "Guten Abend zusammen. Heute zeige ich euch den Kontext Fit Fenster "
    "Algorithmus für live Untertitel. Sieben überlappende Fenster sehen "
    "alle dasselbe Audio. Der Fitter synchronisiert die Textströme wort "
    "für Wort. Ähnliche Wörter wie Fenster und Tänzer werden nach "
    "Ähnlichkeit gefittet. Fehlende Wörter werden aufgefüllt und "
    "gestretched. Am Ende entsteht ein kontinuierlicher Textstrom. "
    "Jetzt kommen wir zum zweiten Beispiel. Der Strom wächst von links "
    "nach rechts. Jedes neue Fenster bestätigt die Wörter der letzten "
    "Sekunden. Ein Wort gilt erst dann als endgültig, wenn ein späteres "
    "Fenster es erneut erkennt. Auf diese Weise verschwinden "
    "Halluzinationen von selbst. Der finale Text bleibt stabil und "
    "lückenlos."
)


def ensure_audio(path: Path) -> None:
    if path.exists():
        return
    print(f"Erzeuge {path.name} per espeak-ng …")
    subprocess.run(["espeak-ng", "-v", "de", "-w", str(path), REF_TEXT], check=True)


def transcribe(model, audio: np.ndarray, word_timestamps: bool = False):
    segments, _ = model.transcribe(audio, language="de", beam_size=5,
                                   condition_on_previous_text=False,
                                   word_timestamps=word_timestamps)
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs).strip()
    return text, segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--update", type=float, default=3.0)
    ap.add_argument("--context", type=float, default=24.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    wav = _ROOT / "testaudio" / "stitch_ref.wav"
    wav.parent.mkdir(exist_ok=True)
    ensure_audio(wav)

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = soxr.resample(audio, sr, SAMPLE_RATE)
    dur = len(audio) / SAMPLE_RATE
    n_windows = int(np.ceil(dur / args.update))
    print(f"Audio: {dur:.1f}s → {n_windows} Fenster "
          f"(update={args.update}s, context={args.context}s)\n")

    from faster_whisper import WhisperModel
    print(f"Lade {args.model} ({args.device}) …")
    model = WhisperModel(args.model, device=args.device,
                         compute_type="int8_float16" if args.device == "cuda" else "int8")

    print("Referenz: transkribiere GESAMTES Audio …")
    ref_transcript, _ = transcribe(model, audio)
    print(f"Referenz: {ref_transcript}\n")

    old_st = TextStitcher()
    py_st = PyramidStitcher(context_s=args.context, update_s=args.update)
    last_window_text = ""

    print("—" * 78)
    for w in range(1, n_windows + 1):
        t_end = min(w * args.update, dur)
        t_start = max(0.0, t_end - args.context)
        window = audio[int(t_start * SAMPLE_RATE):int(t_end * SAMPLE_RATE)]
        win_len = len(window) / SAMPLE_RATE
        t0 = time.monotonic()
        text, segs = transcribe(model, window, word_timestamps=True)
        dt = time.monotonic() - t0
        last_window_text = text

        old_st.add_chunk(text.split())
        words_ts = [(wd.word.strip(), wd.start, wd.end)
                    for s in segs for wd in (s.words or [])]
        py_emitted = py_st.add_window(words_ts, win_end=t_end, win_len=win_len)

        print(f"Fenster {w:02d} ({t_start:4.1f}→{t_end:4.1f}s, {dt:4.1f}s)"
              f"  py+{' '.join(py_emitted) if py_emitted else '—'}")

    old_st.flush()
    py_st.flush()
    stitched_old = old_st.stream_text
    stitched_py = py_st.committed_text

    print("\n" + "=" * 78)
    print("PYRAMIDEN-STROM (Zeitanker):")
    print(stitched_py)
    print()
    print("ITERATIVER STROM (text_stitch, zum Vergleich):")
    print(stitched_old)
    print("=" * 78)

    w_py = wer(ref_transcript, stitched_py)
    w_old = wer(ref_transcript, stitched_old)
    w_last = wer(ref_transcript, last_window_text)
    print(f"\nWER gegen Voll-Audio-Referenz:")
    print(f"  Pyramiden-Stitcher  : {w_py*100:5.1f} %")
    print(f"  iterativer Stitcher : {w_old*100:5.1f} %")
    print(f"  nur letztes Fenster : {w_last*100:5.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

