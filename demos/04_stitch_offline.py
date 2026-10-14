#!/usr/bin/env python
"""demos/04_stitch_offline.py — Offline-Nachweis des Text-Stitchers.

Erzeugt (einmalig) eine Referenz-Audio-Datei per espeak-ng, transkribiert
sie (a) komplett als Referenz und (b) wie live: Fenster à context_interval
Sekunden, alle update_interval Sekunden ein neues Fenster — genau wie in
livetranslate.py. Die Fenster-Hypothesen gehen in den TextStitcher.

Ausgabe: Stream-Entwicklung pro Tick + WER-Vergleich
  - gestitchter kontinuierlicher Strom  vs.  Referenz (Voll-Audio)
  - Baseline "nur letztes Fenster"      vs.  Referenz

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


def transcribe(model, audio: np.ndarray) -> str:
    segments, _ = model.transcribe(audio, language="de", beam_size=5,
                                   condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--update", type=float, default=3.0,
                    help="update_interval in Sekunden")
    ap.add_argument("--context", type=float, default=24.0,
                    help="context_interval in Sekunden")
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
          f"(update={args.update}s, context={args.context}s → "
          f"~{int(args.context/args.update)} überlappende Fenster)\n")

    from faster_whisper import WhisperModel
    print(f"Lade {args.model} ({args.device}) …")
    model = WhisperModel(args.model, device=args.device,
                         compute_type="int8_float16" if args.device == "cuda" else "int8")

    print("Referenz: transkribiere GESAMTES Audio …")
    t0 = time.monotonic()
    ref_transcript = transcribe(model, audio)
    print(f"Referenz in {time.monotonic()-t0:.1f}s: {ref_transcript}\n")

    st = TextStitcher()
    print("—" * 78)
    last_window_text = ""
    for w in range(1, n_windows + 1):
        t_end = min(w * args.update, dur)
        t_start = max(0.0, t_end - args.context)
        window = audio[int(t_start * SAMPLE_RATE):int(t_end * SAMPLE_RATE)]
        t0 = time.monotonic()
        text = transcribe(model, window)
        dt = time.monotonic() - t0
        last_window_text = text
        confirmed = st.add_chunk(text.split())
        corr = " ".join(f"⇄{o}→{n}" for o, n in st.last_corrections)
        show = " ".join(st.committed[-14:])
        pend = " ".join(st.pending[-6:])
        print(f"Fenster {w:02d} ({t_start:4.1f}→{t_end:4.1f}s, {dt:4.1f}s) "
              f"+{' '.join(confirmed) if confirmed else '—'}"
              f"{'  ' + corr if corr else ''}")
        print(f"    Stream: … {show}")
        print(f"    pending: {pend}")

    st.flush()
    stitched = st.stream_text

    print("\n" + "=" * 78)
    print("GESTITCHTER STROM:")
    print(stitched)
    print("=" * 78)

    w_stitch = wer(ref_transcript, stitched)
    w_last = wer(ref_transcript, last_window_text)
    print(f"\nWER gegen Voll-Audio-Referenz:")
    print(f"  gestitchter Strom : {w_stitch*100:5.1f} %")
    print(f"  nur letztes Fenster (Baseline, deckt nur {args.context:.0f}s ab): "
          f"{w_last*100:5.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
