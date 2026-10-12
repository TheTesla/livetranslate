"""livetranslate.py — Mikrofon → Whisper → Konsole, in EINER Datei.

Prinzip (Queue-Architektur, 3 Threads):
  - Recorder:     Mikrofon (16 kHz mono) füllt kontinuierlich einen Ring-
                  speicher der Länge context_interval Sekunden.
  - Committer:    Alle update_interval Sekunden wird der GESAMTE Ring-
                  speicherinhalt (Anfang bis Ende!) in die Erkennungs-Queue
                  geworfen.
  - Whisper-Worker: Holt die Snapshots aus der Queue und transkribiert.

Die Aufzeichnung läuft PARALLEL zur Erkennung — egal wie lange Whisper
braucht, es geht kein Audio verloren (interleaved Betrieb). Ist Whisper
dauerhaft zu langsam, verdrängt der neueste Snapshot die alten.

Aufruf:
  .venv/bin/python livetranslate.py                      # Defaults
  .venv/bin/python livetranslate.py --model medium       # bessere Qualität
  .venv/bin/python livetranslate.py --update 2 --context 15

Beenden mit Strg+C.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import queue
import threading
import time
from pathlib import Path

# CTranslate2 dlopens libcublas.so.12 (CUDA 12); das Torch-Wheel bringt nur
# CUDA-13 mit. Die cu12-Bibliotheken liegen im venv — vor dem Import laden.
_ROOT = Path(__file__).resolve().parent
for _lib in ("cublas", "cudnn"):
    for _so in sorted(glob.glob(str(_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / _lib / "lib" / "lib*.so*"))):
        try:
            ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

SAMPLE_RATE = 16000


def main() -> int:
    ap = argparse.ArgumentParser(description="Mikrofon → Whisper → Konsole")
    ap.add_argument("--model", default="medium",
                    help="Whisper-Modell (small/medium/large-v3)")
    ap.add_argument("--update", type=float, default=2.0,
                    help="update_interval: Sekunden zwischen zwei Queue-Einwürfen")
    ap.add_argument("--context", type=float, default=10.0,
                    help="context_interval: Ringpuffer-Länge in Sekunden")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute", default=None,
                    help="int8_float16 (GPU) | int8 (CPU) | float16 | float32")
    args = ap.parse_args()

    compute = args.compute or ("int8_float16" if args.device == "cuda" else "int8")

    from faster_whisper import WhisperModel
    print(f"Lade Whisper {args.model} ({args.device}/{compute}) …")
    model = WhisperModel(args.model, device=args.device, compute_type=compute)
    print(f"bereit. Alle {args.update:.1f}s werden die letzten {args.context:.1f}s "
          f"in die Queue geworfen. (Strg+C beendet)\n")

    # ---------------------------------------------------------------- Zustand
    max_samples = int(args.context * SAMPLE_RATE)
    ring = np.zeros(0, dtype=np.float32)
    ring_lock = threading.Lock()
    mic_q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=400)
    # Erkennungs-Queue (Größe 2): wenn Whisper dauerhaft zu langsam ist,
    # gewinnt immer der NEUESTE Snapshot — alte sind ohnehin veraltet.
    work_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        try:
            mic_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass                          # Frame verwerfen statt blockieren

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=512, callback=callback)
    stream.start()

    # ------------------------------------------------------ Thread 1: Recorder
    stop_event = threading.Event()

    def recorder() -> None:
        nonlocal ring
        while not stop_event.is_set():
            try:
                block = mic_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if block is None:
                return
            with ring_lock:
                ring = np.concatenate([ring, block])
                if len(ring) > max_samples:
                    ring = ring[-max_samples:]   # Anfang abschneiden (Ring!)

    threading.Thread(target=recorder, daemon=True).start()

    # ------------------------------------------------------ Thread 2: Committer
    # Alle update_interval Sekunden: Snapshot des GESAMTEN Ringspeichers in
    # die work_q. Blockiert nie — ist die Queue voll (Whisper hängt hinterher),
    # verdrängt der neueste Snapshot den ältesten.
    def committer_loop() -> None:
        next_commit = time.monotonic() + args.update
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_commit:
                time.sleep(min(0.05, next_commit - now))
                continue
            with ring_lock:
                snapshot = ring.copy()
            if len(snapshot) > SAMPLE_RATE // 2:
                try:
                    work_q.put_nowait(snapshot)
                except queue.Full:
                    try:
                        work_q.get_nowait()      # ältesten verwerfen
                    except queue.Empty:
                        pass
                    try:
                        work_q.put_nowait(snapshot)
                    except queue.Full:
                        pass
            next_commit += args.update
            if next_commit < time.monotonic():
                next_commit = time.monotonic()

    threading.Thread(target=committer_loop, daemon=True).start()

    # -------------------------------------------------- Thread 3: Whisper-Worker
    run_number = 0
    try:
        while True:
            try:
                snapshot = work_q.get(timeout=0.5)
            except queue.Empty:
                continue
            run_number += 1
            t0 = time.monotonic()
            segments, _info = model.transcribe(
                np.ascontiguousarray(snapshot, dtype=np.float32),
                language="de",
                beam_size=5,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            dt = time.monotonic() - t0
            print(f"[#{run_number:03d} {time.strftime('%H:%M:%S')}] "
                  f"({dt:.1f}s) {text if text else '—'}", flush=True)
    except KeyboardInterrupt:
        print("\n— Ende —")
    finally:
        stop_event.set()
        stream.stop()
        stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
