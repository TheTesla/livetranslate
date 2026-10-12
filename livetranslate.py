"""livetranslate.py — Mikrofon → Whisper → Konsole, in EINER Datei, OHNE Threads.

Prinzip (minimal):
  - Der Mikrofon-Callback akkumuliert update_interval lange Chunks und wirft
    sie in eine Queue (der Soundkarten-Puffer überbrückt die Whisper-Zeit).
  - Die Whisper-Schleife nimmt einen Chunk aus der Queue, hängt ihn ans
    Ring-Array an und schneidet am Anfang so viel ab, dass die Gesamtlänge
    genau context_interval beträgt. Dann transkribieren, print.
  - Kein Lock, kein Worker-Thread, kein Committer — ein einziger
    Python-Thread zwischen zwei Callbacks.

Alle Intervalle werden vor dem Start in Sample-Anzahlen umgerechnet und
angezeigt.

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
                    help="update_interval: Chunk-Länge für die Queue (Sekunden)")
    ap.add_argument("--context", type=float, default=10.0,
                    help="context_interval: Ring-Array-Länge (Sekunden)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute", default=None,
                    help="int8_float16 (GPU) | int8 (CPU) | float16 | float32")
    args = ap.parse_args()

    compute = args.compute or ("int8_float16" if args.device == "cuda" else "int8")

    # ---------------------------------------------------------------- Intervalle
    # VOR dem Start in Sample-Anzahlen umrechnen und anzeigen:
    update_samples = int(round(args.update * SAMPLE_RATE))
    context_samples = int(round(args.context * SAMPLE_RATE))
    update_blocks = update_samples // 512          # Mikrofon-Blöcke à 512 Samples

    from faster_whisper import WhisperModel
    print(f"Lade Whisper {args.model} ({args.device}/{compute}) …")
    model = WhisperModel(args.model, device=args.device, compute_type=compute)
    print(
        f"bereit.\n"
        f"  update_interval : {args.update:.2f} s = {update_samples} Samples "
        f"({update_blocks} Mikrofon-Blöcke à 512)\n"
        f"  context_interval: {args.context:.2f} s = {context_samples} Samples\n"
        f"  (Strg+C beendet)\n"
    )

    # ---------------------------------------------------------------- Queue
    # Der Callback sammelt update_interval lange Chunks und wirft sie hier ein.
    chunk_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)

    pending = []                 # Mikrofon-Blöcke bis zur Chunk-Größe
    pending_samples = 0
    dropped_chunks = 0
    overflow_events = 0

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        nonlocal pending, pending_samples, dropped_chunks, overflow_events
        if status:
            overflow_events += 1
        block = indata[:, 0].copy()
        pending.append(block)
        pending_samples += len(block)
        # Chunk voll? → exakt update_samples heraustrennen, Rest behalten.
        while pending_samples >= update_samples:
            data = b"".join(b.tobytes() for b in pending)
            chunk_all = np.frombuffer(data, dtype=np.float32)
            head, tail = chunk_all[:update_samples], chunk_all[update_samples:]
            pending = [tail.copy()] if len(tail) else []
            pending_samples = len(tail)
            try:
                chunk_q.put_nowait(head.copy())
            except queue.Full:
                dropped_chunks += 1   # Whisper hängt → Chunk verwerfen (Zeitlücke)

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=512, callback=callback)
    stream.start()

    # ---------------------------------------------------------------- Whisper-Loop
    ring = np.zeros(0, dtype=np.float32)   # wächst bis context_samples, dann gekürzt
    run_number = 0
    last_dropped = 0
    last_overflow = 0
    try:
        while True:
            try:
                chunk = chunk_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Chunk ans Ring-Array anhängen …
            ring = np.concatenate([ring, chunk])
            # … und am ANFANG so viel abschneiden, dass die Gesamtlänge
            # exakt context_samples beträgt.
            if len(ring) > context_samples:
                ring = ring[-context_samples:]

            run_number += 1
            t0 = time.monotonic()
            segments, _info = model.transcribe(
                np.ascontiguousarray(ring, dtype=np.float32),
                language="de",
                beam_size=5,
                word_timestamps=False,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            dt = time.monotonic() - t0

            warn = ""
            if dropped_chunks > last_dropped:
                warn += f" ⚠ {dropped_chunks - last_dropped} Chunks verworfen (Whisper zu langsam → Zeitlücke)"
                last_dropped = dropped_chunks
            if overflow_events > last_overflow:
                warn += f" ⚠ {overflow_events - last_overflow} Overflows"
                last_overflow = overflow_events

            print(f"[#{run_number:03d} {time.strftime('%H:%M:%S')}] "
                  f"({dt:.1f}s, {len(ring)/SAMPLE_RATE:.1f}s Kontext) "
                  f"{text if text else '—'}{warn}", flush=True)
    except KeyboardInterrupt:
        print("\n— Ende —")
    finally:
        stream.stop()
        stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
