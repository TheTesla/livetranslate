# livetranslate

Echtzeit-Spracherkennung vom Mikrofon mit Whisper — komplett lokal, in einer Datei.

**Neu:** `text_stitch.py` fügt die überlappenden Fenster-Transkripte zu einem
kontinuierlichen, rückwirkend selbstkorrigierenden Textstrom zusammen
(Needleman-Wunsch-Alignment + Fuzzy-Ähnlichkeit + ROVER-Mehrheitsvotum).
Details und Entwicklungsgeschichte: [PROJECT_HISTORY.md](PROJECT_HISTORY.md).

## Prinzip

Drei Threads, entkoppelt über zwei Puffer:

```
Thread 1: Recorder      Mikrofon (16 kHz mono) → Ringspeicher
                        (Länge: context_interval Sekunden, gleitend)

Thread 2: Committer     alle update_interval Sekunden:
                        Snapshot des GESAMTEN Ringspeichers → Queue

Thread 3: Whisper-Worker  holt Snapshots aus der Queue, transkribiert, print
```

- **Interleaved:** Die Aufzeichnung läuft parallel zur Erkennung — während
  Whisper rechnet, geht kein Audio verloren.
- **Neuester gewinnt:** Ist Whisper langsamer als der Committer-Takt,
  verdrängt der neueste Snapshot die alten (veraltete Fenster sind für
  Live-Untertitel irrelevant).

## Installation

```bash
git clone https://github.com/TheTesla/livetranslate.git
cd livetranslate

python3.12 -m venv .venv
.venv/bin/pip install faster-whisper sounddevice numpy
```

Für GPU (NVIDIA, CUDA 12): zusätzlich

```bash
.venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Das Skript lädt die cu12-Bibliotheken automatisch aus dem venv (siehe
Preload am Dateianfang). Modelle werden beim ersten Start automatisch
aus dem HuggingFace-Hub geladen und gecached.

## Nutzung

```bash
.venv/bin/python livetranslate.py                      # Defaults
.venv/bin/python livetranslate.py --model small        # schneller
.venv/bin/python livetranslate.py --model large-v3     # beste Qualität (GPU empfohlen)
.venv/bin/python livetranslate.py --update 2 --context 15
.venv/bin/python livetranslate.py --device cpu         # ohne GPU
```

Beenden mit **Strg+C**.

## Parameter

| Parameter | Default | Bedeutung |
|---|---|---|
| `--update` | `2.0` | Sekunden zwischen zwei Whisper-Läufen (Commit-Takt) |
| `--context` | `10.0` | Ringpuffer-Länge in Sekunden (wie viel Audio Whisper pro Lauf sieht) |
| `--model` | `medium` | Whisper-Modellgröße (`tiny`/`base`/`small`/`medium`/`large-v3`) |
| `--device` | `cuda` | `cuda` oder `cpu` |
| `--compute` | auto | `int8_float16` (GPU, schnellste), `int8` (CPU), `float16`, `float32` |

## Ausgabe

Jeder Whisper-Lauf druckt eine Zeile:

```
[#001 18:37:21] (1.5s) Das ist ein Test des Mikrofons.
[#002 18:37:23] (1.6s) und hier kommt der nächste Satz.
```

- `#NNN` — laufende Nummer
- Uhrzeit — Druckzeitpunkt
- `(1.5s)` — Dauer des Whisper-Laufs
- danach der erkannte Text des letzten `context_interval`-Fensters

## Hardware-Hinweise

- **GPU (getestet: RTX-Turing 4 GB):** `medium` mit `int8_float16` schafft
  ~0,8–2 s Dekodierzeit pro 10-s-Fenster — passt locker in den 2-s-Takt.
- **CPU:** `small` mit `int8` empfohlen; `medium` braucht ~2 s pro Lauf
  (dann `--update` größer wählen, z. B. 4).
- Mikrofonqualität dominiert: Ein Headset liefert deutlich bessere Ergebnisse
  als Laptop-Arrays. Bei Halluzinationen auf leisem Signal zuerst den
  Eingangspegel prüfen (`wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 1.0`).

## Lizenz / Credits

Baut auf [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2-Reimplementation von OpenAI Whisper) und
[Silero/sounddevice](https://python-sounddevice.readthedocs.io/) für Audio-I/O.
