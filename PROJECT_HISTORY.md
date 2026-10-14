# Projektverlauf: Von der Konsole zum kontinuierlichen Textstrom

Entwicklungsdokumentation des livetranslate-Projekts — die wichtigsten
Schritte, Fehlannahmen und deren Korrekturen im Gespräch mit dem Entwickler.
Die Datei ist chronologisch aufgebaut; jede Phase nennt: Anforderung →
Umsetzung → gemessenes Ergebnis → ggf. Korrektur.

---

## Phase 1: Minimal-Setup — Mikrofon → Whisper → Konsole

**Anforderung:** Eine einzige Datei. Mikrofon füllt einen Ringspeicher der
Länge `context_interval`; alle `update_interval` Sekunden geht der komplette
Pufferinhalt (Anfang beachten!) an Whisper, das Ergebnis wird mit `print`
angezeigt. Intervalle werden vor dem Start in Sample-Anzahlen umgerechnet
und angezeigt.

**Umsetzung (`livetranslate.py`):** Der Mikrofon-Callback akkumuliert exakt
`update_samples` große Chunks (Rest wandert in den nächsten Chunk — lückenlos,
keine Zeitdrift). Die Whisper-Schleife nimmt einen Chunk aus der Queue, hängt
ihn ans Ring-Array an und schneidet vorne so viel ab, dass die Gesamtlänge
exakt `context_samples` beträgt. Kein Lock, kein Worker-Thread.

```
update_interval : 2.00 s = 32000 Samples (62 Mikrofon-Blöcke à 512)
context_interval: 10.00 s = 160000 Samples
```

Verifizierter Lauf: 19 Updates in 50 s, null verworfene Chunks, null Overflows.

## Phase 2: Die Ringpuffer-Ausrichtung

**Beobachtung:** „Irgendwie scheint der Ringpuffer nicht immer korrekt
ausgerichtet zu sein."

**Diagnose:** Verworfene Mikrofon-Frames erzeugten Zeitlücken (Queue lief
voll unter Whisper-Last, Callback warf stillschweigend Frames weg), plus
O(n)-Kopiererei pro Frame. Der Soundkarten-Callback läuft auf PortAudio's
C-Thread und muss sich den GIL holen — hält der Whisper-Thread den GIL in
seinen Python-Abschnitten (Mel-Spektrogramm, Tokenizer), verzögert sich der
Callback bis zum Overflow.

**Diskussion:** „Das Modell läuft nebenläufig und beansprucht keine CPU-Zeit"
stimmt nur für die CTranslate2-Kernel selbst. `transcribe()` ist ~280 Zeilen
Python-Orchestrierung plus 227 Zeilen NumPy-Feature-Extraktion — alles
GIL-gebunden. Messung im Quellcode verifiziert.

**Fix:** Fest vorallocierter Puffer, Drop-/Overflow-Zähler mit ⚠-Warnung.
Später (Phase 1-Umbau) ohnehin obsolet, aber die Zähler blieben.

## Phase 3: Queue-Design und interleaved Betrieb

Zwischenstand mit 3 Threads (Recorder → Committer → Whisper-Worker), dann
vom Entwickler radikal vereinfacht: der Mikrofon-Callback selbst wirft
`update_interval`-lange Chunks in die Queue, der Whisper-Loop konsumiert.
Die Soundkarte puffert die Whisper-Zeit von selbst. Die Chunk-Queue hat
Kapazität 8 (= 16 s Rückstau bei 2-s-Chunks) — danach werden Chunks mit ⚠
verworfen statt still.

## Phase 4: Modellwahl — large-v3-turbo

| Modell | Parameter | Decoder-Layer | T550 (4 GB) |
|---|---|---|---|
| large (v1) | 1550 M | 32 | veraltet, nicht verwenden |
| large-v3 | 1550 M | 32 | ~4–6 s/Fenster — zu langsam für 2–3-s-Takt |
| **large-v3-turbo** | 809 M | 4 | ~8× schneller, nahe v3-Genauigkeit — **passt** |

faster-whisper wird verwendet (CTranslate2, `int8_float16`) — gleiche
Gewichte wie OpenAI-Whisper, aber deutlich weniger VRAM und schneller.

## Phase 5: Der kontinuierliche Text-Strom (`text_stitch.py`)

**Anforderung:** 3 s update / 24 s context = 7–8 überlappende 24-s-Fenster.
Die Textströme müssen synchronisiert/„gefittet" werden — Wort für Wort,
„weich": manchmal fehlt ein Wort, manchmal ist eins zu viel, Wörter können
fuzzy ähnlich sein („Context-Fit-Fenster" / „Context-Fit-Tänzer"). Lücken
auffüllen/stretchen nach Ähnlichkeit.

**Verfahren (die etablierten Namen):**

1. **Needleman-Wunsch-Alignment** (globales Sequenzalignment): ordnet jedes
   Fenster der Hypothese dem Stromende zu; Gaps = fehlende/zu viel Wörter.
   Substitution nur bei sim ≥ 0.75, sonst erzwungener Gap (Score −2.5 < zwei
   Gaps −2.0) — sonst blickt das Alignment fremde Wörter zwangszuordnen.
2. **Fuzzy-Ähnlichkeit** (Levenshtein-Ratio via `difflib.SequenceMatcher`,
   normalisiert auf Kleinbuchstaben ohne Satzzeichen).
3. **ROVER** (Recognizer Output Voting Error Reduction, NIST): Mehrheits-
   votum pro Wortstelle statt neuester Sichtung.
4. **Weiche Zone:** Die letzten `soft_tail` Wörter (Default 40) bleiben
   rückwirkend korrigierbar — der lange Kontext verbessert die Vergangenheit.
   Ein Wort sammelt bis zu context/update = 8 Stimmen (eine pro Fenster, das
   sein Audio noch sieht), dann friert es ein.

**Abdeckungspyramide:** Vom ältesten Fenster ist nur sein führender 3-s-
Streifen eindeutig, vom zweitältesten 6 s usw. Neue Textbeiträge kommen nur
aus den letzten 3 s; die redundanten 21 s jedes Fensters sind die
rückwirkenden Korrektur-Stimmen.

**Offline-Nachweis** (`demos/04_stitch_offline.py`): espeak-ng-Referenztext
→ 42 s Audio → Fenster-Transkription wie live → Stitcher → WER gegen
Voll-Audio-Referenz:

```
WER gestitchter Strom ......... 3.0 %
WER nur letztes Fenster ....... 44.4 %
```

## Phase 6: Die Fehler, die die Tests fanden

Der Stitcher wurde durch drei gemeldete/beobachtete Fälle hart getestet:

**a) Flip-Flop und Korruption (weiche Zone, erste Version „neueste Sichtung
gewinnt"):**
```
gestrescht → gestrecht → gestrechert → gestrescht   (Churn)
Halluzinationen → Halluziationen                     (ein schlechtes Fenster
                                                      zerstört ein gutes Wort)
```
Fix: **Mehrheitsvotum** — eine einzelne abweichende Sichtung kann nichts
überschreiben; Änderungen nur beim Stimmengleichstand (dann neueste).

**b) Positions-Verrutsch („geworden/gekommen"):** Sprecher sagte „…kalte
Wasser geworden", ein Fenster erkannte „gekommen" (sim 0.625 < 0.75, korrekt
kein Match) — aber das Alignment legte die Einfügungen VOR das gap-
beibehaltene Wort: „geworden" verrutschte ans Stromende und ging verloren;
zeitweise erschienen beide Varianten als sichtbares Duplikat
(„geworden, gekommen"). Fix: Einfügungen werden an Ref-Positionen verankert
(„Anti-Verrutsch-Regel") und Varianten-Paare bekommen einen expliziten
Slot-Tag und verschmelzen zu einem Slot mit kombinierten Stimmen.

**c) Ähnlichkeits-Scan als Antipattern:** Der erste Verschmelzungsversuch
matchte benachbarte Wörter per String-Ratio — und verschmolz echte deutsche
Nachbarn: „erneut"/„erkennt." (0.667) → „erkennt." verschwand; „ist"/„ins"
(0.667) → Doppelung; „eins"/„zwei" (0.57) → kippte zu „zwei". WER stieg
3.0 → 4.0 %. Fix: Verschmelzung NUR für explizit slot-getaggte Paare aus
der Varianten-Regel (beide Wörter ≥ 6 Zeichen); fuzzy-Nachbar-Votum
ebenfalls auf ≥ 6 Zeichen beschränkt.

**Nach allen Fixes:** 18/18 Unit-Tests, WER wieder 3.0 %, „erkennt." wieder
im Strom, Nutzer-Fall korrekt rekonstruiert.

## Phase 7: Einordnung in existierende Bibliotheken

| Anforderung | whisper_streaming (UFAL) | ROVER (NIST SCTK) | text_stitch.py |
|---|---|---|---|
| Live-Streaming | ✅ (LocalAgreement-2) | ❌ (Batch) | ✅ |
| 8 überlappende Vollfenster mergen | ❌ (commit + trim) | ✅ | ✅ |
| Rückwirkende Korrektur | ❌ | n/a | ✅ |
| Produktionserprobt | ✅ | ✅ (offline) | 18 Tests + WER-Nachweis |

`whisper_streaming` friert die Vergangenheit ein (committet den stabilen
Präfix und trimmt den Puffer) — löst Instabilität durch Wegwerfen, nicht
durch Korrektur. ROVER ist ein offline Batch-Tool für mehrere ASR-Systeme.
Die Kombination „gleitende Vollfenster + rückwirkende Verbesserung" ist die
Lücke, die text_stitch.py füllt.

## Architektur-Entscheidungen im Rückblick

- **Keine Threads:** Der Soundkarten-Puffer überbrückt die Whisper-Zeit;
  ein einziger Python-Thread zwischen zwei Callbacks. Weniger
  Nebenläufigkeits-Randfälle als die 3-Thread-Version.
- **faster-whisper statt openai-whisper:** CTranslate2, int8_float16 —
  ohne die wäre `large-v3-turbo` auf der 4-GB-Karte nicht lauffähig.
- **Grenze des Designs:** Bei dauerhaft >2 s Dekodierzeit pro 2-s-Chunk
  staut die Queue und Chunks werden verworfen (sichtbar per ⚠). Dann
  `--update` erhöhen oder später Whisper in einen eigenen Prozess
  auslagern (der einzige vollständige GIL-Entwurf).

## Nutzung

```bash
git clone https://github.com/TheTesla/livetranslate.git
cd livetranslate
python3.12 -m venv .venv
.venv/bin/pip install faster-whisper sounddevice numpy jiwer soxr soundfile
.venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   # nur für GPU

# Live-Untertitel mit kontinuierlichem Strom:
.venv/bin/python livetranslate.py --model large-v3-turbo --update 3 --context 24

# Rohe Fenstertexte statt Strom:
.venv/bin/python livetranslate.py --raw

# Offline-Nachweis des Stitchers:
.venv/bin/python demos/04_stitch_offline.py

# Tests:
.venv/bin/python -m pytest tests/test_stitch.py -q
```
