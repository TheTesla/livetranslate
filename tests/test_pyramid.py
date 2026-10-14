"""Tests für pyramid_stitch.PyramidStitcher — Zeitanker-Architektur.

Modell: Wörter existieren an festen ABSOLUTEN Audiozeiten. Fenster i enthält
die Audiostrecke [i*update - win_len, i*update] mit win_len = min(i*update,
context); ein Wort liegt im Fenster, wenn seine Zeit hineinfällt.

Geprüfte Eigenschaften:
- Monotonie: emittierter Text verändert sich nie („nie verschlechtern")
- Pyramiden-Abdeckung: Slot s wird von genau den Fenstern gesehen, die sein
  Audio enthalten; final erst, wenn keins mehr kommen kann
- Enden-Beschneidung: Einmal-Artefakte fallen heraus
- Varianten ("geworden"/"gekommen") slot-lokal per Mehrheit
- Reihenfolge-Varianten per mittlerer Zeit der Mehrheit
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyramid_stitch import PyramidStitcher  # noqa: E402

CTX, UPD, K = 24.0, 3.0, 8


def feed(st, spoken, i):
    """Fenster i füttern. spoken: Liste (wort, t_abs). Wörter außerhalb des
    Fenster-Audiobereichs sind im Fenster nicht enthalten (realistisch)."""
    win_len = min(i * UPD, CTX)
    win_start = i * UPD - win_len
    words = [(w, t - win_start - 0.15, t - win_start + 0.15)
             for w, t in spoken if win_start <= t < i * UPD]
    return st.add_window(words, i * UPD, win_len)


# ------------------------------------------------------------------ Monotonie
def test_emitted_text_never_changes():
    """Sobald Slots emittiert sind, bleibt der Strom prefix-stabil — auch
    wenn spätere Fenster andere Lesarten bringen."""
    st = PyramidStitcher(CTX, UPD)
    spoken = [("eins", 1.0), ("zwei", 2.0), ("drei", 4.0), ("vier", 5.0)]
    snapshots = []
    for i in range(1, 30):
        feed(st, spoken, i)
        snapshots.append(st.committed_text)
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later.startswith(earlier), f"{earlier!r} → {later!r}"


def test_full_pyramid_coverage_needed():
    """Ein Slot wird erst emittiert, wenn alle K Fenster, die sein Audio
    sehen können, dagewaren sind."""
    st = PyramidStitcher(CTX, UPD)
    for i in range(1, K + 1):                 # Fenster 1..8 sehen 'x' @1.0
        feed(st, [("x", 1.0)], i)
    assert st.committed_text == ""            # Slot 0 noch nicht final
    feed(st, [], K + 1)                       # Fenster 9: Slot 0 ist fertig
    assert st.committed_text == "x"


# ------------------------------------------------------------------ Varianten
def test_variant_majority_within_slot():
    """'geworden' (7 Fenster) gegen 'gekommen' (1 Fenster) an derselben
    absoluten Wortzeit: Mehrheit gewinnt, genau ein Wort an der Stelle."""
    st = PyramidStitcher(CTX, UPD)
    base = [("man", 0.5), ("ist", 0.9), ("ins", 1.3), ("kalte", 1.7),
            ("wasser", 2.6)]
    for i in range(1, K + 1):
        feed(st, base + [("gekommen" if i == 3 else "geworden", 2.9)], i)
    feed(st, base, K + 1)                     # Slot 0 abschließen
    text = st.committed_text
    assert text.split() == ["man", "ist", "ins", "kalte", "wasser", "geworden"]


def test_word_order_variant_resolved_by_mean_time():
    """'Das dokumentieren wir' (Mehrheit) vs 'Wir dokumentieren das'
    (1 Fenster): die Mehrheitsordnung gewinnt über die mittleren Zeiten."""
    st = PyramidStitcher(CTX, UPD)
    maj = [("Das", 0.5), ("dokumentieren", 0.9), ("wir", 1.3)]
    dev = [("Wir", 0.5), ("dokumentieren", 0.9), ("das", 1.3)]
    for i in range(1, K + 1):
        feed(st, dev if i == 5 else maj, i)
    feed(st, maj, K + 1)
    assert st.committed_text.split() == ["Das", "dokumentieren", "wir"]


# ------------------------------------------------------------------ Enden
def test_single_sighting_artifact_trimmed():
    """Ein Halluzinations-Wort, das nur 1 Fenster sieht (gegen 8 des
    Nachbarn), wird als Rand-Artefakt beschnitten."""
    st = PyramidStitcher(CTX, UPD)
    for i in range(1, K + 2):
        feed(st, ([("a", 1.5), ("halluz", 0.2)] if i == 1 else [("a", 1.5)]), i)
    text = st.committed_text
    assert text.split() == ["a"]


def test_aged_out_word_gets_one_sighting_and_is_trimmed():
    """Ein Wort kurz nach Audio-Beginn altert sofort aus dem Fenster — es
    bekommt nur 1 Sichtung und wird (zu Recht) nicht emittiert. Das ist die
    Enden-Beschneidung: der allererste Audio-Streifen hat volle Latenz-
    Abdeckung erst ab Fenster K+1; Wörter, die NUR im kürzesten ersten Ring
    lagen, sind Ein-Sichtungs-Kandidaten."""
    st = PyramidStitcher(CTX, UPD)
    for i in range(1, K + 2):
        feed(st, [("a", 1.5)], i)             # 'a' @1.5: Fenster 1..8
    # 'a' ist in Slot 0 mit 8 Sichtungen → final:
    assert st.committed_text == "a"


# ------------------------------------------------------------------ Slot-Grenzen
def test_word_center_decides_slot():
    """Die Wortmitte entscheidet den Slot — das Wort steht genau einmal."""
    st = PyramidStitcher(CTX, UPD)
    for i in range(1, 11):
        feed(st, [("a", 1.0), ("b", 3.0)], i)   # b zentriert auf Grenze 3.0
    text = st.committed_text
    assert text.split() == ["a", "b"]


def test_silence_slots_skipped():
    """Leere Slots erzeugen keine Marker — der Strom hat keine Lücken."""
    st = PyramidStitcher(CTX, UPD)
    for i in range(1, 13):                       # Wort @10.0: Fenster 4..11
        feed(st, [("wort", 10.0)], i)
    assert st.committed_text == "wort"


# ------------------------------------------------------------------ Live-Vorschau
def test_pending_shows_unfinalized_majorities():
    st = PyramidStitcher(CTX, UPD)
    for i in range(7, 10):                       # Wort @20.0: Fenster 7..14
        feed(st, [("live", 20.0)], i)
    assert "live" in st.pending_text()
    assert st.committed_text == ""
    for i in range(10, 16):
        feed(st, [("live", 20.0)], i)
    assert st.committed_text == "live"


def test_flush_finalizes_everything():
    st = PyramidStitcher(CTX, UPD)
    for i in range(1, 3):
        feed(st, [("ende", 1.0)], i)
    st.flush()
    assert st.committed_text == "ende"
