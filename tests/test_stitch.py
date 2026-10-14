"""Tests für text_stitch.TextStitcher — inkl. der vom Nutzer genannten Fälle:
überlappende Fenster, fehlende/zusätzliche Wörter, fuzzy ähnliche Wörter
("Context-Fit-Fenster" / "Context-Fit-Tänzer").
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_stitch import TextStitcher, similarity  # noqa: E402


# ------------------------------------------------------------------ Fuzzy-Ähnlichkeit
def test_similarity_identical():
    assert similarity("Fenster", "fenster.") == 1.0

def test_similarity_tänzer_fenster():
    # Der vom Nutzer genannte Fall: sehr ähnlich, aber nicht identisch.
    s = similarity("Context-Fit-Fenster", "Context-Fit-Tänzer")
    assert 0.75 <= s < 1.0

def test_similarity_disjoint():
    assert similarity("Hund", "Katharina") < 0.4


# ------------------------------------------------------------------ Grundmechanik
def test_first_chunk_goes_to_pending():
    st = TextStitcher()
    confirmed = st.add_chunk(["a", "b", "c"])
    assert confirmed == []
    assert st.pending == ["a", "b", "c"]

def test_sliding_window_stitching():
    """Realistische Fenster: Kontext 8 Wörter, Schritt 2 — jedes Wort erscheint
    in mehreren Fenstern und wird dadurch bestätigt (ROVER-Semantik)."""
    ref = "a b c d e f g h i j k".split()
    st = TextStitcher()
    for pos in range(2, len(ref) + 3, 2):
        st.add_chunk(ref[max(0, pos - 8):pos])
    st.flush()
    assert st.stream_text.split() == ref


# ------------------------------------------------------------------ Fuzzy-Fälle
def test_fuzzy_substitution_confirms():
    # Erstes Fenster erkennt "Tänzer", zweites "Fenster" — Konfirmation gewinnt.
    st = TextStitcher()
    st.add_chunk("Der Kontext-Fit-Tänzer funktioniert".split())
    st.add_chunk("Der Kontext-Fit-Fenster funktioniert gut".split())
    # "gut" ist neu → pending; der Rest ist bestätigt.
    assert "Kontext-Fit-Fenster" in st.committed
    assert "Kontext-Fit-Tänzer" not in st.committed

def test_fuzzy_confirmation_with_punctuation():
    st = TextStitcher()
    st.add_chunk("Heute zeige ich euch das".split())
    st.add_chunk("Heute zeige ich euch das Fenster,".split())
    assert st.committed == "Heute zeige ich euch das".split()


# ------------------------------------------------------------------ Lücken & Halluzinationen
def test_inserted_word_fills_gap_when_confirmed():
    # Fenster 2 hat ein Wort mehr in der Überlappung → wird committet,
    # wenn Fenster 3 es bestätigt.
    st = TextStitcher()
    st.add_chunk("eins zwei vier".split())
    st.add_chunk("eins zwei drei vier fünf".split())
    st.add_chunk("zwei drei vier fünf sechs".split())
    assert "drei" in st.committed

def test_hallucination_dropped():
    # Ein Wort, das nur EINMAL erscheint und nie bestätigt wird, fliegt
    # nach _MAX_MISSES verpassten Bestätigungen raus.
    st = TextStitcher()
    st.add_chunk("eins zwei drei".split())
    st.add_chunk("zwei drei vier".split())     # "eins" zum ersten Mal verpasst
    st.add_chunk("drei vier fünf".split())     # "eins" zum zweiten Mal verpasst
    st.flush()
    assert "eins" not in st.committed
    assert st.stream_text.split() == ["zwei", "drei", "vier", "fünf"]

def test_missing_word_stretches_stream():
    # Spätere Fenster lassen bereits Bestätigtes aus — der Strom behält es
    # (der Strom wird "gestretched", nie rückwärts beschnitten).
    st = TextStitcher()
    st.add_chunk("eins zwei drei vier".split())
    st.add_chunk("zwei drei vier fünf".split())    # "eins" fehlt
    st.add_chunk("drei vier fünf sechs".split())   # "eins","zwei" fehlen
    st.flush()
    got = st.stream_text.split()
    assert got == ["zwei", "drei", "vier", "fünf", "sechs"]


def test_no_position_slippage_on_refused_substitution():
    """Nutzer-gemeldeter Fall: 'geworden' ↔ 'gekommen' (sim 0.625 < 0.75).
    Die Wortstelle darf NICHT verrutschen oder doppelt auftauchen; beide
    Lesarten verschmelzen zu EINEM Slot, und die Mehrheit der Fenster
    entscheidet die Lesart."""
    assert similarity("geworden", "gekommen") < 0.75
    st = TextStitcher()
    st.add_chunk("Man ist ins kalte Wasser geworden".split())
    st.add_chunk("Man ist ins kalte Wasser gekommen hat man gefragt".split())
    # Genau EIN Slot für die Wortstelle (verschmolzen, kein Duplikat):
    slots = [it for it in st._stream if it.word in ("geworden", "gekommen")]
    assert len(slots) == 1, f"Varianten nicht verschmolzen: {[it.word for it in st._stream]}"
    assert [it.word for it in st._stream].index(slots[0].word) == 5  # direkt nach 'Wasser'
    # Weitere Fenster sehen 'geworden' → Mehrheit kippt auf die korrekte Lesart:
    st.add_chunk("Wasser geworden hat man gefragt wo".split())
    st.add_chunk("geworden hat man gefragt wo ist".split())
    st.flush()
    committed = st.committed
    assert committed.count("geworden") == 1
    assert "gekommen" not in committed
    assert committed == "Man ist ins kalte Wasser geworden hat man gefragt wo ist".split()


# ------------------------------------------------------------------ Rückwirkende Korrektur (weiche Zone)
def test_retroactive_correction_in_soft_zone():
    # sim("Tänzer","Fenster") ist klein (~0.46) — echte rückwirkende
    # Korrektur passiert bei langen Compounds, wie vom Nutzer beschrieben:
    sim = similarity("Kontext-Fit-Tänzer", "Kontext-Fit-Fenster")
    assert sim >= 0.75, f"Prämisse: sim={sim:.2f}"
    st = TextStitcher(soft_tail=10)
    st.add_chunk("eins zwei Kontext-Fit-Tänzer vier".split())
    st.add_chunk("zwei Kontext-Fit-Tänzer vier fünf".split())
    assert "Kontext-Fit-Tänzer" in st.committed
    # EIN späteres Fenster mit "Kontext-Fit-Fenster" kippt die Mehrheit
    # noch NICHT (2:1 Stimmen) — einzelne schlechte Sichtungen überschreiben
    # nichts (ROVER-Mehrheitsvotum):
    st.add_chunk("Kontext-Fit-Fenster vier fünf sechs".split())
    assert st.committed[1] == "Kontext-Fit-Tänzer"
    assert st.last_corrections == []
    # Erst das zweite Votum für die Variante (Gleichstand → neueste Sichtung)
    # kippt das Wort rückwirkend:
    st.add_chunk("Kontext-Fit-Fenster vier fünf sechs sieben".split())
    assert st.committed[1] == "Kontext-Fit-Fenster"
    assert ("Kontext-Fit-Tänzer", "Kontext-Fit-Fenster") in st.last_corrections

def test_old_words_frozen_outside_soft_zone():
    # Außerhalb der weichen Zone bleibt ein Wort endgültig stehen.
    st = TextStitcher(soft_tail=2)
    st.add_chunk("eins zwei Kontext-Fit-Tänzer vier".split())
    st.add_chunk("zwei Kontext-Fit-Tänzer vier fünf".split())   # bestätigt
    # Vor diesem Aufruf: 4 Wörter im Strom → Zone = letzte 2 (vier, fünf).
    # "Kontext-Fit-Tänzer" (Index 1) liegt außerhalb → gefroren.
    st.add_chunk("Kontext-Fit-Fenster vier fünf sechs".split())
    assert st.committed[1] == "Kontext-Fit-Tänzer"
    assert st.last_corrections == []

def test_correction_flips_only_within_zone():
    # Innerhalb der Zone gewinnt die Mehrheit; Gleichstand → neueste Sichtung.
    st = TextStitcher(soft_tail=5)
    st.add_chunk("Tänzer läuft".split())
    st.add_chunk("Fenster läuft".split())   # 1:1 Gleichstand → neueste: Fenster
    st.add_chunk("Fenster läuft weiter".split())  # Mehrheit Fenster
    assert st.committed[0] == "Fenster"

def test_single_deviant_reading_cannot_override():
    # Zwei korrekte Sichtungen, dann eine abweichende: kein Überschreiben.
    st = TextStitcher(soft_tail=10)
    st.add_chunk("das Fenster bleibt".split())
    st.add_chunk("das Fenster bleibt offen".split())   # bestätigt
    st.add_chunk("das Tänzer bleibt offen weiter".split())  # schlechte Sichtung
    assert "Fenster" in st.committed
    assert not any(new == "Tänzer" for _old, new in st.last_corrections)


# ------------------------------------------------------------------ Robustheit über viele Fenster
def test_long_sliding_window_reconstructs_reference():
    """Simulation: 30-Wörter-Referenz, Fenster à 10 Wörter, Schritt 3,
    mit realistischem Rauschen (ein Wort gelöscht, ein Wort substituiert,
    eine Halluzination eingefügt pro Fenster)."""
    ref = ("heute zeige ich euch den kontext fit fenster algorithmus für "
           "live untertitel sieben überlappende fenster sehen alle dasselbe "
           "audio der fitter synchronisiert die texte ströme wort für wort "
           "ähnliche wörter werden nach ähnlichkeit gefittet").split()

    def noisy(window, seed):
        import random
        rng = random.Random(seed)
        out = list(window)
        if len(out) > 4 and rng.random() < 0.7:
            out.pop(rng.randrange(len(out)))          # Wort fehlt
        if len(out) > 2 and rng.random() < 0.5:
            i = rng.randrange(len(out))
            out[i] = out[i][:-2] + "xx"               # verschrieben
        if rng.random() < 0.4:
            out.insert(rng.randrange(len(out) + 1), "hmm")  # Halluzination
        return out

    ctx, step = 10, 3
    st = TextStitcher()
    pos = 0
    seed = 0
    while pos < len(ref):
        window = ref[max(0, pos + step - ctx): pos + step]
        st.add_chunk(noisy(window, seed)); seed += 1
        pos += step
    st.flush()

    # Der Strom muss die Referenz rekonstruieren (bis auf Fensterrand-Effekte).
    got = st.stream_text.split()
    # WER klein gegen die Referenz — tolerant, aber deutlich besser als Kaputt.
    import jiwer
    wer = jiwer.wer(" ".join(ref), " ".join(got))
    assert wer < 0.25, f"WER zu hoch: {wer:.2f}\nGOT: {' '.join(got)}"


# ------------------------------------------------------------------ Edge cases
def test_empty_chunk_ignored():
    st = TextStitcher()
    assert st.add_chunk([]) == []
    st.add_chunk(["a"])
    st.add_chunk([])
    assert st.pending == ["a"]

def test_flush_commits_pending():
    st = TextStitcher()
    st.add_chunk(["a", "b"])
    st.flush()
    assert st.committed == ["a", "b"]
    assert st.pending == []
