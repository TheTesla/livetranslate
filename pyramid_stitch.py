"""pyramid_stitch.py — Zeitbasierter Pyramiden-Stitcher.

Architektur (vom Entwickler vorgegeben):
  - Das Stitching darf das Ergebnis NIE verschlechtern.
  - Es wird immer die gesamte Abdeckungspyramide verarbeitet (Enden ggf.
    beschnitten).
  - Der älteste Text im Context-Window hat die höchste Abdeckung und damit
    das höchste Vertrauen.

Prinzip: Zeit ist der Anker. Bei 3 s update / 24 s context überdecken sich
8 Fenster; ein 3-s-Scheibchen Audio wird vom jüngsten Fenster 1× gesehen,
vom nächstälteren 2×, … vom ältesten 8× (Abdeckungspyramide). Jedes Wort
bekommt von Whisper einen Wort-Zeitstempel und wird dem Scheibchen (Slot)
zugeordnet, in das sein Zeitpunkt fällt.

Ein Slot sammelt Sichtungen, bis ALLE Fenster, die sein Audio je sehen
können, dagewesen sind (das sind context/update = K Stück). Dann wird das
Mehrheitsvotum über die Sichtungen gebildet, das Ergebnis emittiert — und
der Slot ist für immer abgeschlossen. Emittierter Text kann deshalb per
Konstruktion nicht mehr verschlechtert werden: keine spätere Information
erreicht ihn.

Die Enden der Pyramide werden beschnitten: der jüngste Bereich (1 Sichtung)
bleibt als "pending" sichtbar, wird aber erst emittiert, wenn seine
Abdeckung vollständig ist. Slots ohne ausreichende Sichtungen (Fenster-
rand-Artefakte, Stille) werden leer abgeschlossen.

Kein globales Text-Alignment nötig — damit entfallen strukturell:
Anker-Fehler (Strom als Referenz), Positions-Verrutschen, und die
Ähnlichkeits-Kollisionen benachbarter Wörter. Varianten derselben
Wortstelle ("geworden"/"gekommen") werden nur INNERHALB eines Slots
geclustert — der Vergleichsraum ist klein und lokal.

Schnittstelle:
    ps = PyramidStitcher(context_s=24, update_s=3)
    emitted = ps.add_window(words, win_end, win_len)
        words: Liste (surface, start_rel, end_rel) relativ zum Fensteranfang
        win_end: absolute Audiozeit des Fensterendes (= Index * update)
    ps.flush()            # Ende: alle offenen Slots abschließen
    ps.committed_text     # fertiger Strom (unveränderlich)
    ps.pending_text       # aktuelle Mehrheiten der offenen Slots (Live-Anzeige)
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache


@lru_cache(maxsize=100_000)
def _norm(word: str) -> str:
    """Kleingeschrieben, nur Buchstaben/Zahlen/Bindestriche."""
    return re.sub(r"[^\w\-]", "", word.lower(), flags=re.UNICODE)


@lru_cache(maxsize=100_000)
def similarity(a: str, b: str) -> float:
    """Fuzzy-Ähnlichkeit zweier Wörter: 1.0 identisch, 0.0 disjunkt."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


@dataclass
class Sighting:
    word: str          # Oberflächenform (mit Satzzeichen)
    t: float           # absolute Audiozeit der Wortmitte
    win: int           # Fenster-Index der Sichtung


@dataclass
class Cluster:
    sightings: list[Sighting] = field(default_factory=list)

    def add(self, s: Sighting) -> None:
        self.sightings.append(s)

    @property
    def windows(self) -> set[int]:
        return {s.win for s in self.sightings}

    @property
    def t_mean(self) -> float:
        return sum(s.t for s in self.sightings) / len(self.sightings)

    def surface(self) -> str:
        """Häufigste Oberflächenform (Satzzeichen der Mehrheit)."""
        counts: dict[str, int] = {}
        for s in self.sightings:
            counts[s.word] = counts.get(s.word, 0) + 1
        return max(counts, key=counts.get)


class PyramidStitcher:
    def __init__(self, context_s: float, update_s: float,
                 sim_threshold: float = 0.75):
        self.slot_len = update_s
        self.K = max(1, int(round(context_s / update_s)))  # max. Sichtungen/Slot
        self.sim_threshold = sim_threshold
        self.slots: dict[int, list[Sighting]] = {}
        self.committed: dict[int, str] = {}   # Slot -> emitted text (unveränderlich)
        self._committed_sightings: dict[int, list[Sighting]] = {}
        self.next_window = 0

    # ------------------------------------------------------------ Zeit-Ankerung
    def _banked_index(self) -> list[tuple[float, str]]:
        """Alle bisherigen Sichtungen (offen + abgeschlossen) als
        zeit-sortierte Referenz für die Fenster-Ankerung."""
        items = [(s.t, s.word)
                 for v in list(self.slots.values()) + list(
                     [self._committed_sightings[s] for s in self._committed_sightings])
                 for s in v]
        items.sort(key=lambda x: x[0])
        return items

    def _reanchor(self, words: list[tuple[str, float, float]],
                  win_start: float, win_end: float) -> list[tuple[str, float, float]]:
        """Korrigiert die Wortzeiten eines Fensters gegen die gespeicherte
        Pyramide.

        Whisper überspringt am Fensteranfang gelegentlich Audio und komprimiert
        die folgenden Wortzeiten, bis es am Fensterende wieder einhängt — die
        Verzerrung ist affin (Skip + Stauchung). Wir matchen vertrauenswürdige
        Fensterwörter (sim >= 0.9) gegen die gespeicherten Sichtungen und
        interpolieren die Korrektur linear zwischen erstem und letztem Match.
        Ohne ausreichend Matches bleiben die Zeiten unverändert.
        """
        banked = self._banked_index()
        if len(banked) < 8:
            return words
        banked_t = [t for t, _ in banked]

        pairs: list[tuple[float, float]] = []   # (window_t, delta)
        for surface, s_rel, e_rel in words:
            w_t = win_start + (s_rel + e_rel) / 2.0
            pos = bisect.bisect_left(banked_t, w_t - 3.0)
            best_delta = None
            best_sim = 0.0
            for k in range(pos, len(banked)):
                if banked_t[k] > w_t + 3.0:
                    break
                sim = similarity(surface, banked[k][1])
                if sim >= 0.9 and sim > best_sim:
                    best_sim = sim
                    best_delta = banked_t[k] - w_t
            if best_delta is not None:
                pairs.append((w_t, best_delta))

        if len(pairs) < 3:
            return words

        # Robuster konstanter Offset: Median aller Match-Deltas. Ein affiner
        # Fit (Skip + Stauchung) reagierte in Tests fatal auf einzelne
        # Fehl-Matches — der Median ignoriert sie.
        deltas = sorted(d for _, d in pairs)
        delta = deltas[len(deltas) // 2]
        if abs(delta) > 8.0:
            return words

        return [(surface, s_rel + delta, e_rel + delta)
                for surface, s_rel, e_rel in words]

    # ------------------------------------------------------------ Eingabe
    @staticmethod
    def _merge_tokens(words):
        """Whisper-Tokenizer zerlegt Komposita in Stücke ("Kontext", "-Feed",
        "-Fenster"). Stücke, die mit Bindestrich/Apostroph beginnen oder nur
        aus Satzzeichen bestehen, werden mit dem vorherigen Wort verschmolzen
        (Oberfläche + Zeitspanne) — sonst landen die Stücke in verschiedenen
        Slots und die Zeiten passen fensterübergreifend nicht mehr.
        """
        merged: list[list] = []   # [surface, start, end]
        for surface, s, e in words:
            piece = surface.strip()
            if (merged
                    and (piece.startswith(("-", "'", "’", "„"))
                         or (piece and not any(c.isalnum() for c in piece)))):
                merged[-1][0] += piece
                merged[-1][2] = e
            else:
                merged.append([surface, s, e])
        return [(w, s, e) for w, s, e in merged]

    def add_window(self, words, win_end: float, win_len: float) -> list[str]:
        """Neues Fenster-Ergebnis einspeisen.

        words: Liste von (surface, start_rel, end_rel) — Zeiten relativ zum
               Fensteranfang. win_end = absolute Audiozeit des Fensterendes.
        Returns: neu emittierter Text je abgeschlossener Slot (in Slot-
                 Reihenfolge, leere Slots fehlen).
        """
        i = self.next_window
        self.next_window += 1
        win_start = win_end - win_len
        words = self._merge_tokens(words)
        words = self._reanchor(words, win_start, win_end)
        for surface, s_rel, e_rel in words:
            t_abs = win_start + (s_rel + e_rel) / 2.0
            if t_abs < 0:
                continue
            slot = int(t_abs // self.slot_len)
            self.slots.setdefault(slot, []).append(Sighting(surface, t_abs, i))

        # Slots abschließen, deren Abdeckung vollständig ist. Zählunabhängig
        # über die absolute Audiozeit: Slot s (Audio [(s)*u, (s+1)*u)) wird
        # von allen Fenstern mit win_end in [(s+1)*u, (s+1)*u + context]
        # gesehen. Das erste Fenster, das ihn NICHT mehr sieht, hat
        # win_end >= (s+1)*u + context + u = (s+K+1)*u. Sobald so ein Fenster
        # ankommt, kann keine Sichtung mehr folgen → abschließen.
        emitted: list[str] = []
        for s in sorted(list(self.slots.keys())):
            if win_end >= (s + self.K + 1) * self.slot_len:
                text = self._consensus(s)
                self.committed[s] = text
                self._committed_sightings[s] = self.slots[s]
                emitted.append(text)
                del self.slots[s]
        return [e for e in emitted if e]

    def flush(self) -> list[str]:
        """Ende des Stroms: alle offenen Slots abschließen (Restvertrauen)."""
        emitted: list[str] = []
        for s in sorted(self.slots.keys()):
            text = self._consensus(s)
            self.committed[s] = text
            emitted.append(text)
        self.slots.clear()
        return [e for e in emitted if e]

    # ------------------------------------------------------------ Konsens
    def _consensus(self, slot: int) -> str:
        """Mehrheitsvotum über alle Sichtungen eines Slots.

        Sichtungen werden entlang der Zeit zu Clustern (Wortstellen)
        zusammengefasst: benachbarte Sichtungen mit sim >= threshold sind
        dieselbe Wortstelle. Gewonnen hat das Cluster mit den meisten
        EINZIGARTIGEN Fenstern (ein Fenster zählt 1×, auch wenn Whisper das
        Wort im selben Fenster doppelt synthetisiert); bei Gleichstand mehr
        Sichtungen, dann frühere Zeit. Ausgegeben wird die häufigste
        Oberflächenform des Gewinnclusters; die Reihenfolge folgt der
        mittleren Zeit der Cluster (Mehrfach-Sichtungen mitteln die
        Zeitunsicherheit).
        """
        sightings = self.slots.get(slot, [])
        if not sightings:
            return ""
        clusters: list[Cluster] = []
        for s in sorted(sightings, key=lambda x: x.t):
            best: Cluster | None = None
            best_sim = 0.0
            for cl in clusters:
                sim = min(similarity(s.word, cl.sightings[j].word)
                          for j in (0, len(cl.sightings) - 1))
                if sim >= self.sim_threshold and sim > best_sim:
                    best, best_sim = cl, sim
            if best is not None:
                best.add(s)
            else:
                cl = Cluster()
                cl.add(s)
                clusters.append(cl)

        # Cluster mit zu wenig Rückhalt verwerfen: weniger als die Hälfte
        # der stärksten Abdeckung = Fensterrand-Artefakt (Enden beschneiden).
        if not clusters:
            return ""
        max_votes = max(len(cl.windows) for cl in clusters)
        keep = [cl for cl in clusters if len(cl.windows) * 2 >= max_votes]
        keep.sort(key=lambda cl: cl.t_mean)
        return " ".join(cl.surface() for cl in keep)

    # ------------------------------------------------------------ Auslesen
    @property
    def committed_text(self) -> str:
        return " ".join(self.committed[s] for s in sorted(self.committed)
                        if self.committed[s])

    def pending_text(self) -> str:
        """Aktuelle Mehrheiten der noch offenen Slots (Live-Vorschau)."""
        parts = []
        for s in sorted(self.slots.keys()):
            text = self._consensus(s)
            if text:
                parts.append(text)
        return " ".join(parts)
