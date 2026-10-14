"""text_stitch.py — Füge überlappende ASR-Textströme zu einem kontinuierlichen
Text zusammen.

Verfahren (Kombination aus drei etablierten Techniken):

1. Needleman-Wunsch-Alignment (globales Sequenzalignment):
   Aligniert die Hypothese eines Fensters gegen das Ende des Stroms.
   Gaps behandeln "Wort fehlt" (Deletion) und "Wort zu viel" (Insertion)
   auf beiden Seiten; Einfügungen werden an ihrer Position im Strom
   verankert, nicht hinten angehängt.

2. Fuzzy-Ähnlichkeit statt exakter Gleichheit:
   SequenceMatcher-Ratio (Levenshtein-Verwandtschaft) auf normalisierten
   Wörtern (kleingeschrieben, ohne Satzzeichen). "Context-Fit-Fenster" und
   "Context-Fit-Tänzer" haben Ratio ≈ 0.89 und gelten als dasselbe Wort.
   UNTER der Schwelle wird eine Positionszuordnung verweigert (dann Gap) —
   sonst würde das Alignment fremde Wörter zwangszuordnen.

3. ROVER-Prinzip (Konfirmation durch wiederholte Sichtung):
   Jedes Wort im Strom hat einen Status: pending (einmal gesehen) oder
   confirmed (von einem späteren Fenster erneut gesehen). Nur bestätigte
   Wörter gehören zum finalen Text; Halluzinationen und Fensterrand-
   Artefakte fallen heraus, weil sie nie bestätigt werden. Ein pending-
   Wort, das zu oft nicht bestätigt wird, wird verworfen.

4. Weiche Zone (rückwirkende Korrektur, ROVER-Mehrheitsvotum):
   Die letzten soft_tail Wörter des Stroms bleiben korrigierbar. Jedes
   spätere Fenster, das ein Wort in dieser Zone fuzzy-matcht, gibt ein
   VOTUM für seine Schreibweise ab; der Strom zeigt die Schreibweise mit
   den meisten Stimmen (Gleichstand: neueste Sichtung). Ein einzelnes
   schlechteres Fenster kann nichts mehr überschreiben — der lange
   Kontext verbessert die Vergangenheit statistisch. Erst außerhalb der
   Zone friert ein Wort endgültig ein. Vorgenommene Änderungen stehen in
   last_corrections.

Der Strom selbst ist EINE Liste in Textreihenfolge — Bestätigen markiert
nur, Einfügen erfolgt an der alignierten Position. Dadurch bleibt die
Wortreihenfolge immer korrekt, auch wenn Wörter versetzt bestätigt werden.

Schnittstelle:
    st = TextStitcher()
    confirmed = st.add_chunk(words)   # -> neu bestätigte Wörter
    st.flush()                        # Ende: pending zwangsweise bestätigen
    st.committed / st.pending / st.stream_text
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

_GAP = -1.0                 # Gap-Strafe im Alignment (1 Gap kostet -1.0)
_NO_SUB = -2.5              # Score für "verweigerte" Substitution (< 2 Gaps)
_DEFAULT_THRESHOLD = 0.75   # sim >= threshold  ->  dasselbe Wort
_REF_ITEMS = 200            # wie viele Strom-Ende-Items maximal aligniert werden
_MAX_MISSES = 1             # pending-Wort übersteht 1 verpasste Bestätigung
_DEFAULT_SOFT_TAIL = 40     # letzte N Wörter des Stroms bleiben rückwirkend
                            # korrigierbar (weiche Zone)


@lru_cache(maxsize=100_000)
def _norm(word: str) -> str:
    """Kleingeschrieben, nur Buchstaben/Zahlen/Bindestriche."""
    return re.sub(r"[^\w\-]", "", word.lower(), flags=re.UNICODE)


@lru_cache(maxsize=100_000)
def similarity(a: str, b: str) -> float:
    """Fuzzy-Ähnlichkeit zweier Wörter: 1.0 identisch, 0.0 Disjunkt."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _align(ref: list[str], hyp: list[str], threshold: float) -> list[tuple]:
    """Needleman-Wunsch mit Fuzzy-Substitutionsscores.

    Substitution nur, wenn similarity >= threshold; sonst wird die Stelle
    zwanghaft als del+ins behandelt (_NO_SUB < 2 Gaps).

    Returns: Liste von Operationen in Stromreihenfolge:
      ('sub', ref_idx, hyp_idx)  — Positionszuordnung (immer sim >= threshold)
      ('del', ref_idx, None)     — ref-Wort ohne Hypothesen-Partner
      ('ins', None, hyp_idx)     — Hypothesen-Wort ohne ref-Partner
    """
    n, m = len(ref), len(hyp)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _GAP
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _GAP
        bt[0][j] = "ins"

    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            sim = similarity(ri, hyp[j - 1])
            s = sim * 2.0 - 1.0 if sim >= threshold else _NO_SUB
            best = prev_row[j - 1] + s
            op = "sub"
            d = prev_row[j] + _GAP
            if d > best:
                best, op = d, "del"
            e = row[j - 1] + _GAP
            if e > best:
                best, op = e, "ins"
            row[j] = best
            bt[i][j] = op

    ops: list[tuple] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            op = bt[i][j]
        else:
            op = "del" if i > 0 else "ins"
        if op == "sub":
            ops.append(("sub", i - 1, j - 1)); i -= 1; j -= 1
        elif op == "del":
            ops.append(("del", i - 1, None)); i -= 1
        else:
            ops.append(("ins", None, j - 1)); j -= 1
    ops.reverse()
    return ops


@dataclass
class _Item:
    word: str
    confirmed: bool
    miss: int = 0
    votes: dict[str, int] = field(default_factory=dict)
    last_seen: str = ""
    slot: int | None = None   # Varianten-Paar: gleicher slot => verschmelzen

    def record(self, w: str) -> tuple[str, str]:
        """Eine Sichtung w abgeben. ROVER-Mehrheit gewinnt, Gleichstand
        entscheidet die neueste Sichtung. Returns (altes Wort, neues Wort)."""
        self.votes[w] = self.votes.get(w, 0) + 1
        self.last_seen = w
        old = self.word
        best = max(self.votes.items(),
                   key=lambda kv: (kv[1], kv[0] == self.last_seen))
        self.word = best[0]
        return old, self.word


def _fuzzy_hit(word: str, hyp_indices: list[int], words: list[str]) -> int | None:
    """Index des ersten fuzzy-verwandten Hypothesen-Worts oder None.
    Nur für lange Wörter (>= 6 Zeichen) — bei kurzen überbewertet die
    Ratio 2 gemeinsame Buchstaben ("eins"/"zwei" = 0.57)."""
    if len(word) < 6:
        return None
    for j in hyp_indices:
        if similarity(word, words[j]) >= 0.5:
            return j
    return None


class TextStitcher:
    """Sammelt Fenster-Hypothesen und baut einen kontinuierlichen Wortstrom."""

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD,
                 soft_tail: int = _DEFAULT_SOFT_TAIL):
        self.threshold = threshold
        self.soft_tail = soft_tail
        self._stream: list[_Item] = []
        self.last_corrections: list[tuple[str, str]] = []

    # ------------------------------------------------------------ Öffentliche API
    def add_chunk(self, words: list[str]) -> list[str]:
        """Neue Fenster-Hypothese (Liste von Wörtern) einspeisen.

        Returns: die Wörter, die DURCH DIESEN Aufruf neu bestätigt wurden.
        """
        words = list(words)
        if not words:
            return []

        ref_items = self._stream[-_REF_ITEMS:]
        base = len(self._stream) - len(ref_items)
        ops = _align([it.word for it in ref_items], words, self.threshold)

        # --- Einfügungen an Ref-Positionen verankern (Anti-Verrutsch-Regel):
        # Ein 'del' (altes Wort ohne Partner) muss an SEINER Ref-Position
        # bleiben, auch wenn im Alignment-Pfad danach noch 'ins' folgen.
        # Sonst verrutscht das Wort ans Stromende und geht später verloren
        # ("geworden" ↔ "gekommen"-Fall). Anker: ins_before[i] = vor Ref i
        # einfügen (i == n heißt: ans Ende).
        n_ref = len(ref_items)
        ins_before: dict[int, list[int]] = {}
        matched_ref: dict[int, int] = {}       # ref-Index → hyp-Index (sub)
        anchor = 0
        for op, i, j in ops:
            if op == "ins":
                ins_before.setdefault(anchor, []).append(j)
            else:
                anchor = i + 1                 # nach Ref i → vor Ref i+1
                if op == "sub":
                    matched_ref[i] = j

        # Varianten-Fall: Das Fenster hat ein Strom-Wort NICHT wiedergefunden
        # (del), aber unmittelbar danach ein ähnliches Wort eingefügt
        # ("geworden" → "gekommen"). Dann ist das neue Wort eine Variante an
        # derselben Stelle: der Block wandert hinter das alte Wort und BEIDE
        # bekommen denselben slot-Tag (die Verschmelzung passiert unten nur
        # für solche explizit getaggten Paare — nie per Ähnlichkeits-Scan,
        # sonst verschmelzen echte Nachbarn wie "erneut"/"erkennt").
        next_slot = max((it.slot for it in self._stream if it.slot is not None),
                        default=-1) + 1
        slot_for_hyp: dict[int, int] = {}
        for i in range(n_ref):
            blk = ins_before.get(i)
            if blk and i not in matched_ref:
                if (len(ref_items[i].word) >= 6
                        and similarity(ref_items[i].word, words[blk[0]]) >= 0.5):
                    ins_before[i + 1] = blk + ins_before.get(i + 1, [])
                    del ins_before[i]
                    ref_items[i].slot = next_slot
                    slot_for_hyp[blk[0]] = next_slot
                    next_slot += 1

        # Strom in Ref-Reihenfolge neu aufbauen — Reihenfolge bleibt exakt.
        new_stream: list[_Item] = list(self._stream[:base])
        confirmed_now: list[str] = []
        corrections: list[tuple[str, str]] = []
        # Weiche Zone: die letzten soft_tail Wörter (vor diesem Fenster)
        # bleiben rückwirkend korrigierbar.
        zone_start = len(self._stream) - self.soft_tail

        def _emit_ins(hyp_indices: list[int]) -> None:
            for j in hyp_indices:
                item = _Item(words[j], confirmed=False)
                item.record(words[j])   # erste Sichtung = erste Stimme
                item.slot = slot_for_hyp.get(j)
                new_stream.append(item)

        _emit_ins(ins_before.get(0, []))       # Einfügungen vor dem ersten Ref-Wort
        for i, item in enumerate(ref_items):
            abs_idx = base + i
            if i in matched_ref:
                # 'sub': Fenster hat das Wort (fuzzy) wiedergefunden.
                if not item.confirmed:
                    # Noch nicht bestätigt: ROVER-Votum, Mehrheit/Neuestes.
                    item.record(words[matched_ref[i]])
                    item.confirmed = True
                    confirmed_now.append(item.word)
                elif abs_idx >= zone_start:
                    # Weiche Zone: Votum abgeben; die Schreibweise ändert sich
                    # nur, wenn die MEHRHEIT kippt (einzelne schlechte
                    # Sichtungen überschreiben nichts mehr).
                    old = item.word
                    item.record(words[matched_ref[i]])
                    if item.word != old:
                        corrections.append((old, item.word))
                # außerhalb der Zone: gefroren, Votum wird ignoriert
                new_stream.append(item)
            else:
                # 'del': kein exakter/fuzzy Partner (>= threshold) im Fenster.
                # ABER: Gibt es ein fuzzy-VERWANDTES Wort (>= 0.5) unter den
                # Einfügungen an dieser Stelle ODER unter den gematchten
                # Wörtern der Nachbar-Slots, ist das dieselbe Wortstelle in
                # einer anderen Lesart — KEIN Miss, plus ein Votum (das
                # ROVER-Mehrheitsvotum entscheidet später, welche Lesart
                # gilt). Ohne diese Ausnahme stirbt ein pending-Wort nach
                # 1 Miss, obwohl sein Audio noch in 6+ Fenstern liegt
                # ("geworden" ↔ "gekommen"-Fall).
                cand_j = (_fuzzy_hit(item.word,
                                     ins_before.get(i + 1, []) + ins_before.get(i, []),
                                     words))
                if cand_j is None:
                    if len(item.word) >= 6:   # Varianten-Logik nur bei langen Wörtern
                        for nb in (i - 1, i + 1):
                            j = matched_ref.get(nb)
                            if j is not None and similarity(item.word, words[j]) >= 0.5:
                                cand_j = j
                                break
                if not item.confirmed:
                    if cand_j is None:
                        item.miss += 1
                        if item.miss <= _MAX_MISSES:
                            new_stream.append(item)
                        # sonst: verwerfen (nie bestätigt, nie wieder gesehen)
                    else:
                        item.record(words[cand_j])    # Variante = Votum
                        new_stream.append(item)       # kein Miss
                else:
                    new_stream.append(item)  # bestätigt bleibt stehen
            _emit_ins(ins_before.get(i + 1, []))   # Einfügungen NACH dem Ref-Wort

        # Varianten verschmelzen: NUR Paare mit explizitem slot-Tag (von der
        # Varianten-Regel oben gesetzt). Kein Ähnlichkeits-Scan — der würde
        # echte Nachbarn ("erneut"/"erkennt", "ist"/"ins") verschmelzen.
        merged: list[_Item] = []
        for it in new_stream:
            if (merged and merged[-1].slot is not None
                    and merged[-1].slot == it.slot):
                prev = merged[-1]
                for w, v in it.votes.items():
                    prev.votes[w] = prev.votes.get(w, 0) + v
                prev.confirmed = prev.confirmed or it.confirmed
                prev.miss = max(prev.miss, it.miss)
                best = max(prev.votes.items(),
                           key=lambda kv: (kv[1], kv[0] == prev.last_seen))
                prev.word = best[0]
                continue
            merged.append(it)
        self._stream = merged
        self.last_corrections = corrections
        return confirmed_now

    def flush(self) -> list[str]:
        """Ende des Stroms: alle pending-Wörter zwangsweise bestätigen."""
        confirmed = [it.word for it in self._stream if not it.confirmed]
        for it in self._stream:
            it.confirmed = True
        return confirmed

    # ------------------------------------------------------------ Auslesen
    @property
    def committed(self) -> list[str]:
        """Finaler Strom (nur bestätigte Wörter, in Textreihenfolge)."""
        return [it.word for it in self._stream if it.confirmed]

    @property
    def pending(self) -> list[str]:
        """Noch nicht bestätigte Wörter (in Textreihenfolge)."""
        return [it.word for it in self._stream if not it.confirmed]

    @property
    def stream_text(self) -> str:
        return " ".join(self.committed)
