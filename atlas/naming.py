"""Invented toponymy.

Random syllables produce letter soup.  What makes a map feel like a place is
that names within one region *sound* related: the same consonants recur, the
same endings, the same rhythm.  So each region gets a culture -- a small
phonology plus a set of toponymic suffixes -- and every name in that region is
drawn from it.

Nothing here is a real language.  It is the impression of one, which is all a
map needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

ONSETS = ["b", "br", "c", "ch", "d", "dr", "f", "g", "gl", "gr", "h", "k",
          "l", "m", "n", "p", "r", "s", "sh", "sk", "st", "t", "th", "tr",
          "v", "w", "y", "z", "dh", "kh", "vr", "sv"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ai", "au", "ei", "ia", "ou"]
CODAS = ["", "", "", "", "n", "r", "l", "s", "th", "m", "nd", "rk", "st",
         "ng", "sk", "ld", "rn"]

SUFFIXES = {
    "region": ["ia", "mark", "land", "heim", "gard", "esse", "ora", "ath",
               "wyn", "reach", "vale", "moor", "holt"],
    "range": ["horn", "crag", "fell", "spine", "rim", "teeth", "peaks",
              "crest", "wall", "heights"],
    "water": ["sea", "ocean", "deep", "sound", "gulf", "mere", "strand"],
    "river": ["flod", "wyn", "burn", "ea", "water", "rush", "linn"],
    "town": ["burg", "ford", "mouth", "haven", "wick", "stead", "gate",
             "bridge", "holm", "thorpe"],
}


@dataclass
class Culture:
    name: str
    onsets: list
    vowels: list
    codas: list
    syllables: tuple = (2, 3)
    suffixes: dict = field(default_factory=lambda: SUFFIXES)


def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def make_cultures(seed: int, count: int = 5):
    """Carve a handful of distinct-sounding phonologies out of the inventory."""
    rng = np.random.default_rng(seed * 31 + 7)
    cultures = []
    used = set()
    for i in range(count):
        while True:
            onsets = list(rng.choice(ONSETS, size=int(rng.integers(7, 12)), replace=False))
            vowels = list(rng.choice(VOWELS, size=int(rng.integers(4, 7)), replace=False))
            codas = list(rng.choice(CODAS, size=int(rng.integers(4, 8)), replace=False))
            key = tuple(sorted(onsets)) + tuple(sorted(vowels))
            if key not in used:
                used.add(key)
                break
        cultures.append(Culture(
            name=f"culture-{i}",
            onsets=onsets,
            vowels=vowels,
            codas=codas,
            syllables=(int(rng.integers(2, 4)), int(rng.integers(2, 4)) + 1),
        ))
    return cultures


class Namer:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed * 977 + 13)
        self.cultures = make_cultures(seed)
        self._used = set()

    def word(self, culture: Culture, syllables: int | None = None) -> str:
        lo, hi = culture.syllables
        n = syllables if syllables is not None else int(self.rng.integers(lo, hi + 1))
        parts = []
        for i in range(n):
            parts.append(_pick(self.rng, culture.onsets))
            parts.append(_pick(self.rng, culture.vowels))
        # A coda closes a word.  Putting one after every syllable is what
        # turned a three-syllable root into "Paesgriagh": each coda stacked
        # onto the next onset until the name was mostly consonants.
        if self.rng.random() < 0.62:
            parts.append(_pick(self.rng, culture.codas))
        w = "".join(parts)
        if len(w) < 3:
            w += "a"
        return w.capitalize()

    def place(self, culture: Culture, kind: str = "region", suffix: bool = True) -> str:
        """A name that reads as a place, not as a random string."""
        for _ in range(40):
            # A suffixed name gets a two-syllable root, so "Svaith" + "land"
            # stays a name and does not become a paragraph.
            base = self.word(culture, syllables=2 if suffix else None)
            if suffix:
                suf = _pick(self.rng, culture.suffixes[kind])
                # Join cleanly: drop a trailing vowel before a vowel suffix.
                if base[-1].lower() in "aeiouy" and suf[0].lower() in "aeiouy":
                    base = base[:-1]
                elif base[-1].lower() == suf[0].lower():
                    base = base[:-1]
                # "Vark" + "gard" is a three-consonant pile-up.  Drop the coda
                # when the suffix begins with a consonant of its own.
                if suf[0].lower() not in "aeiouy":
                    while (len(base) > 2 and base[-1].lower() not in "aeiouy"
                           and base[-2].lower() not in "aeiouy"):
                        base = base[:-1]
                name = base + suf
            else:
                name = base
            name = name.capitalize()
            if len(name) > 13:
                continue
            if name not in self._used:
                self._used.add(name)
                return name
        return name

    def title(self, culture: Culture) -> str:
        return self.place(culture, "region", suffix=True)


def roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
            (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
            (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out or "I"
