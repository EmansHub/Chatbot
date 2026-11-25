"""Arabic text clean-up used for matching (search).

The SAME function must be used on the document text before it is embedded and on the
user's question before it is embedded, so both sides are cleaned the same way.
The text that is stored and shown to the assistant stays as originally written.
Text with no Arabic letters is returned unchanged, so English is not affected.
"""
import re
import unicodedata

_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

# Diacritics (tashkeel), Quranic marks, dagger alef
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
# Tatweel (the stretching line), zero-width and direction marks that Word and websites add
_NOISE = re.compile(r"[\u0640\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")

_MAP = str.maketrans({
    # alef forms -> bare alef
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    # alef maqsura, Persian ya, hamza on ya -> ya
    "ى": "ي", "ی": "ي", "ئ": "ي",
    # hamza on waw -> waw
    "ؤ": "و",
    # ta marbuta -> ha
    "ة": "ه",
    # Persian kaf -> Arabic kaf
    "ک": "ك",
    # Arabic punctuation -> plain punctuation
    "،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ",",
    # Arabic-Indic and Persian digits -> 0-9
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
})


def normalize_for_embedding(text: str) -> str:
    if not _ARABIC_LETTER.search(text):
        return text                                   # no Arabic letters: leave it exactly as it is
    text = unicodedata.normalize("NFKC", text)        # joined forms (for example the lam-alef ligature) -> normal letters
    text = _DIACRITICS.sub("", text)
    text = _NOISE.sub("", text)
    text = text.translate(_MAP)
    return re.sub(r"[ \t]+", " ", text)               # collapse repeated spaces, keep line breaks
