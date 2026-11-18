"""Text for the TTS: normalisation, sentence/chunk split, stress marks (RUAccent) and the user's stress dictionary.

Stress notation inside the studio: '+' BEFORE the stressed vowel (like RUAccent and Speech Studio: молок+о).
For Qwen3-TTS ru-stress the '+' is converted to the combining acute U+0301 AFTER the vowel (молоко́) - that is the
notation the model was trained to follow.
"""
import re
import threading

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
ACUTE = "́"
_WORD = re.compile(r"[А-Яа-яЁё+́]+(?:-[А-Яа-яЁё+́]+)*")
_ACC = None
_ACC_LOCK = threading.Lock()


# ---------------------------------------------------------------- normalisation
def _num_ru(m):
    try:
        from num2words import num2words
        return num2words(int(m.group(0)), lang="ru")
    except Exception:
        return m.group(0)


def normalise(text):
    t = text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    t = re.sub(r"[«»„“”]", '"', t)
    t = t.replace("…", "...").replace("–", "—").replace(" - ", " — ")
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\d+", _num_ru, t)
    lines = []
    for ln in t.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ln[-1] not in ".!?…:;\"—)":            # paragraph without a full stop: add one (keeps intonation)
            ln += "."
        lines.append(ln)
    return "\n".join(lines)


# ---------------------------------------------------------------- split
def sentences(text):
    """Split into sentences after . ! ? ... when the next sentence starts (space + capital / quote / dash)."""
    out = []
    for para in text.split("\n"):
        parts = re.split(r'(?<=[.!?…])["»)]?\s+(?=["«—\-(]?\s*[А-ЯЁA-Z0-9])', para)
        # re.split dropped the closing quote, put it back
        pos = 0
        for p in parts:
            i = para.find(p, pos)
            j = i + len(p)
            while j < len(para) and para[j] in '"»)':
                j += 1
            out.append(para[i:j].strip())
            pos = j
    return [s for s in out if re.search(r"[А-Яа-яЁёA-Za-z]", s)]


def chunks(text, max_chars=220, min_chars=50):
    """Sentences packed into synthesis blocks: short ones glued, long ones split at ; : , —."""
    res = []
    for s in sentences(normalise(text)):
        while len(s) > max_chars:
            cut = -1
            for sep in ("; ", ": ", " — ", ", "):
                k = s.rfind(sep, int(max_chars * 0.4), max_chars)
                if k > cut:
                    cut = k + len(sep.rstrip())
                    break
            if cut <= 0:
                cut = s.rfind(" ", 0, max_chars)
            res.append(s[:cut].strip())
            s = s[cut:].strip(" ,")
            if s:
                s = s[0].upper() + s[1:] if res[-1].endswith((".", "!", "?")) else s
        if s:
            if res and len(res[-1]) < min_chars and len(res[-1]) + len(s) + 1 <= max_chars:
                res[-1] = res[-1] + " " + s
            else:
                res.append(s)
    return res


# ---------------------------------------------------------------- stress
def load_accentizer(workdir, log=print):
    global _ACC
    with _ACC_LOCK:
        if _ACC is None:
            from ruaccent import RUAccent
            acc = RUAccent()
            acc.load(omograph_model_size="turbo3.1", use_dictionary=True, workdir=workdir)
            _ACC = acc
            log("  RUAccent загружен")
        return _ACC


def homographs(acc):
    """Set of words RUAccent treats as homographs (стрелки/стрелки, замок/замок): worth a manual check."""
    for attr in ("omographs", "omograph_dict"):
        d = getattr(acc, attr, None)
        if isinstance(d, dict):
            return set(d)
    return set()


def strip_marks(s):
    return s.replace("+", "").replace(ACUTE, "")


def plus_to_acute(s):
    """'+' before a vowel -> vowel + U+0301; any other '+' is dropped (the TTS would read it as «плюс»)."""
    s = re.sub(r"\+([" + VOWELS + "])", lambda m: m.group(1) + ACUTE, s)
    return s.replace("+", "")


def acute_to_plus(s):
    return re.sub(r"([" + VOWELS + "])" + ACUTE, lambda m: "+" + m.group(1), s)


def n_vowels(w):
    return sum(c in VOWELS for c in w)


def stress_index(w):
    """index (0-based, among vowels) of the vowel after '+', or None. ё is always stressed."""
    k = 0
    for i, c in enumerate(w):
        if c == "+" and i + 1 < len(w) and w[i + 1] in VOWELS:
            return k
        if c in VOWELS:
            k += 1
    clean = strip_marks(w)
    if "ё" in clean.lower():
        return [c for c in clean.lower() if c in VOWELS].index("ё")
    return None


def _lower_titles(s):
    """RUAccent stresses Capitalised words worse: lowercase them (except all-caps), remember to restore."""
    return re.sub(r"\b([А-ЯЁ])([а-яё]+)", lambda m: m.group(1).lower() + m.group(2), s)


def _restore_case(orig, acc):
    """copy letter case from orig onto acc (acc = orig + '+' marks, maybe ё instead of е)."""
    out, j = [], 0
    for c in acc:
        if c == "+":
            out.append(c)
            continue
        o = orig[j] if j < len(orig) else c
        out.append(c.upper() if o.isupper() else c)
        j += 1
    return "".join(out)


_W1 = re.compile(r"[А-Яа-яЁё+́]+")


def _key(w):
    return strip_marks(w).lower().replace("ё", "е")


def _apply_yo(orig, model_word):
    """take ё from the model's word where the original has е (same letters otherwise)."""
    mw = strip_marks(model_word)
    if len(mw) != len(orig):
        return orig
    return "".join(("Ё" if o.isupper() else "ё") if (m in "ёЁ" and o in "еЕ") else o for o, m in zip(orig, mw))


def accentuate(text, acc, user_dict=None):
    """-> (text with '+' marks, list of words [{w, stressed, src, homograph, idx}]).
    RUAccent marks the text; its marks are moved back onto the ORIGINAL words by alignment (RUAccent drops quotes
    and may change spacing/case). Manual '+' / U+0301 in the text and the user dictionary win."""
    user_dict = {_key(k): v for k, v in (user_dict or {}).items()}
    text = acute_to_plus(text)
    spans = [(m.start(), m.end(), m.group(0)) for m in _W1.finditer(text)]
    plain = strip_marks(text)
    model_words = []
    if acc is not None:
        try:
            # RUAccent loses the marks of the words after quotes/brackets: give it the text without them
            for_model = re.sub(r"\s+", " ", re.sub(r"[\"«»„“”()\[\]{}*_]", " ", plain))
            model_words = _W1.findall(acc.process_all(_lower_titles(for_model)))
        except Exception:
            model_words = []
    # align original words to the model's words (same order, same letters up to ё/е and case)
    mapped, j = {}, 0
    for k, (_, _, w) in enumerate(spans):
        key = _key(w)
        for look in range(j, min(j + 4, len(model_words))):
            if _key(model_words[look]) == key:
                mapped[k] = model_words[look]
                j = look + 1
                break
    hom = homographs(acc) if acc is not None else set()
    out, words, pos = [], [], 0
    for k, (a, b, w) in enumerate(spans):
        clean = strip_marks(w)
        key = _key(w)
        if "+" in w:
            res, src = w, "manual"
        elif key in user_dict and user_dict[key]:
            ud = user_dict[key]
            idx = stress_index(ud)
            base = _apply_yo(clean, ud)
            res, src = (set_stress(base, idx) if idx is not None else base), "dict"
        elif k in mapped:
            mw = mapped[k]
            base = _apply_yo(clean, mw)
            idx = stress_index(mw)
            res = set_stress(base, idx) if idx is not None and n_vowels(base) >= 2 else base
            src = "auto"
        else:
            res, src = clean, "none"
        if src != "manual" and "ё" in strip_marks(res).lower():
            res = strip_marks(res)            # ё is always stressed: no extra mark
        if n_vowels(strip_marks(res)) < 2 and src != "manual":
            res = strip_marks(res)
        out.append(text[pos:a])
        out.append(res)
        pos = b
        if n_vowels(clean) >= 2:
            words.append(dict(w=strip_marks(res), stressed=res, src=src, homograph=key in hom,
                              idx=stress_index(res)))
    out.append(text[pos:])
    return "".join(out), words


def set_stress(word, vowel_idx):
    """word (any marks) -> same word with '+' before vowel number vowel_idx."""
    clean = strip_marks(word)
    out, k = [], 0
    for c in clean:
        if c in VOWELS:
            if k == vowel_idx:
                out.append("+")
            k += 1
        out.append(c)
    return "".join(out)


if __name__ == "__main__":
    import sys
    txt = open(sys.argv[1], encoding="utf-8-sig").read()
    cs = chunks(txt)
    print(len(cs), "блоков; первые:")
    for c in cs[:8]:
        print(" -", c)
