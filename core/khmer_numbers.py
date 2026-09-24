"""Arabic / Western numbers → Khmer digits (captions) + spoken words (TTS).

Place values (count right → left), matching common Khmer teaching charts:
  រាយ · ដប់ · រយ · ពាន់ · ម៉ឺន · សែន · លាន
"""

from __future__ import annotations

import re

# Caption / on-screen digits
_ARABIC_TO_KHMER = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")
_FULLWIDTH_TO_KHMER = str.maketrans("０１２３４５６７８９", "០១២៣៤៥៦៧៨៩")
_KHMER_TO_ASCII = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")

_ONES = [
    "",
    "មួយ",
    "ពីរ",
    "បី",
    "បួន",
    "ប្រាំ",
    "ប្រាំមួយ",
    "ប្រាំពីរ",
    "ប្រាំបី",
    "ប្រាំបួន",
]
_TENS = {
    1: "ដប់",
    2: "ម្ភៃ",
    3: "សាមសិប",
    4: "សែសិប",
    5: "ហាសិប",
    6: "ហុកសិប",
    7: "ចិតសិប",
    8: "ប៉ែតសិប",
    9: "កៅសិប",
}

# Western / Khmer digit runs (optional thousands commas, optional decimal)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])"  # not inside Latin words (e.g. H2O keeps as-is if letter-bound)
    r"(?:"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 1,234 or 1,234.56
    r"|\d+(?:\.\d+)?"  # 12 or 12.5
    r"|[០-៩]{1,3}(?:[០-៩]{3})*(?:[.][០-៩]+)?"  # already Khmer digits
    r")"
)


def to_khmer_digits(text: str) -> str:
    """Replace Western / fullwidth digits with Khmer digits ០–៩."""
    if not text:
        return text
    t = text.translate(_ARABIC_TO_KHMER).translate(_FULLWIDTH_TO_KHMER)
    return t


def _digit_word(d: int) -> str:
    if 0 <= d <= 9:
        return _ONES[d] if d else "សូន្យ"
    return number_to_khmer_words(d)


def _under_100(n: int) -> str:
    if n <= 0:
        return ""
    if n < 10:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    if tens == 1:
        # 10=ដប់, 12=ដប់ពីរ
        return "ដប់" + (_ONES[ones] if ones else "")
    return _TENS.get(tens, "") + (_ONES[ones] if ones else "")


def number_to_khmer_words(n: int) -> str:
    """
    Integer → spoken Khmer (chart style).
    Examples: 1→មួយ, 12→ដប់ពីរ, 123→មួយរយម្ភៃបី
    """
    if n == 0:
        return "សូន្យ"
    if n < 0:
        return "ដក" + number_to_khmer_words(-n)

    parts: list[str] = []

    if n >= 1_000_000:
        mil, n = divmod(n, 1_000_000)
        parts.append(number_to_khmer_words(mil) + "លាន")
    if n >= 100_000:
        saen, n = divmod(n, 100_000)
        parts.append(_digit_word(saen) + "សែន")
    if n >= 10_000:
        meun, n = divmod(n, 10_000)
        parts.append(_digit_word(meun) + "ម៉ឺន")
    if n >= 1_000:
        poan, n = divmod(n, 1_000)
        parts.append(_digit_word(poan) + "ពាន់")
    if n >= 100:
        roy, n = divmod(n, 100)
        parts.append(_digit_word(roy) + "រយ")
    if n > 0:
        parts.append(_under_100(n))

    # Space between large place groups (as in teaching charts)
    if len(parts) >= 3:
        return " ".join(parts)
    return "".join(parts)


def _raw_number_to_words(raw: str) -> str:
    """Parse a matched number string → spoken Khmer."""
    s = raw.strip().translate(_KHMER_TO_ASCII).replace(",", "")
    if not s or s == ".":
        return raw
    if "." in s:
        whole, frac = s.split(".", 1)
        whole = whole or "0"
        try:
            w = number_to_khmer_words(int(whole))
        except ValueError:
            return to_khmer_digits(raw)
        # Decimal digits spoken one-by-one
        frac_words = " ".join(
            number_to_khmer_words(int(ch)) for ch in frac if ch.isdigit()
        )
        if frac_words:
            return f"{w} ចុច {frac_words}"
        return w
    try:
        return number_to_khmer_words(int(s))
    except ValueError:
        return to_khmer_digits(raw)


def expand_numbers_for_speak(text: str) -> str:
    """Turn digit runs into spoken Khmer words for clearer Edge TTS."""
    if not text:
        return text
    return _NUMBER_RE.sub(lambda m: _raw_number_to_words(m.group(0)), text)


def apply_khmer_digits_in_text(text: str) -> str:
    """Show Khmer digits on captions / Khmer script (Western 0-9 → ០-៩)."""
    return to_khmer_digits(text)
