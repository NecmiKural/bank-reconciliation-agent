"""Remove personal data before any text leaves for a model provider, and fence payer text.

Bank account, IRD, phone and email patterns follow New Zealand formats. An IRD-shaped
number is replaced only if its mod-11 check digit is valid, so invoice and order numbers
survive.
"""

from __future__ import annotations

import re

_IRD_PRIMARY = (3, 2, 7, 6, 5, 4, 3, 2)
_IRD_SECONDARY = (7, 4, 3, 2, 5, 2, 7, 6)

_SEP = r"[\s\-\u2013.]"  # space, hyphen, en dash or dot between digit groups
_BANK_ACCOUNT = re.compile(rf"\b\d{{2}}{_SEP}?\d{{4}}{_SEP}?\d{{7}}{_SEP}?\d{{2,3}}\b")
_IRD = re.compile(rf"\b(?:\d{{2,3}}{_SEP}\d{{3}}{_SEP}\d{{3}}|\d{{8,9}})\b")
# Mobile (02x), landline (03 to 09) and toll-free (0800, 0508) numbers, written nationally or
# with +64, 0064 or (+64), with or without a (0) trunk digit, in any grouping.
_PHONE = re.compile(
    r"(?<![\w+])(?:\(?(?:\+|00)64\)?[\s-]*(?:\(0\)[\s-]*)?|\(?0)"
    r"(?:2\d|800|508|[3-9])\)?(?:[\s-]*\d){6,8}\b"
)
_EMAIL = re.compile(r"\b[\w.%+-]+@[\w-]+(?:\.[\w-]+)+\b")


def ird_check_digit(base: int) -> int | None:
    """IRD mod-11 check digit for the number without its last digit, or None if invalid."""
    digits = [int(d) for d in f"{base:08d}"]
    for weights in (_IRD_PRIMARY, _IRD_SECONDARY):
        remainder = sum(d * w for d, w in zip(digits, weights, strict=True)) % 11
        check = 0 if remainder == 0 else 11 - remainder
        if check != 10:
            return check
    return None


def ird_is_valid(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if not digits or not 10_000_000 <= int(digits) <= 150_000_000:
        return False
    number = int(digits)
    return ird_check_digit(number // 10) == number % 10


def redact(text: str) -> str:
    text = _BANK_ACCOUNT.sub("[BANK_ACCOUNT]", text)
    text = _IRD.sub(lambda m: "[IRD]" if ird_is_valid(m.group()) else m.group(), text)
    text = _PHONE.sub("[PHONE]", text)
    return _EMAIL.sub("[EMAIL]", text)


def untrusted_block(fields: dict[str, str], redacted: bool = True) -> str:
    """Payer-written fields, redacted and fenced so they read as data, never as instructions.

    Angle brackets are replaced, so text inside cannot close the fence early.
    """
    lines = []
    for name, value in fields.items():
        clean = (redact(value) if redacted else value).replace("<", "(").replace(">", ")")
        lines.append(f"{name}: {clean}")
    return "<untrusted_payer_text>\n" + "\n".join(lines) + "\n</untrusted_payer_text>"
