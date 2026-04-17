# rag/claim_extractor.py
"""
Extract factual claims made by the salesperson from a conversation transcript.

Supports Arabic-indic digits (٠-٩), Western digits, and mixed text.
Never raises — always returns a list (empty if no claims found).

Input:
    transcript: list of {speaker, text, turn_number} dicts

Output:
    list of {turn_number, claim_type, claimed_value, claimed_unit, raw_text, property_hint}
"""
from __future__ import annotations

import logging
import re
from typing import Optional

_log = logging.getLogger(__name__)

# ── Arabic-indic → Western digit map ──────────────────────────────────────────
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _to_western(s: str) -> str:
    return s.translate(_AR_DIGITS)


def _parse_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = _to_western(s).replace(",", "").replace("،", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


# ── Arabic fraction words → decimal addition to the million unit ──────────────
_FRACTIONS: dict[str, float] = {
    "ونص":              0.5,
    "ونصف":             0.5,
    "وربع":             0.25,
    "وثلث":             0.333,
    "وثلاثة أرباع":     0.75,
}

# ── Regex building blocks ─────────────────────────────────────────────────────
_N = r"[٠-٩\d]+(?:[.,][٠-٩\d]+)?"   # one number (Arabic or Western)

# Price: [N] مليون[ين] [fraction]  or  N ألف
_MILLION_RE = re.compile(
    rf"(?:({_N})\s+)?(مليونين|مليون|مليار)"
    rf"(?:\s+(ونص|ونصف|وربع|وثلث|وثلاثة أرباع))?",
    re.UNICODE,
)
_THOUSAND_RE = re.compile(rf"({_N})\s*(?:ألف|الف)", re.UNICODE)

# Raw large number written out (e.g. 2,500,000 جنيه)
_RAW_PRICE_RE = re.compile(
    rf"({_N})\s*(?:جنيه|ج\.م\.|EGP)",
    re.UNICODE | re.IGNORECASE,
)

# Size: N متر / م٢ / م2 / sqm
_SIZE_RE = re.compile(
    rf"({_N})\s*(?:متر(?:\s*مربع)?|م٢|م2|sqm)",
    re.UNICODE | re.IGNORECASE,
)

# Down payment: مقدم N%  or  دفعة أولى N%
_DOWN_RE = re.compile(
    rf"(?:مقدم|دفعة\s*أولى|دفعة\s*مقدمة)\s*[:\s]*({_N})\s*%",
    re.UNICODE,
)

# Installment years: تقسيط N سنة / N سنوات / على N سنة
_INST_RE = re.compile(
    rf"(?:تقسيط|أقساط|على)\s+({_N})\s*(?:سنة|سنوات|سنين)",
    re.UNICODE,
)

# Delivery year: تسليم [سنة] 20XX  or  استلام 20XX
_DELIV_RE = re.compile(
    r"(?:تسليم|استلام)\s+(?:سنة\s+)?(20\d\d)",
    re.UNICODE,
)

# Feature keywords → canonical feature ID
_FEATURE_KEYWORDS: dict[str, str] = {
    "حمام سباحة": "swimming_pool",
    "مسبح":       "swimming_pool",
    "نادي صحي":   "gym",
    "نادي رياضي": "gym",
    "جيم":        "gym",
    "أمن 24":     "security",
    "أمن":        "security",
    "حراسة":      "security",
    "جراج":       "parking",
    "موقف":       "parking",
    "مصعد":       "elevator",
    "حديقة":      "garden",
    "شاطئ":       "beach",
    "شاطئ خاص":   "private_beach",
}


# ── Price parsing ──────────────────────────────────────────────────────────────

def _extract_price_egp(text: str) -> Optional[float]:
    """
    Try to parse a price from Arabic text.
    Handles: "مليون ونص", "3 مليون", "مليونين", "500 ألف", "2,500,000 جنيه".
    Returns EGP amount as float, or None if no price found.
    """
    m = _MILLION_RE.search(text)
    if m:
        num_str  = m.group(1)    # optional number before مليون (e.g., "3")
        unit     = m.group(2)    # "مليون", "مليونين", "مليار"
        frac_key = m.group(3)    # "ونص" etc. (or None)

        if unit == "مليونين":
            base = 2_000_000.0
        elif unit == "مليار":
            base = (_parse_float(num_str) or 1.0) * 1_000_000_000.0
        else:  # مليون
            base = (_parse_float(num_str) or 1.0) * 1_000_000.0

        frac = _FRACTIONS.get(frac_key or "", 0.0) * 1_000_000.0
        return base + frac

    m = _THOUSAND_RE.search(text)
    if m:
        val = _parse_float(m.group(1))
        if val is not None:
            return val * 1_000.0

    m = _RAW_PRICE_RE.search(text)
    if m:
        val = _parse_float(m.group(1))
        if val is not None and val >= 100_000:   # must be a realistic property price
            return val

    return None


# ── Property hint extraction ───────────────────────────────────────────────────

def _find_property_hint(text: str) -> str:
    """
    Scan text for known property name/keywords.
    Returns best-matching property ID string, or "" if nothing found.
    Never raises.
    """
    try:
        from rag.structured_store import load_properties
        props = load_properties()
    except Exception:
        return ""

    best_id    = ""
    best_score = 0

    text_lower = text.lower()

    for prop in props:
        score = 0

        # Arabic name exact substring match — highest signal
        name_ar = prop.get("name_ar", "")
        if name_ar and name_ar in text:
            score += 10

        # English name (case-insensitive)
        name_en = prop.get("name_en", "")
        if name_en and name_en.lower() in text_lower:
            score += 8

        # Arabic keywords
        for kw in prop.get("keywords_ar", []):
            if kw and kw in text:
                score += 3

        # English keywords
        for kw in prop.get("keywords_en", []):
            if kw and kw.lower() in text_lower:
                score += 2

        if score > best_score:
            best_score = score
            best_id    = prop.get("id", "")

    return best_id


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_salesperson_claims(transcript: list[dict]) -> list[dict]:
    """
    Extract factual claims from salesperson turns in a conversation transcript.

    Each transcript entry must have at minimum:
        {
            "speaker":      "salesperson" | "vc" | ...,
            "text":         str,
            "turn_number":  int,
        }

    Returns a list of claim dicts:
        {
            "turn_number":    int,
            "claim_type":     "price" | "size" | "down_payment" |
                              "installment_years" | "delivery" | "feature",
            "claimed_value":  str,
            "claimed_unit":   str,
            "raw_text":       str,   # the full turn text
            "property_hint":  str,   # best-guess property ID from keyword scan
        }
    """
    claims: list[dict] = []

    try:
        # Pre-compute a global property hint from ALL salesperson turns combined.
        # This handles the common pattern where the property name is mentioned once
        # ("عندنا شقق في مدينتي") and prices are given in a later turn without
        # repeating the property name.
        _sp_speakers = {"salesperson", "sales", "مندوب", "بائع"}
        _full_sp_text = " ".join(
            (t.get("text") or "")
            for t in transcript
            if t.get("speaker", "") in _sp_speakers
        )
        _global_hint = _find_property_hint(_full_sp_text)
        _log.debug("[ClaimExtractor] global_hint=%r from full salesperson text", _global_hint)

        for turn in transcript:
            speaker = turn.get("speaker", "")
            if speaker not in _sp_speakers:
                continue

            text = (turn.get("text") or "").strip()
            if not text:
                continue

            turn_num  = int(turn.get("turn_number", 0))
            # Prefer per-turn hint (most specific), fall back to global hint
            prop_hint = _find_property_hint(text) or _global_hint

            def _claim(ctype: str, value: str, unit: str) -> dict:
                return {
                    "turn_number":   turn_num,
                    "claim_type":    ctype,
                    "claimed_value": value,
                    "claimed_unit":  unit,
                    "raw_text":      text,
                    "property_hint": prop_hint,
                }

            # ── Price ──────────────────────────────────────────────────────
            price = _extract_price_egp(text)
            if price is not None:
                claims.append(_claim("price", str(int(price)), "EGP"))

            # ── Size ───────────────────────────────────────────────────────
            seen_sizes: set[int] = set()
            for m in _SIZE_RE.finditer(text):
                val = _parse_float(m.group(1))
                if val is not None and int(val) not in seen_sizes:
                    seen_sizes.add(int(val))
                    claims.append(_claim("size", str(int(val)), "sqm"))

            # ── Down payment ───────────────────────────────────────────────
            for m in _DOWN_RE.finditer(text):
                val = _parse_float(m.group(1))
                if val is not None:
                    claims.append(_claim("down_payment", str(int(val)), "percent"))

            # ── Installment years ──────────────────────────────────────────
            for m in _INST_RE.finditer(text):
                val = _parse_float(m.group(1))
                if val is not None:
                    claims.append(_claim("installment_years", str(int(val)), "years"))

            # ── Delivery year ──────────────────────────────────────────────
            for m in _DELIV_RE.finditer(text):
                claims.append(_claim("delivery", m.group(1), "year"))

            # ── Feature claims (deduplicate by feature ID) ─────────────────
            seen_features: set[str] = set()
            for keyword, feature_id in _FEATURE_KEYWORDS.items():
                if keyword in text and feature_id not in seen_features:
                    seen_features.add(feature_id)
                    claims.append(_claim("feature", feature_id, "feature"))

    except Exception as exc:
        _log.error("[ClaimExtractor] Unexpected error: %s", exc)

    return claims
