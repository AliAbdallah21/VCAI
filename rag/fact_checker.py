# rag/fact_checker.py
"""
Hybrid fact-checker: FAISS semantic search → structured KB lookup → field comparison.

Entry points:
    check_facts(claims, properties, policies) → result dict

Internal flow per claim:
    1. FAISS search on raw_text + property_hint to identify the property
    2. Load the structured record from properties list
    3. Compare claimed value against the KB field for that claim_type
    4. Classify: accurate | inaccurate | unverifiable | no_kb_match
"""
from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger(__name__)

# ── Severity classification ────────────────────────────────────────────────────
_CRITICAL_CLAIM_TYPES = {"price", "size", "delivery"}
_MINOR_CLAIM_TYPES    = {"down_payment", "installment_years", "feature", "location"}


# ── FAISS-based property identification ───────────────────────────────────────

def _find_property_id_by_search(raw_text: str, property_hint: str) -> Optional[str]:
    """
    Use FAISS semantic search to find which property is being discussed.
    Prefers hits that have 'property_id' in their metadata.
    Returns property_id string or None.
    """
    try:
        from rag.vector_store import faiss_search
        query = f"{raw_text} {property_hint}".strip()
        hits  = faiss_search(query, top_k=5)

        # First pass: look for a structured chunk hit with property_id
        for hit in hits:
            if hit["score"] < 0.35:
                continue
            md = hit.get("metadata", {})
            pid = md.get("property_id")
            if pid:
                return pid

        # Second pass: look inside the content text for "property_id: <id>" pattern
        # (This handles the case where generic loader chunks have the id in content but not metadata)
        for hit in hits:
            if hit["score"] < 0.40:
                continue
            content = hit.get("content", "")
            for line in content.split("\n"):
                if line.strip().startswith("id:"):
                    candidate = line.split(":", 1)[1].strip()
                    if candidate:
                        return candidate

    except Exception as exc:
        _log.warning("[FactChecker] FAISS search error: %s", exc)

    return None


# ── Per-claim comparison logic ─────────────────────────────────────────────────

def _check_claim(claim: dict, prop: dict) -> dict:
    """
    Compare one claim against the structured property record.

    Returns:
        {
            "result":       "accurate" | "inaccurate" | "unverifiable",
            "correct_value": str,
            "explanation_ar": str,
        }
    """
    ctype  = claim["claim_type"]
    cvalue = claim["claimed_value"]

    def _unverifiable(reason_ar: str = "لا يمكن التحقق") -> dict:
        return {"result": "unverifiable", "correct_value": "", "explanation_ar": reason_ar}

    # ── price ────────────────────────────────────────────────────────────────
    if ctype == "price":
        try:
            cv   = float(cvalue)
            pmin = float(prop.get("price_min") or 0)
            pmax = float(prop.get("price_max") or 0)
            if pmax == 0:
                return _unverifiable("نطاق السعر غير متوفر")
            # 10% tolerance on both ends
            if pmin * 0.9 <= cv <= pmax * 1.1:
                return {
                    "result": "accurate",
                    "correct_value": f"{pmin:,.0f} - {pmax:,.0f} EGP",
                    "explanation_ar": "",
                }
            return {
                "result": "inaccurate",
                "correct_value": f"{pmin:,.0f} - {pmax:,.0f} EGP",
                "explanation_ar": (
                    f"السعر المذكور {cv:,.0f} ج.م. غير صحيح. "
                    f"نطاق السعر الحقيقي: {pmin:,.0f} - {pmax:,.0f} ج.م."
                ),
            }
        except (TypeError, ValueError):
            return _unverifiable("قيمة السعر غير صالحة")

    # ── size ─────────────────────────────────────────────────────────────────
    if ctype == "size":
        try:
            cv   = float(cvalue)
            smin = float(prop.get("size_min_sqm") or 0)
            smax = float(prop.get("size_max_sqm") or 0)
            if smax == 0:
                return _unverifiable("نطاق المساحة غير متوفر")
            if smin * 0.9 <= cv <= smax * 1.1:
                return {
                    "result": "accurate",
                    "correct_value": f"{smin:.0f} - {smax:.0f} متر",
                    "explanation_ar": "",
                }
            return {
                "result": "inaccurate",
                "correct_value": f"{smin:.0f} - {smax:.0f} متر",
                "explanation_ar": (
                    f"المساحة المذكورة {cv:.0f} متر غير صحيحة. "
                    f"المساحات المتاحة: {smin:.0f} - {smax:.0f} متر"
                ),
            }
        except (TypeError, ValueError):
            return _unverifiable("قيمة المساحة غير صالحة")

    # ── down_payment ─────────────────────────────────────────────────────────
    if ctype == "down_payment":
        try:
            cv = float(cvalue)
            dp = float(prop.get("down_payment_percent") or 0)
            if dp == 0:
                return _unverifiable("نسبة المقدم غير متوفرة")
            if abs(cv - dp) <= 2:  # 2% tolerance
                return {
                    "result": "accurate",
                    "correct_value": f"{dp:.0f}%",
                    "explanation_ar": "",
                }
            return {
                "result": "inaccurate",
                "correct_value": f"{dp:.0f}%",
                "explanation_ar": (
                    f"المقدم المذكور {cv:.0f}% غير صحيح. "
                    f"المقدم الأساسي: {dp:.0f}%"
                ),
            }
        except (TypeError, ValueError):
            return _unverifiable("قيمة المقدم غير صالحة")

    # ── installment_years ────────────────────────────────────────────────────
    if ctype == "installment_years":
        try:
            cv = int(float(cvalue))
            iy = int(prop.get("installment_years") or 0)
            if iy == 0:
                return _unverifiable("فترة التقسيط غير متوفرة")
            if cv == iy:
                return {
                    "result": "accurate",
                    "correct_value": f"{iy} سنوات",
                    "explanation_ar": "",
                }
            return {
                "result": "inaccurate",
                "correct_value": f"{iy} سنوات",
                "explanation_ar": (
                    f"فترة التقسيط المذكورة {cv} سنوات غير صحيحة. "
                    f"الفترة الصحيحة: {iy} سنوات"
                ),
            }
        except (TypeError, ValueError):
            return _unverifiable("قيمة فترة التقسيط غير صالحة")

    # ── delivery ─────────────────────────────────────────────────────────────
    if ctype == "delivery":
        try:
            cv = int(cvalue)
            dy = int(prop.get("delivery_year") or 0)
            if dy == 0:
                return _unverifiable("سنة التسليم غير متوفرة")
            if cv == dy:
                return {
                    "result": "accurate",
                    "correct_value": str(dy),
                    "explanation_ar": "",
                }
            return {
                "result": "inaccurate",
                "correct_value": str(dy),
                "explanation_ar": (
                    f"سنة التسليم المذكورة {cv} غير صحيحة. "
                    f"سنة التسليم الصحيحة: {dy}"
                ),
            }
        except (TypeError, ValueError):
            return _unverifiable("قيمة سنة التسليم غير صالحة")

    # ── feature ──────────────────────────────────────────────────────────────
    if ctype == "feature":
        features_text = " ".join(f.lower() for f in prop.get("features", []))
        # Map canonical feature ID to search terms
        _terms: dict[str, list[str]] = {
            "swimming_pool":  ["pool", "مسبح", "سباح"],
            "gym":            ["gym", "جيم", "رياضي", "صحي"],
            "security":       ["security", "أمن", "حراس", "guard"],
            "parking":        ["parking", "جراج", "garage"],
            "elevator":       ["elevator", "lift", "مصعد"],
            "garden":         ["garden", "حديقة"],
            "beach":          ["beach", "شاطئ"],
            "private_beach":  ["private beach", "شاطئ خاص"],
        }
        search_terms = _terms.get(cvalue, [cvalue])
        if any(t.lower() in features_text for t in search_terms):
            return {
                "result": "accurate",
                "correct_value": cvalue,
                "explanation_ar": "",
            }
        # Feature not found — unverifiable rather than inaccurate
        return _unverifiable(f"لا يمكن التحقق من وجود '{cvalue}' في قاعدة البيانات")

    return _unverifiable("نوع المطالبة غير معروف")


# ── Main entry point ───────────────────────────────────────────────────────────

def check_facts(
    claims:     list[dict],
    properties: list[dict],
    policies:   list[dict],
) -> dict:
    """
    Check each claim against the structured knowledge base.

    Returns:
    {
        "claims_checked":      int,
        "accurate_count":      int,
        "inaccurate_count":    int,
        "accuracy_rate":       float,   # accurate / (accurate + inaccurate), 1.0 if none
        "errors": [
            {
                "turn_number":    int,
                "claim_type":     str,
                "severity":       "critical" | "minor",
                "claimed_value":  str,
                "correct_value":  str,
                "property_name":  str,
                "raw_text":       str,
                "explanation_ar": str,
            }
        ],
        "property_mentions": [{"property_id", "property_name", "mention_count"}],
        "unverifiable_claims": list[dict],
    }
    """
    if not claims:
        return {
            "claims_checked":   0,
            "accurate_count":   0,
            "inaccurate_count": 0,
            "accuracy_rate":    1.0,
            "errors":           [],
            "property_mentions": [],
            "unverifiable_claims": [],
        }

    # Build fast lookup
    props_by_id: dict[str, dict] = {p["id"]: p for p in properties}

    errors:       list[dict] = []
    unverifiable: list[dict] = []
    accurate_count   = 0
    inaccurate_count = 0
    property_mentions: dict[str, dict] = {}   # property_id → mention record

    for claim in claims:
        raw_text  = claim.get("raw_text", "")
        hint      = claim.get("property_hint", "")

        # ── Step 1: identify the property being discussed ────────────────────
        property_id = _find_property_id_by_search(raw_text, hint)

        # Fall back to property_hint if FAISS returned nothing with an ID
        if not property_id and hint and hint in props_by_id:
            property_id = hint

        if not property_id or property_id not in props_by_id:
            unverifiable.append({**claim, "reason": "no_kb_match"})
            continue

        prop      = props_by_id[property_id]
        prop_name = prop.get("name_ar") or prop.get("name_en") or property_id

        # Track property mentions
        if property_id not in property_mentions:
            property_mentions[property_id] = {
                "property_id":   property_id,
                "property_name": prop_name,
                "mention_count": 0,
            }
        property_mentions[property_id]["mention_count"] += 1

        # ── Step 2: compare the claim against the KB ─────────────────────────
        check = _check_claim(claim, prop)
        result = check["result"]

        if result == "accurate":
            accurate_count += 1

        elif result == "inaccurate":
            inaccurate_count += 1
            severity = (
                "critical"
                if claim["claim_type"] in _CRITICAL_CLAIM_TYPES
                else "minor"
            )
            errors.append({
                "turn_number":    claim["turn_number"],
                "claim_type":     claim["claim_type"],
                "severity":       severity,
                "claimed_value":  claim["claimed_value"],
                "correct_value":  check["correct_value"],
                "property_name":  prop_name,
                "raw_text":       raw_text,
                "explanation_ar": check["explanation_ar"],
            })

        else:   # unverifiable
            unverifiable.append({**claim, "reason": result})

    verifiable   = accurate_count + inaccurate_count
    accuracy_rate = (accurate_count / verifiable) if verifiable > 0 else 1.0

    return {
        "claims_checked":      len(claims),
        "accurate_count":      accurate_count,
        "inaccurate_count":    inaccurate_count,
        "accuracy_rate":       round(accuracy_rate, 3),
        "errors":              errors,
        "property_mentions":   list(property_mentions.values()),
        "unverifiable_claims": unverifiable,
    }


# ── Manual test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    # Allow running as: python rag/fact_checker.py from project root
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from rag.claim_extractor import extract_salesperson_claims
    from rag.structured_store import load_properties, load_policies

    print("=" * 60)
    print("Fact-Checker Test: wrong price for Madinaty")
    print("=" * 60)

    # Salesperson claims 1.5M EGP for Madinaty — actual min is 2M EGP
    mock_transcript = [
        {
            "speaker":     "salesperson",
            "text":        "شقة في مدينتي بسعر مليون ونص",
            "turn_number": 1,
        }
    ]

    # Step 1: extract claims
    claims = extract_salesperson_claims(mock_transcript)
    print(f"\n[1] Extracted {len(claims)} claim(s):")
    for c in claims:
        print(f"    type={c['claim_type']}  value={c['claimed_value']}  "
              f"unit={c['claimed_unit']}  hint={c['property_hint']}")

    # Step 2: fact-check
    result = check_facts(claims, load_properties(), load_policies())
    print(f"\n[2] Fact-check result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Step 3: assert inaccuracy was detected
    print("\n[3] Assertions:")
    assert result["inaccurate_count"] >= 1, "FAIL: expected at least 1 inaccurate claim"
    assert len(result["errors"]) >= 1,      "FAIL: expected at least 1 error"
    price_errors = [e for e in result["errors"] if e["claim_type"] == "price"]
    assert price_errors,                    "FAIL: expected a price error"
    assert price_errors[0]["severity"] == "critical", "FAIL: price error should be critical"
    print("    PASS: inaccurate price detected")
    print("    PASS: error severity = critical")
    print("    PASS: property_name =", price_errors[0]["property_name"])
    print("    PASS: explanation_ar =", price_errors[0]["explanation_ar"])
    print("\nAll assertions passed.")
