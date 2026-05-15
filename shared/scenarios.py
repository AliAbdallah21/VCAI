"""
Buyer scenarios for training sessions.

A scenario is the *situation* the virtual customer is in — independent of the
persona (personality) and difficulty. It carries:

    buyer_context   first_time | investor | family_upgrade | downsizer
    budget_min/max  the customer's price range (EGP)
    timeline        urgent | flexible | exploring
    must_haves      list of Arabic requirement phrases
    deal_breakers   list of Arabic phrases that make the customer walk away
    label / label_en  human-readable summary

One generator powers all three setup modes:
    - preset  : every field pinned (from SCENARIO_PRESETS)
    - custom  : some fields pinned, the rest drawn coherently
    - random  : nothing pinned

Coherence: each buyer_context owns its own budget bands, requirement pool, and
typical timelines, so a generated scenario never produces an absurd combo
(e.g. a first-time buyer with a 6M budget).
"""
from __future__ import annotations

import random
from typing import Any, Optional


# ── Timelines ─────────────────────────────────────────────────────────────────

TIMELINES: dict[str, dict[str, str]] = {
    "urgent": {
        "label_ar": "مستعجل — محتاج يستلم خلال شهر",
        "deal_breaker_ar": "التسليم بعد أكتر من شهرين",
    },
    "flexible": {
        "label_ar": "مرن — خلال ٣ لـ ٦ شهور",
        "deal_breaker_ar": "التسليم بعد أكتر من سنة",
    },
    "exploring": {
        "label_ar": "لسه بيستكشف — مش مستعجل",
        "deal_breaker_ar": "ضغط عليه يقرّر بسرعة",
    },
}


# ── Buyer contexts ────────────────────────────────────────────────────────────
# Each context owns coherent pools so generated scenarios always make sense.

BUYER_CONTEXTS: dict[str, dict[str, Any]] = {
    "first_time": {
        "name_ar": "مشتري لأول مرة",
        "name_en": "First-time buyer",
        "budget_bands": [(900_000, 1_400_000), (1_200_000, 1_800_000)],
        "must_have_pool": [
            "شقة جاهزة للسكن من غير تشطيب إضافي",
            "تقسيط مريح على سنين",
            "قريبة من مواصلات",
            "غرفتين على الأقل",
            "مقدم معقول مش كبير",
        ],
        "timelines": ["flexible", "exploring"],
    },
    "investor": {
        "name_ar": "مستثمر عقاري",
        "name_en": "Property investor",
        "budget_bands": [(2_000_000, 3_500_000), (3_000_000, 5_000_000)],
        "must_have_pool": [
            "موقع بيرتفع سعره مع الوقت",
            "إمكانية تأجير بعائد كويس",
            "كمبوند له سمعة",
            "تسليم قريب عشان يبدأ يأجّر",
            "سعر المتر تنافسي",
        ],
        "timelines": ["exploring", "flexible"],
    },
    "family_upgrade": {
        "name_ar": "عيلة بتكبر ومحتاجة شقة أوسع",
        "name_en": "Growing family upgrading",
        "budget_bands": [(2_500_000, 3_800_000), (3_200_000, 4_500_000)],
        "must_have_pool": [
            "٣ غرف نوم على الأقل",
            "قريبة من مدارس كويسة",
            "كمبوند فيه أمان ومساحات خضرا",
            "قريبة من خدمات وسوبر ماركت",
            "جراج للعربية",
        ],
        "timelines": ["urgent", "flexible"],
    },
    "downsizer": {
        "name_ar": "بيدوّر على شقة أصغر وأبسط",
        "name_en": "Downsizer",
        "budget_bands": [(1_500_000, 2_200_000), (1_900_000, 2_800_000)],
        "must_have_pool": [
            "دور أرضي أو واطي مع أسانسير",
            "مساحة أصغر سهلة الصيانة",
            "منطقة هادية",
            "مصاريف صيانة قليلة",
            "تشطيب جاهز",
        ],
        "timelines": ["flexible", "exploring"],
    },
}


# ── Curated presets ───────────────────────────────────────────────────────────
# Eight hand-written, fully-specified scenarios for the "predictable" mode.

SCENARIO_PRESETS: dict[str, dict[str, Any]] = {
    "young_couple_starter": {
        "buyer_context": "first_time",
        "budget_min": 1_000_000, "budget_max": 1_500_000,
        "timeline": "flexible",
        "must_haves": ["غرفتين على الأقل", "تقسيط مريح على سنين", "قريبة من مواصلات"],
    },
    "first_apartment_tight": {
        "buyer_context": "first_time",
        "budget_min": 900_000, "budget_max": 1_200_000,
        "timeline": "exploring",
        "must_haves": ["شقة جاهزة للسكن من غير تشطيب إضافي", "مقدم معقول مش كبير"],
    },
    "investor_rental_yield": {
        "buyer_context": "investor",
        "budget_min": 2_000_000, "budget_max": 3_000_000,
        "timeline": "exploring",
        "must_haves": ["إمكانية تأجير بعائد كويس", "موقع بيرتفع سعره مع الوقت"],
    },
    "investor_premium": {
        "buyer_context": "investor",
        "budget_min": 3_500_000, "budget_max": 5_000_000,
        "timeline": "flexible",
        "must_haves": ["كمبوند له سمعة", "سعر المتر تنافسي", "تسليم قريب عشان يبدأ يأجّر"],
    },
    "family_needs_schools": {
        "buyer_context": "family_upgrade",
        "budget_min": 2_800_000, "budget_max": 3_800_000,
        "timeline": "urgent",
        "must_haves": ["٣ غرف نوم على الأقل", "قريبة من مدارس كويسة", "كمبوند فيه أمان ومساحات خضرا"],
    },
    "family_spacious": {
        "buyer_context": "family_upgrade",
        "budget_min": 3_200_000, "budget_max": 4_500_000,
        "timeline": "flexible",
        "must_haves": ["٣ غرف نوم على الأقل", "جراج للعربية", "قريبة من خدمات وسوبر ماركت"],
    },
    "retiree_quiet": {
        "buyer_context": "downsizer",
        "budget_min": 1_500_000, "budget_max": 2_200_000,
        "timeline": "flexible",
        "must_haves": ["دور أرضي أو واطي مع أسانسير", "منطقة هادية", "مصاريف صيانة قليلة"],
    },
    "downsize_simple": {
        "buyer_context": "downsizer",
        "budget_min": 1_900_000, "budget_max": 2_600_000,
        "timeline": "exploring",
        "must_haves": ["مساحة أصغر سهلة الصيانة", "تشطيب جاهز"],
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_egp(amount: int) -> str:
    """Format an EGP amount as a short readable string (e.g. '1.5 مليون')."""
    millions = amount / 1_000_000
    if millions == int(millions):
        return f"{int(millions)} مليون"
    return f"{millions:.1f} مليون"


def _build_deal_breakers(budget_max: int, timeline: str) -> list[str]:
    """Derive the walk-away triggers from budget + timeline."""
    breakers = [f"السعر النهائي يعدّي {_fmt_egp(budget_max)}"]
    tl = TIMELINES.get(timeline)
    if tl:
        breakers.append(tl["deal_breaker_ar"])
    return breakers


def _build_label(scenario: dict[str, Any]) -> tuple[str, str]:
    """Human-readable Arabic + English one-liners for a scenario."""
    ctx = BUYER_CONTEXTS.get(scenario["buyer_context"], {})
    name_ar = ctx.get("name_ar", scenario["buyer_context"])
    name_en = ctx.get("name_en", scenario["buyer_context"])
    budget = f"{_fmt_egp(scenario['budget_min'])}–{_fmt_egp(scenario['budget_max'])}"
    tl_ar = TIMELINES.get(scenario["timeline"], {}).get("label_ar", scenario["timeline"])
    label_ar = f"{name_ar} · ميزانية {budget} · {tl_ar}"
    label_en = f"{name_en} · budget {scenario['budget_min']:,}-{scenario['budget_max']:,} EGP"
    return label_ar, label_en


def _finalize(scenario: dict[str, Any]) -> dict[str, Any]:
    """Fill derived fields (deal_breakers, labels) on a partly-built scenario."""
    scenario["deal_breakers"] = _build_deal_breakers(
        scenario["budget_max"], scenario["timeline"]
    )
    label_ar, label_en = _build_label(scenario)
    scenario["label"] = label_ar
    scenario["label_en"] = label_en
    return scenario


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_scenario(
    *,
    buyer_context: Optional[str] = None,
    budget_min: Optional[int] = None,
    budget_max: Optional[int] = None,
    timeline: Optional[str] = None,
    must_haves: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
) -> dict[str, Any]:
    """
    Build a coherent scenario, filling any unpinned field by drawing from the
    buyer context's pools.

    - Pin nothing  -> fully random scenario.
    - Pin some     -> custom scenario (rest drawn coherently).
    - Pin all      -> deterministic scenario.

    Unknown / invalid pins are ignored (fall back to a random draw) so callers
    can pass user input loosely.
    """
    r = rng or random

    # 1. Buyer context — the anchor everything else is coherent with.
    if buyer_context not in BUYER_CONTEXTS:
        buyer_context = r.choice(list(BUYER_CONTEXTS.keys()))
    ctx = BUYER_CONTEXTS[buyer_context]

    # 2. Budget — keep a pinned pair if both given, else draw a band.
    if not (isinstance(budget_min, int) and isinstance(budget_max, int)
            and 0 < budget_min < budget_max):
        budget_min, budget_max = r.choice(ctx["budget_bands"])

    # 3. Timeline — must be one this context plausibly has.
    if timeline not in TIMELINES:
        timeline = r.choice(ctx["timelines"])

    # 4. Must-haves — pinned list, else 2-3 drawn from the context pool.
    if not must_haves:
        pool = ctx["must_have_pool"]
        k = min(len(pool), r.randint(2, 3))
        must_haves = r.sample(pool, k)

    scenario = {
        "buyer_context": buyer_context,
        "budget_min": int(budget_min),
        "budget_max": int(budget_max),
        "timeline": timeline,
        "must_haves": list(must_haves),
    }
    return _finalize(scenario)


def scenario_from_preset(preset_id: str) -> dict[str, Any]:
    """Build a full scenario from a curated preset id. Raises KeyError if unknown."""
    preset = SCENARIO_PRESETS[preset_id]
    return generate_scenario(
        buyer_context=preset["buyer_context"],
        budget_min=preset["budget_min"],
        budget_max=preset["budget_max"],
        timeline=preset["timeline"],
        must_haves=preset["must_haves"],
    )


def resolve_scenario(spec: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Single entry point used by session creation.

    `spec` shapes accepted:
      None / {}                    -> fully random
      {"mode": "random"}           -> fully random
      {"mode": "preset", "preset_id": "..."}  -> a curated preset
      {"mode": "custom", "pins": {...}}       -> custom (pins kept, rest drawn)

    Always returns a complete, coherent scenario dict — never raises on bad
    input (falls back to random).
    """
    if not spec:
        return generate_scenario()

    mode = spec.get("mode", "random")

    if mode == "preset":
        preset_id = spec.get("preset_id")
        if preset_id in SCENARIO_PRESETS:
            return scenario_from_preset(preset_id)
        return generate_scenario()  # unknown preset -> random fallback

    if mode == "custom":
        pins = spec.get("pins") or {}
        return generate_scenario(
            buyer_context=pins.get("buyer_context"),
            budget_min=pins.get("budget_min"),
            budget_max=pins.get("budget_max"),
            timeline=pins.get("timeline"),
            must_haves=pins.get("must_haves"),
        )

    # mode == "random" or anything unrecognised
    return generate_scenario()


def list_presets() -> list[dict[str, Any]]:
    """Return all presets as resolved scenarios with their ids — for the UI picker."""
    out = []
    for pid in SCENARIO_PRESETS:
        sc = scenario_from_preset(pid)
        sc["preset_id"] = pid
        out.append(sc)
    return out
