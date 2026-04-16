# rag/structured_store.py
"""
Structured JSON store for properties and policies.
Loads data/documents/properties.json and data/documents/policies.json once per process.

Usage:
    from rag.structured_store import load_properties, load_policies
    from rag.structured_store import get_property_by_id, get_policy_by_id
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)

_PROPERTIES_PATH = Path("data/documents/properties.json")
_POLICIES_PATH   = Path("data/documents/policies.json")

# Module-level singletons — populated on first access
_properties: list[dict] | None = None
_policies:   list[dict] | None = None


def load_properties() -> list[dict]:
    """Load and cache all property records. Returns list of property dicts."""
    global _properties
    if _properties is None:
        with open(_PROPERTIES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        _properties = data.get("properties", [])
        _maybe_log_combined()
    return _properties


def load_policies() -> list[dict]:
    """Load and cache all policy records. Returns list of policy dicts."""
    global _policies
    if _policies is None:
        with open(_POLICIES_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        _policies = data.get("policies", [])
        _maybe_log_combined()
    return _policies


def _maybe_log_combined() -> None:
    """Print combined log line once BOTH datasets have been loaded."""
    if _properties is not None and _policies is not None:
        msg = (
            f"[StructuredStore] Loaded {len(_properties)} properties, "
            f"{len(_policies)} policies"
        )
        _log.info(msg)
        print(msg)


def get_property_by_id(property_id: str) -> dict | None:
    """Return the full property dict for the given ID, or None if not found."""
    for prop in load_properties():
        if prop.get("id") == property_id:
            return prop
    return None


def get_policy_by_id(policy_id: str) -> dict | None:
    """Return the full policy dict for the given ID, or None if not found."""
    for pol in load_policies():
        if pol.get("id") == policy_id:
            return pol
    return None
