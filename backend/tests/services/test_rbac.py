# backend/tests/services/test_rbac.py
"""
Unit tests for the Phase 1 RBAC + tenant-scoping helpers.

These do not touch the database — they exercise the pure permission logic in
auth_service (require_role, assert_same_company) using lightweight fake users.
"""
import asyncio
import uuid

import pytest
from fastapi import HTTPException

from backend.services.auth_service import require_role, assert_same_company


class FakeUser:
    def __init__(self, role, company_id=None):
        self.role = role
        self.company_id = company_id


def _run_dep(dep, user):
    """Invoke a require_role dependency directly with an injected user."""
    return asyncio.run(dep(current_user=user))


def test_require_role_allows_listed_role():
    dep = require_role("manager", "superadmin")
    user = FakeUser("manager")
    assert _run_dep(dep, user) is user


def test_require_role_blocks_other_role():
    dep = require_role("manager", "superadmin")
    user = FakeUser("salesperson")
    with pytest.raises(HTTPException) as exc:
        _run_dep(dep, user)
    assert exc.value.status_code == 403


def test_assert_same_company_superadmin_bypasses():
    user = FakeUser("superadmin", company_id=None)
    # Should not raise even for a foreign company.
    assert_same_company(user, uuid.uuid4()) is None


def test_assert_same_company_matching_company_ok():
    cid = uuid.uuid4()
    user = FakeUser("manager", company_id=cid)
    assert assert_same_company(user, cid) is None


def test_assert_same_company_cross_tenant_denied():
    user = FakeUser("manager", company_id=uuid.uuid4())
    with pytest.raises(HTTPException) as exc:
        assert_same_company(user, uuid.uuid4())
    assert exc.value.status_code == 403
