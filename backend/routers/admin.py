# backend/routers/admin.py
"""
Super-admin (platform owner) API. Cross-tenant visibility + manual levers.
Every endpoint requires superadmin (get_current_superadmin); managers and
salespeople get 403. See 00_ARCHITECTURE.md sections 3, 6, 10.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.health import get_health_status
from backend.models import User
from backend.schemas import (
    TenantListResponse,
    TenantSummary,
    PlatformUsage,
    TenantStatusChange,
    AbuseFlagOut,
    AuditLogOut,
)
from backend.services import (
    get_current_superadmin,
    platform_usage,
    list_tenants,
    tenant_detail,
    set_tenant_status,
    global_abuse,
    global_audit,
)

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/tenants", response_model=TenantListResponse)
def tenants(
    search: Optional[str] = Query(None),
    plan: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_superadmin),
):
    """All companies with plan/billing/usage summary. Search + plan filter + pagination."""
    data = list_tenants(db, search=search, plan=plan, limit=limit, offset=offset)
    return TenantListResponse(
        tenants=[TenantSummary(**t) for t in data["tenants"]],
        total=data["total"],
    )


@router.get("/tenants/{company_id}")
def tenant_detail_endpoint(
    company_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_superadmin),
):
    """One company's full detail: subscription, agents, usage history, flags, audit."""
    return tenant_detail(db, company_id)


@router.post("/tenants/{company_id}/suspend", response_model=TenantStatusChange)
def suspend_tenant(
    company_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_superadmin),
):
    """Suspend a tenant: blocks new sessions (Phase 4 billing gate) + marks inactive."""
    company = set_tenant_status(db, company_id=company_id, actor=current_user, suspend=True)
    return TenantStatusChange(company_id=company.id, name=company.name, is_active=company.is_active)


@router.post("/tenants/{company_id}/reactivate", response_model=TenantStatusChange)
def reactivate_tenant(
    company_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_superadmin),
):
    """Reactivate a previously suspended tenant."""
    company = set_tenant_status(db, company_id=company_id, actor=current_user, suspend=False)
    return TenantStatusChange(company_id=company.id, name=company.name, is_active=company.is_active)


@router.get("/usage", response_model=PlatformUsage)
def usage(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_superadmin),
):
    """Cross-tenant rollups: companies, subs by plan, platform sessions, top tenants."""
    return platform_usage(db)


@router.get("/abuse", response_model=list[AbuseFlagOut])
def abuse(
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_superadmin),
):
    """Global abuse queue across all tenants, newest first."""
    flags = global_abuse(db, severity=severity, status_filter=status)
    return [AbuseFlagOut.model_validate(f) for f in flags]


@router.get("/audit", response_model=list[AuditLogOut])
def audit(
    company_id: Optional[UUID] = Query(None),
    action: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_superadmin),
):
    """Global audit log, filterable by company/action, newest first."""
    rows = global_audit(db, company_id=company_id, action=action, limit=limit)
    return [AuditLogOut.model_validate(r) for r in rows]


@router.get("/health")
def health(
    _: User = Depends(get_current_superadmin),
):
    """Proxy the existing 6-module health check (reused, not reimplemented)."""
    return get_health_status()
