#!/usr/bin/env python
"""
Phase 3 manual test: signup -> invite -> accept -> hit seat limit -> deactivate
-> blocked downgrade. Runs against the live app via FastAPI TestClient (so it
exercises the real DB configured in .env). No server needs to be running.

Run from the VCAI repo root with the project venv:
    <venv-python> scripts/phase3_manual_test.py

The curl equivalents of each step are printed in comments below.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
import backend.main as m  # noqa: E402


def main() -> int:
    c = TestClient(m.app)
    s = uuid.uuid4().hex[:8]
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok = ok and cond
        print(f"[{'PASS' if cond else 'FAIL'}] {label} {extra}")

    # 1. Signup on a paid plan (mocked checkout). starter -> seat_limit 5.
    #    curl -X POST localhost:8000/api/onboarding/signup \
    #      -H 'Content-Type: application/json' \
    #      -d '{"company_name":"Acme","plan_name":"starter","billing_cycle":"monthly",
    #           "manager_name":"Jane","manager_email":"jane@acme.com","password":"secret123"}'
    r = c.post("/api/onboarding/signup", json={
        "company_name": f"Acme {s}", "plan_name": "starter", "billing_cycle": "monthly",
        "manager_name": "Jane Manager", "manager_email": f"mgr_{s}@example.com",
        "password": "secret123"})
    check("signup -> 201 + trial", r.status_code == 201 and r.json()["billing_status"] == "trial",
          f"({r.status_code})")
    H = {"Authorization": f"Bearer {r.json()['access_token']}"}

    # 2. Manager sees subscription + limits.
    #    curl localhost:8000/api/subscriptions/me -H "Authorization: Bearer $TOKEN"
    r = c.get("/api/subscriptions/me", headers=H)
    check("subscription/me -> seat_limit 5", r.status_code == 200 and r.json()["seat_limit"] == 5)

    # 3. Invite an agent (email stubbed -> link returned).
    #    curl -X POST localhost:8000/api/seats/invite -H "Authorization: Bearer $TOKEN" \
    #      -H 'Content-Type: application/json' -d '{"email":"agent@acme.com"}'
    r = c.post("/api/seats/invite", headers=H, json={"email": f"a1_{s}@x.com"})
    check("invite -> 201 + link", r.status_code == 201 and bool(r.json().get("invite_link")))
    token = r.json()["token"]

    # 4. Agent accepts -> salesperson in same company.
    #    curl -X POST localhost:8000/api/onboarding/accept -H 'Content-Type: application/json' \
    #      -d '{"token":"<token>","full_name":"Sam","password":"secret123"}'
    r = c.post("/api/onboarding/accept",
               json={"token": token, "full_name": "Sam Agent", "password": "secret123"})
    check("accept -> salesperson + company_id",
          r.status_code == 200 and r.json()["user"]["role"] == "salesperson"
          and bool(r.json()["user"]["company_id"]))
    sp_id = r.json()["user"]["id"]

    # 5. Fill the remaining seats, then the (limit+1)th invite is blocked with 409.
    for i in range(2, 6):
        c.post("/api/seats/invite", headers=H, json={"email": f"a{i}_{s}@x.com"})
    r = c.post("/api/seats/invite", headers=H, json={"email": f"over_{s}@x.com"})
    check("invite over limit -> 409", r.status_code == 409, f"({r.json().get('detail')})")

    # 6. Deactivate frees a seat.
    #    curl -X POST localhost:8000/api/seats/<user_id>/deactivate -H "Authorization: Bearer $TOKEN"
    r = c.post(f"/api/seats/{sp_id}/deactivate", headers=H)
    check("deactivate -> is_active False", r.status_code == 200 and r.json()["is_active"] is False)

    # 7. Downgrade below active seats is blocked. Use a fresh company with 2
    #    accepted agents on starter, then attempt to move to free (1 seat) -> 409.
    s2 = uuid.uuid4().hex[:8]
    r = c.post("/api/onboarding/signup", json={
        "company_name": f"Beta {s2}", "plan_name": "starter", "billing_cycle": "monthly",
        "manager_name": "Bob Manager", "manager_email": f"mgr_{s2}@example.com",
        "password": "secret123"})
    H2 = {"Authorization": f"Bearer {r.json()['access_token']}"}
    for i in range(2):
        t = c.post("/api/seats/invite", headers=H2, json={"email": f"b{i}_{s2}@x.com"}).json()["token"]
        c.post("/api/onboarding/accept", json={"token": t, "full_name": f"B {i}", "password": "secret123"})
    r = c.post("/api/subscriptions/change-plan", headers=H2,
               json={"plan_name": "free", "billing_cycle": "monthly"})
    check("downgrade below seats -> 409", r.status_code == 409, f"({r.json().get('detail')})")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
