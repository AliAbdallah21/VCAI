# VCAI Bug-Fix & Polish Pass — Handoff / Resume

This file hands off an in-progress "pre-demo hardening" task on the **VCAI** codebase
(React + Vite frontend, FastAPI + PostgreSQL backend) to another AI agent.

Repo root: `d:\miu\Freshman Year (4)\Grad Proj\Code\VCAI`
Frontend: `frontend/src` · Backend: `backend/` · Migrations: `migrations/versions/`

---

## How to run things on this machine (verified)

- **Python interpreter** (the ONLY one with the full dep set — alembic, sqlalchemy,
  psycopg2, pydantic_settings): `d:\miu\Freshman Year (4)\Grad Proj\Code\venv\Scripts\python.exe`
  (conda `Ali` lacks alembic; global Py3.13 lacks pydantic_settings.)
- **Alembic** (run from VCAI repo root; migrations dir is `migrations/`, not `alembic/`):
  - `<venv-python> -c "from alembic.config import main; main(argv=['upgrade','head'])"`
  - `<venv-python> -c "from alembic.config import main; main(argv=['downgrade','-1'])"`
  - Do NOT use `alembic.exe` on PATH — wrong interpreter.
- **Backend dev server:** `python -m backend.main` or `uvicorn backend.main:app --reload --port 8000`
- **Frontend dev:** `cd frontend && npm run dev` (http://localhost:5173)
- **DB:** Postgres `vcai` on localhost:5432 (postgres / pwd `Ali24680#`, URL-encode `#` as `%23`).
- **bcrypt must stay pinned at 4.0.1** in the venv or password hashing breaks.
- Tests use SQLite in-memory (`Base.metadata.create_all`), so they match the models even
  when the live Postgres has column drift.

---

## CRITICAL CONTEXT: the QA brief is stale

The original task was a QA "Bug Fix & Polish" brief with 10 numbered issues. **The brief
predates the June-2026 UI redesign, so several items are ALREADY fixed in current code.**
Verify, do not re-implement these:

| # | Brief claim | Reality in current code |
|---|---|---|
| 1 | "Start free" CTAs do nothing | Already wired: hero + navbar `<Link to="/onboarding?plan=free">`; "See pricing" → `#pricing`. `frontend/src/pages/Landing.jsx` |
| 6 | Paid tiers have no feature lists | Already present via `planFeatures()` in `frontend/src/components/PricingCards.jsx` |
| 7 | "Built for realistic practice" empty | Already has 1-sentence body per feature. `Landing.jsx` FEATURES array |
| 10 | Empty Compare/Progress no guidance | Already have empty states + `/setup` CTAs (`ProgressPage` `EmptyState`; `ComparePage` help footer) |

Also note: brief #8 mentions a blank page at GET `/api/auth/signout` — **there is no signout
route at all** (no GET/POST). Logout is purely client-side in `AuthContext`. So that part is moot.

---

## The API base-URL mechanism (was required before changes)

- **Frontend** (`frontend/src/services/api.js`): same-origin. `API_URL = VITE_API_URL ||
  \`${window.location.origin}/api\``; WS swaps `http→ws`. The frontend already adapts to any
  serving origin — it is NOT the source of the hardcoded port 5173.
- **Backend**: `backend/config.py` → `frontend_base_url` defaults to `"http://localhost:5173"`
  and `.env` does NOT override it. Invite links are built backend-side in
  `backend/services/seat_service.py::_invite_link` from that setting. **This is the real
  source of the hardcoded-5173 bug (#3).**

---

## Decisions already made with the user (do not re-litigate)

Already-fixed items (#1,#6,#7,#10): **skip, verify only.**

**#2 was reframed by the user into an invite-code feature** (this is the biggest piece of work):
- A user always registers as a **solo salesperson**. There is **no "join a company by typing
  its name."** Joining a company happens ONLY via an invite.
- Three ways to join: (a) the existing invite **link** `/invite/:token`; (b) pasting a
  **6-char alphanumeric invite code** at **registration**; (c) pasting the code later from a
  new **/settings** page (attaches the *existing* logged-in account).
- The 6-char code lives **on the existing email-bound `SeatInvite`** (new `invite_code`
  column) — NOT a company-wide reusable code. Single-use + seat-limit accounting unchanged.
- When a **logged-in** user pastes a code: set their existing account's `company_id` + `role`
  (no new user created; their email need NOT match the invite's target email).
- **Invalid code at registration → reject the whole registration** ("Invalid or expired
  invite code"), do not silently create a solo account.
- **/settings**: build a new minimal page with a "Join a company" card (code input), shown
  to users without a `company_id`; link it from the dashboard menu.

**#3**: derive the invite URL origin from the **incoming request host** (scheme+host) in the
seats router, falling back to `frontend_base_url`. (Chosen over setting an env var.)

---

## Work completed so far

- Full orientation of routes (`App.jsx`), auth (`AuthContext.jsx`), API client (`api.js`),
  seats/onboarding routers + services, and the relevant pages.
- Confirmed which issues are real vs already-fixed (table above).
- Pinned down all open design decisions with the user (above).
- Wrote project memory: `vcai-invite-code-flow.md` (+ MEMORY.md index entry).
- **No code changes committed yet** — implementation was just starting.

## Work still to do (priority order)

1. **#3 invite link origin** — in `backend/routers/seats.py`, derive base URL from the
   FastAPI `Request` (`request.base_url` / `X-Forwarded-*`), pass it down so
   `serialize_invite` / `_invite_link` use it; fall back to `settings.frontend_base_url`.
2. **#2 invite code — data layer** — add `invite_code` (String, 6 alnum, indexed) to
   `backend/models/seat_invite.py`; new migration `migrations/versions/0007_*.py`
   (ADD COLUMN IF NOT EXISTS pattern — see `0006_reconcile_model_columns.py`); generate the
   code in `seat_service.invite_seat`; include it in `serialize_invite` + `InviteResponse`
   schema (`backend/schemas/__init__.py`) + show it in `SeatManagement.jsx`.
3. **#2 invite code — join flows** — (a) `UserCreate` schema + `register_user` accept an
   optional `invite_code`; on register, if a code is present, validate + attach the new user
   to the company (reuse seat-limit checks) or reject if invalid. (b) New `POST /seats/join`
   ({code}) endpoint + `join_company_by_code` service that attaches the *current logged-in*
   user. Export new service fns in `backend/services/__init__.py`.
4. **#2 invite code — frontend** — add an optional "Invite code" field to
   `frontend/src/pages/Register.jsx`; new `frontend/src/pages/Settings.jsx` with a
   "Join a company" card calling the new endpoint; add `/settings` route in `App.jsx` and a
   menu link (see `Layout.jsx` / `components/ui/DashboardShell.jsx`); add `seatsAPI.join` to
   `api.js`.
5. **#4 stale-session invite fetch** — `onboardingAPI.getInvite` goes through the axios
   instance whose request interceptor attaches `localStorage.token` to EVERY request. Make
   the public invite-detail fetch token-free (e.g. a bare axios call, or strip Authorization
   for that path). Verify it renders for a logged-out / wrong-user visitor.
   File: `frontend/src/services/api.js` + `frontend/src/pages/AcceptInvite.jsx`.
6. **#5 company-name fallback** — in `AcceptInvite.jsx`, if `invite.company_name` is
   empty/whitespace/placeholder (e.g. "test"), render a neutral heading ("Join your team").
7. **#8 sign-out** — `AuthContext.logout()` only clears `token` + `user`. Also clear legacy
   keys `fitai_access_token`, `fitai_refresh_token` (and any other stale ones). Ensure all
   logout call sites navigate to `/login` (Layout.jsx, DashboardShell.jsx already do).
8. **#9 deactivate confirm** — in `frontend/src/pages/SeatManagement.jsx::handleDeactivate`,
   add a native `confirm()` that names the user before calling `seatsAPI.deactivate`.
9. Verify #1/#6/#7/#10 unchanged; produce a summary table (issue → files changed → how to verify).

## Constraints (from the brief — still apply)

- Keep the dark/purple visual language and existing component structure.
- Do NOT touch the call/WebSocket training-session logic (`orchestration/`, `ws_*`).
- Follow existing router/service patterns; flag any new dependency before adding it (none
  expected — code generation can use `secrets`/`random` from stdlib).
- After a logged-in user joins via code, their JWT/`user` in localStorage is now stale
  (role/company changed) — make sure the frontend refreshes `user` (call `authAPI.getMe()`
  or re-issue a token) so routing (e.g. `/seats`, manager dashboard) reflects the new role.

---

## Prompt for the next agent

> You are picking up an in-progress pre-demo hardening pass on the VCAI codebase
> (`d:\miu\Freshman Year (4)\Grad Proj\Code\VCAI`; React+Vite frontend, FastAPI+Postgres
> backend). Read this entire `resume.md` first — it captures the orientation, which QA items
> are already fixed (verify only: #1,#6,#7,#10), the run commands for THIS machine, and the
> design decisions already agreed with the user (do not re-ask them). Implement the
> "Work still to do" list in priority order, starting with #3 then the #2 invite-code
> feature, then the small frontend fixes (#4,#5,#8,#9). Match the existing dark/purple visual
> language and the existing router/service patterns; do not touch the WebSocket/training-session
> logic; do not add dependencies without flagging. After each fix, state which files you
> changed and why. When the live Postgres needs a schema change, add an `ADD COLUMN IF NOT
> EXISTS` migration (`0007_*`, modeled on `0006_reconcile_model_columns.py`) and run it with
> the venv python via the alembic-as-module invocation noted above. Finish with a summary
> table: issue # → files changed → how to verify. If a decision genuinely cannot be inferred
> from this file or the code, ask the user rather than guessing.
