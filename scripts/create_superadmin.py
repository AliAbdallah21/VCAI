# scripts/create_superadmin.py
"""
Bootstrap a platform super-admin (role="superadmin", company_id=NULL). This is
how the platform owners get accounts. Idempotent by email: re-running promotes
the existing user to superadmin and (optionally) resets the password.

Usage (interactive):
    <venv-python> scripts/create_superadmin.py

Usage (non-interactive):
    <venv-python> scripts/create_superadmin.py --email you@x.com --name "You" --password secret123

PREREQUISITE:
    <venv-python> -c "from alembic.config import main; main(argv=['upgrade','head'])"
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from backend.database import SessionLocal
from backend.models import User, UserStats
from backend.services.auth_service import get_password_hash


def _prompt_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create or promote a VCAI super-admin.")
    p.add_argument("--email")
    p.add_argument("--name")
    p.add_argument("--password")
    args = p.parse_args()

    if not args.email:
        args.email = input("Super-admin email: ").strip()
    if not args.name:
        args.name = input("Full name: ").strip()
    if not args.password:
        args.password = getpass.getpass("Password (min 6 chars): ").strip()
    return args


def create_superadmin(db, *, email: str, name: str, password: str) -> User:
    if not email or not name or not password:
        raise SystemExit("email, name, and password are all required.")
    if len(password) < 6:
        raise SystemExit("Password must be at least 6 characters.")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(
            email=email,
            password_hash=get_password_hash(password),
            full_name=name,
            role="superadmin",
            company_id=None,
        )
        db.add(user)
        db.flush()
        db.add(UserStats(user_id=user.id))
        action = "CREATED"
    else:
        user.role = "superadmin"
        user.company_id = None
        user.full_name = name
        user.password_hash = get_password_hash(password)
        user.is_active = True
        action = "PROMOTED"

    db.commit()
    db.refresh(user)
    print(f"\n[{action}] super-admin")
    print(f"  id    = {user.id}")
    print(f"  email = {user.email}")
    print(f"  role  = {user.role}")
    return user


if __name__ == "__main__":
    args = _prompt_args()
    db = SessionLocal()
    try:
        create_superadmin(db, email=args.email, name=args.name, password=args.password)
    finally:
        db.close()
