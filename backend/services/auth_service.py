# backend/services/auth_service.py
"""
Authentication service - handles user registration, login, and JWT.
"""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.config import get_settings
from backend.database import get_db
from backend.models import User, UserStats
from backend.schemas import UserCreate, UserResponse, Token, TokenData

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = {"sub": user_id}
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return TokenData(user_id=user_id)
    except JWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user.
    
    Usage:
        @app.get("/me")
        def get_me(user: User = Depends(get_current_user)):
            return user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_token(token)
    if token_data is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Like get_current_user, but returns None instead of 401 when no valid token
    is present. Used by endpoints that are public but tailor their response for
    authenticated users (e.g. persona listing with plan-aware lock flags).
    """
    if not token:
        return None
    token_data = decode_token(token)
    if token_data is None:
        return None
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None or not user.is_active:
        return None
    return user


def register_user(db: Session, user_data: UserCreate) -> User:
    """Register a new user.

    If an invite_code is supplied, the new user is attached to that company as the
    invite's role (consuming the invite). An invalid/expired code rejects the whole
    registration — no solo account is created. Without a code, the user is a solo
    salesperson (joining a company by company name is not supported; use an invite).
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Resolve the invite up-front so an invalid code fails before we create
    # anything. Imported here to avoid a circular import at module load.
    invite = None
    if user_data.invite_code:
        from backend.services.seat_service import validate_invite_code, count_active_seats, seat_limit_for
        invite = validate_invite_code(db, code=user_data.invite_code)
        # Enforce the seat limit at registration time too.
        if count_active_seats(db, invite.company_id) >= seat_limit_for(db, invite.company_id):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Seat limit reached")

    # Create user
    user = User(
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        company=user_data.company,
        company_id=invite.company_id if invite else None,
        role=(invite.role or "salesperson") if invite else "salesperson",
        experience_level=user_data.experience_level
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    # Create empty stats
    stats = UserStats(user_id=user.id)
    db.add(stats)
    db.commit()

    # Consume the invite now that the user exists (audited in the same session).
    if invite is not None:
        from backend.services.seat_service import consume_invite_for_new_user
        consume_invite_for_new_user(db, user=user, invite=invite)
        db.refresh(user)

    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password."""
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.password_hash):
        return None
    
    return user


def login_user(db: Session, email: str, password: str) -> Token:
    """Login a user and return a token."""
    user = authenticate_user(db, email, password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    access_token = create_access_token(str(user.id))

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse.model_validate(user)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Role-based access control + tenant scoping helpers
# Built strictly on top of get_current_user above.
# ─────────────────────────────────────────────────────────────────────────────

def require_role(*allowed_roles: str):
    """Dependency factory: 403 unless current_user.role in allowed_roles."""
    async def _dep(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Insufficient permissions")
        return current_user
    return _dep


get_current_manager = require_role("manager", "superadmin")
get_current_superadmin = require_role("superadmin")


def assert_same_company(current_user: User, target_company_id) -> None:
    """Raise 403 unless superadmin or the user's company matches target."""
    if current_user.role == "superadmin":
        return
    if str(current_user.company_id) != str(target_company_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Cross-tenant access denied")