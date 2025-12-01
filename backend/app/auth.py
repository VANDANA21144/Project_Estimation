# backend/app/auth.py
import os
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def _get_admin_token() -> str:
    return os.getenv("ADMIN_TOKEN", "supersecretadmintoken")

def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency that verifies a Bearer token against ADMIN_TOKEN.
    Raises HTTPException(401) on failure.
    """
    if not credentials or not credentials.scheme or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing credentials")
    token = credentials.credentials
    if token != _get_admin_token():
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return {"admin": True}
