"""Database access layer."""

from fraudlens.database.connection import get_engine, session_scope
from fraudlens.database.models import Base

__all__ = ["Base", "get_engine", "session_scope"]
