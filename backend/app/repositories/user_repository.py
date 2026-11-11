from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import User


class UserRepository:
    """Data access only — no password hashing, no HTTP concerns. Those live in services/."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: int) -> User | None:
        return self.db.get(User, user_id)

    def get_by_username(self, username: str) -> User | None:
        return self.db.scalar(select(User).where(User.username == username))

    def get_by_email(self, email: str) -> User | None:
        return self.db.scalar(select(User).where(User.email == email))

    def get_by_username_or_email(self, identifier: str) -> User | None:
        return self.db.scalar(
            select(User).where(or_(User.username == identifier, User.email == identifier))
        )