import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from acroster.database import get_db_session
from ..core.config import settings


class DatabaseService:
    def __init__(self):
        self.db_path = settings.DATABASE_PATH

    def get_session(self):
        return get_db_session(self.db_path)


db_service = DatabaseService()


async def get_db():
    session = db_service.get_session()
    try:
        yield session
    finally:
        session.close()
