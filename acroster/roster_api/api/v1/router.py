from fastapi import APIRouter

from .endpoints import roster, history, edits

api_router = APIRouter()

api_router.include_router(
    roster.router,
    prefix="/roster",
    tags=["roster"]
)

api_router.include_router(
    history.router,
    prefix="/history",
    tags=["history"]
)

api_router.include_router(
    edits.router,
    prefix="/edits",
    tags=["edits"]
)