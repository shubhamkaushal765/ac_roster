from fastapi import Depends
from typing import Annotated
from ..services.database import get_db
from ..services.roster import roster_generation_service
from ..services.db_operations import db_operations_service


async def get_roster_service():
    return roster_generation_service


async def get_db_operations_service():
    return db_operations_service


DBSession = Annotated[any, Depends(get_db)]
RosterService = Annotated[any, Depends(get_roster_service)]
DBOperationsService = Annotated[any, Depends(get_db_operations_service)]