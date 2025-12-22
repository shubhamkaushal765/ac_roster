import logging

from fastapi import APIRouter, HTTPException, status, Query, Path

from ....api.deps import DBSession, DBOperationsService
from ....schemas.roster import (
    RosterEditCreate,
    RosterEditResponse,
    RosterEditsListResponse,
    SuccessResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=RosterEditResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}}
)
async def create_roster_edit(
        request: RosterEditCreate,
        db_session: DBSession,
        db_ops: DBOperationsService
):
    try:
        edit_data = request.model_dump()
        edit_id = await db_ops.save_roster_edit(db_session, edit_data)

        return RosterEditResponse(
            success=True,
            edit_id=edit_id,
            message="Edit saved successfully"
        )

    except Exception as e:
        logger.error(f"Error creating roster edit: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save roster edit"
        )


@router.get(
    "/",
    response_model=RosterEditsListResponse
)
async def get_roster_edits(
        db_session: DBSession,
        db_ops: DBOperationsService,
        limit: int = Query(default=20, ge=1, le=100),
):
    try:
        edits = await db_ops.get_roster_edits(db_session, limit)

        edit_list = []
        for edit in edits:
            edit_list.append(
                {
                    "id":                edit.id,
                    "timestamp":         edit.timestamp.isoformat(),
                    "edit_type":         edit.edit_type,
                    "officer_id":        edit.officer_id,
                    "officer_id_2":      edit.officer_id_2,
                    "counter_no":        edit.counter_no,
                    "slot_start":        edit.slot_start,
                    "slot_end":          edit.slot_end,
                    "time_start":        edit.time_start,
                    "time_end":          edit.time_end,
                    "roster_history_id": edit.roster_history_id,
                    "notes":             edit.notes
                }
            )

        return RosterEditsListResponse(
            success=True,
            data=edit_list,
            count=len(edit_list)
        )

    except Exception as e:
        logger.error(f"Error retrieving roster edits: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve roster edits"
        )


@router.delete(
    "/{edit_id}",
    response_model=SuccessResponse,
    responses={404: {"model": ErrorResponse}}
)
async def delete_roster_edit(
        edit_id: int = Path(ge=1),
        db_session: DBSession = None,
        db_ops: DBOperationsService = None
):
    try:
        success = await db_ops.delete_roster_edit(db_session, edit_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Roster edit with id {edit_id} not found"
            )

        return SuccessResponse(
            success=True,
            message=f"Roster edit {edit_id} deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting roster edit: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete roster edit"
        )


@router.delete(
    "/",
    response_model=SuccessResponse
)
async def clear_all_roster_edits(
        db_session: DBSession,
        db_ops: DBOperationsService
):
    try:
        await db_ops.clear_all_roster_edits(db_session)

        return SuccessResponse(
            success=True,
            message="All roster edits cleared successfully"
        )

    except Exception as e:
        logger.error(f"Error clearing roster edits: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear roster edits"
        )
