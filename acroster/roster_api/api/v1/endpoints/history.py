import logging

from fastapi import APIRouter, HTTPException, status, Query

from ....api.deps import DBSession, DBOperationsService
from ....schemas.roster import (
    LastInputsResponse,
    RosterHistoryResponse,
    RosterHistoryItem,
    SaveInputsRequest,
    SuccessResponse,
    ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/last-inputs",
    response_model=LastInputsResponse,
    responses={404: {"model": ErrorResponse}}
)
async def get_last_inputs(
        db_session: DBSession,
        db_ops: DBOperationsService
):
    try:
        last_inputs = await db_ops.get_last_inputs(db_session)

        if not last_inputs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No last inputs found"
            )

        return LastInputsResponse(success=True, data=last_inputs)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving last inputs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve last inputs"
        )


@router.post(
    "/last-inputs",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def save_last_inputs(
        request: SaveInputsRequest,
        db_session: DBSession,
        db_ops: DBOperationsService
):
    try:
        inputs = request.model_dump()
        await db_ops.save_last_inputs(db_session, inputs)

        return SuccessResponse(
            success=True,
            message="Last inputs saved successfully"
        )

    except Exception as e:
        logger.error(f"Error saving last inputs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save last inputs"
        )


@router.get(
    "/history",
    response_model=RosterHistoryResponse
)
async def get_roster_history(
        db_session: DBSession,
        db_ops: DBOperationsService,
        limit: int = Query(default=10, ge=1, le=100),
):
    try:
        history_records = await db_ops.get_roster_history(db_session, limit)

        history_items = []
        for record in history_records:
            history_items.append(
                RosterHistoryItem(
                    id=record.id,
                    timestamp=record.timestamp.isoformat(),
                    main_officers=record.main_officers,
                    gl_counters=record.gl_counters,
                    handwritten_counters=record.handwritten_counters,
                    ot_counters=record.ot_counters,
                    ro_ra_officers=record.ro_ra_officers,
                    sos_timings=record.sos_timings,
                    beam_width=record.beam_width,
                    optimization_penalty=record.optimization_penalty,
                    main_officer_count=record.main_officer_count,
                    sos_officer_count=record.sos_officer_count,
                    ot_officer_count=record.ot_officer_count,
                    total_officer_count=record.total_officer_count,
                    notes=record.notes
                )
            )

        return RosterHistoryResponse(
            success=True,
            data=history_items,
            count=len(history_items)
        )

    except Exception as e:
        logger.error(
            f"Error retrieving roster history: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve roster history"
        )
