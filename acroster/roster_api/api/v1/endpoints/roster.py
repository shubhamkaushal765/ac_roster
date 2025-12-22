import logging

from fastapi import APIRouter, HTTPException, status

from ....api.deps import RosterService, DBSession, DBOperationsService
from ....schemas.roster import (
    RosterGenerationRequest,
    RosterGenerationResponse,
    StatisticsData,
    ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate",
    response_model=RosterGenerationResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def generate_roster(
        request: RosterGenerationRequest,
        roster_service: RosterService,
        db_session: DBSession,
        db_ops: DBOperationsService
):
    try:
        roster_data, officer_counts, optimization_penalty, (
            stats1, stats2) = await roster_service.generate_roster(
            mode=request.mode,
            main_officers_reported=request.main_officers_reported,
            report_gl_counters=request.report_gl_counters,
            handwritten_counters=request.handwritten_counters,
            ot_counters=request.ot_counters,
            ro_ra_officers=request.ro_ra_officers,
            sos_timings=request.sos_timings,
            beam_width=request.beam_width
        )

        if request.save_to_history:
            inputs = {
                'main_officers':        request.main_officers_reported,
                'gl_counters':          request.report_gl_counters,
                'handwritten_counters': request.handwritten_counters,
                'ot_counters':          request.ot_counters,
                'ro_ra_officers':       request.ro_ra_officers,
                'sos_timings':          request.sos_timings,
                'beam_width':           request.beam_width
            }

            results = {
                'optimization_penalty': optimization_penalty,
                'main_officer_count':   officer_counts.main,
                'sos_officer_count':    officer_counts.sos,
                'ot_officer_count':     officer_counts.ot,
                'total_officer_count':  officer_counts.total
            }

            await db_ops.save_last_inputs(db_session, inputs)
            await db_ops.save_roster_history(db_session, inputs, results)

        return RosterGenerationResponse(
            success=True,
            data=roster_data,
            officer_counts=officer_counts,
            optimization_penalty=optimization_penalty,
            statistics=StatisticsData(stats1=stats1, stats2=stats2)
        )

    except ValueError as e:
        logger.error(f"Validation error in roster generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating roster: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate roster"
        )
