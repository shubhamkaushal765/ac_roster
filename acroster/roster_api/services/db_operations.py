import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from acroster.database import LastInputs, RosterHistory, RosterEdit

logger = logging.getLogger(__name__)


class DatabaseOperationsService:

    async def save_last_inputs(
            self,
            session,
            inputs: Dict
    ) -> None:
        last_input = session.query(LastInputs).filter_by(id=1).first()
        if last_input:
            last_input.main_officers = inputs.get('main_officers', '')
            last_input.gl_counters = inputs.get('gl_counters', '')
            last_input.handwritten_counters = inputs.get(
                'handwritten_counters', ''
            )
            last_input.ot_counters = inputs.get('ot_counters', '')
            last_input.ro_ra_officers = inputs.get('ro_ra_officers', '')
            last_input.sos_timings = inputs.get('sos_timings', '')
            last_input.raw_sos_text = inputs.get('raw_sos_text', '')
            last_input.beam_width = inputs.get('beam_width', 20)
            last_input.updated_at = datetime.utcnow()
            session.commit()
            logger.info("Updated last inputs")
        else:
            logger.warning("Last inputs record not found")

    async def get_last_inputs(self, session) -> Optional[Dict]:
        last_input = session.query(LastInputs).filter_by(id=1).first()
        if last_input:
            return {
                'main_officers':        last_input.main_officers,
                'gl_counters':          last_input.gl_counters or '',
                'handwritten_counters': last_input.handwritten_counters or '',
                'ot_counters':          last_input.ot_counters or '',
                'ro_ra_officers':       last_input.ro_ra_officers or '',
                'sos_timings':          last_input.sos_timings or '',
                'raw_sos_text':         last_input.raw_sos_text or '',
                'beam_width':           last_input.beam_width
            }
        return None

    async def save_roster_history(
            self,
            session,
            inputs: Dict,
            results: Dict
    ) -> int:
        history = RosterHistory(
            main_officers=inputs.get('main_officers', ''),
            gl_counters=inputs.get('gl_counters', ''),
            handwritten_counters=inputs.get('handwritten_counters', ''),
            ot_counters=inputs.get('ot_counters', ''),
            ro_ra_officers=inputs.get('ro_ra_officers', ''),
            sos_timings=inputs.get('sos_timings', ''),
            beam_width=inputs.get('beam_width', 20),
            optimization_penalty=results.get('optimization_penalty'),
            main_officer_count=results.get('main_officer_count', 0),
            sos_officer_count=results.get('sos_officer_count', 0),
            ot_officer_count=results.get('ot_officer_count', 0),
            total_officer_count=results.get('total_officer_count', 0),
            notes=results.get('notes', '')
        )
        session.add(history)
        session.commit()
        logger.info(f"Saved roster history with id={history.id}")
        return history.id

    async def get_roster_history(
            self,
            session,
            limit: int = 10
    ) -> List[RosterHistory]:
        return session.query(RosterHistory) \
            .order_by(RosterHistory.timestamp.desc()) \
            .limit(limit) \
            .all()

    async def save_roster_edit(
            self,
            session,
            edit_data: Dict
    ) -> int:
        edit = RosterEdit(
            roster_history_id=edit_data.get('roster_history_id'),
            edit_type=edit_data['edit_type'],
            officer_id=edit_data['officer_id'],
            officer_id_2=edit_data.get('officer_id_2'),
            counter_no=edit_data.get('counter_no'),
            slot_start=edit_data['slot_start'],
            slot_end=edit_data['slot_end'],
            time_start=edit_data['time_start'],
            time_end=edit_data['time_end'],
            notes=edit_data.get('notes')
        )
        session.add(edit)
        session.commit()
        logger.info(f"Saved roster edit with id={edit.id}")
        return edit.id

    async def get_roster_edits(
            self,
            session,
            limit: int = 20
    ) -> List[RosterEdit]:
        return session.query(RosterEdit) \
            .order_by(RosterEdit.timestamp.desc()) \
            .limit(limit) \
            .all()

    async def delete_roster_edit(
            self,
            session,
            edit_id: int
    ) -> bool:
        edit = session.query(RosterEdit).filter_by(id=edit_id).first()
        if edit:
            session.delete(edit)
            session.commit()
            logger.info(f"Deleted roster edit with id={edit_id}")
            return True
        logger.warning(f"Roster edit with id={edit_id} not found")
        return False

    async def clear_all_roster_edits(self, session) -> None:
        session.query(RosterEdit).delete()
        session.commit()
        logger.info("Cleared all roster edits")


db_operations_service = DatabaseOperationsService()
