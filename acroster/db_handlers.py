"""
acroster/db_handlers.py
Database handlers using SQLAlchemy ORM
"""

from .database import get_db_session, LastInputs, RosterHistory, RosterEdit
from datetime import datetime
from typing import Optional, Dict, Any, List


# ------------------------------
# Save/Load Last Inputs (Global)
# ------------------------------
def save_last_inputs(inputs: Dict[str, Any]):
    """Save the most recent input configuration (shared by all users)"""
    session = get_db_session()
    try:
        last_input = session.query(LastInputs).filter_by(id=1).first()
        if last_input:
            last_input.main_officers = inputs.get('main_officers', '')
            last_input.gl_counters = inputs.get('gl_counters', '')
            last_input.handwritten_counters = inputs.get('handwritten_counters', '')
            last_input.ot_counters = inputs.get('ot_counters', '')
            last_input.ro_ra_officers = inputs.get('ro_ra_officers', '')
            last_input.sos_timings = inputs.get('sos_timings', '')
            last_input.raw_sos_text = inputs.get('raw_sos_text', '') 
            last_input.beam_width = inputs.get('beam_width', 20)
            last_input.updated_at = datetime.utcnow()
            session.commit()
    finally:
        session.close()


def get_last_inputs() -> Optional[Dict[str, Any]]:
    """Get the most recent input configuration"""
    session = get_db_session()
    try:
        last_input = session.query(LastInputs).filter_by(id=1).first()
        if last_input:
            return {
                'main_officers': last_input.main_officers,
                'gl_counters': last_input.gl_counters,
                'handwritten_counters': last_input.handwritten_counters,
                'ot_counters': last_input.ot_counters,
                'ro_ra_officers': last_input.ro_ra_officers,
                'sos_timings': last_input.sos_timings,
                'raw_sos_text': last_input.raw_sos_text or '',
                'beam_width': last_input.beam_width
            }
        return None
    finally:
        session.close()


# ------------------------------
# Roster History
# ------------------------------
def save_roster_history(inputs: Dict[str, Any], results: Dict[str, Any]):
    """Save a roster generation record to history"""
    session = get_db_session()
    try:
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
        return history.id
    finally:
        session.close()

def save_roster_edit(
    edit_type: str,
    officer_id: str,
    slot_start: int,
    slot_end: int,
    time_start: str,
    time_end: str,
    officer_id_2: Optional[str] = None,
    counter_no: Optional[int] = None,
    roster_history_id: Optional[int] = None,
    notes: Optional[str] = None
):
    """Save a roster edit operation to the database
    
    Args:
        edit_type: Type of edit ('delete', 'swap', 'add')
        officer_id: Primary officer identifier
        slot_start: Starting time slot index (0-47)
        slot_end: Ending time slot index (0-47)
        time_start: Human-readable start time (e.g., "10:00")
        time_end: Human-readable end time (e.g., "10:45")
        officer_id_2: Second officer identifier (for swap operations)
        counter_no: Counter number (for add operations)
        roster_history_id: Link to roster history record
        notes: Additional notes
    """
    session = get_db_session()
    try:
        edit = RosterEdit(
            roster_history_id=roster_history_id,
            edit_type=edit_type,
            officer_id=officer_id,
            officer_id_2=officer_id_2,
            counter_no=counter_no,
            slot_start=slot_start,
            slot_end=slot_end,
            time_start=time_start,
            time_end=time_end,
            notes=notes
        )
        session.add(edit)
        session.commit()
        return edit.id
    finally:
        session.close()

def get_roster_history(limit: int = 10) -> List[RosterHistory]:
    """Get recent roster generation history"""
    session = get_db_session()
    try:
        return session.query(RosterHistory)\
            .order_by(RosterHistory.timestamp.desc())\
            .limit(limit)\
            .all()
    finally:
        session.close()


def get_roster_edits(limit: int = 20) -> List[RosterEdit]:
    """Get recent roster edit operations"""
    session = get_db_session()
    try:
        return session.query(RosterEdit)\
            .order_by(RosterEdit.timestamp.desc())\
            .limit(limit)\
            .all()
    finally:
        session.close()

def get_roster_edits_by_history(roster_history_id: int) -> List[RosterEdit]:
    """Get all edit operations for a specific roster history record"""
    session = get_db_session()
    try:
        return session.query(RosterEdit)\
            .filter_by(roster_history_id=roster_history_id)\
            .order_by(RosterEdit.timestamp.asc())\
            .all()
    finally:
        session.close()

def delete_roster_edit(edit_id: int) -> bool:
    """Delete a specific roster edit record"""
    session = get_db_session()
    try:
        edit = session.query(RosterEdit).filter_by(id=edit_id).first()
        if edit:
            session.delete(edit)
            session.commit()
            return True
        return False
    finally:
        session.close()

def clear_all_roster_edits():
    """Clear all roster edit history"""
    session = get_db_session()
    try:
        session.query(RosterEdit).delete()
        session.commit()
    finally:
        session.close()