"""
acroster/db_handlers.py
Database handlers using SQLAlchemy ORM
"""

from .database import get_db_session, LastInputs, RosterHistory
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