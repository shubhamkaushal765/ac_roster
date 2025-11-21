"""
acroster/database.py
SQLAlchemy Database Setup for AC Roster Application
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, CheckConstraint, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()


class RosterHistory(Base):
    """Table for storing roster generation history"""
    __tablename__ = 'roster_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    main_officers = Column(Text, nullable=False)
    gl_counters = Column(Text)
    handwritten_counters = Column(Text)
    ot_counters = Column(Text)
    ro_ra_officers = Column(Text)
    sos_timings = Column(Text)
    beam_width = Column(Integer, default=20)
    optimization_penalty = Column(Float)
    main_officer_count = Column(Integer)
    sos_officer_count = Column(Integer)
    ot_officer_count = Column(Integer)
    total_officer_count = Column(Integer)
    notes = Column(Text)


class LastInputs(Base):
    """Table for storing last used inputs (single row)"""
    __tablename__ = 'last_inputs'
    __table_args__ = (CheckConstraint('id = 1', name='single_row_check'),)
    
    id = Column(Integer, primary_key=True, default=1)
    main_officers = Column(Text, nullable=False)
    gl_counters = Column(Text)
    handwritten_counters = Column(Text)
    ot_counters = Column(Text)
    ro_ra_officers = Column(Text)
    sos_timings = Column(Text)
    raw_sos_text = Column(Text)
    beam_width = Column(Integer, default=20)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RosterEdit(Base):
    """Table for storing roster edit operations"""
    __tablename__ = 'roster_edits'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    roster_history_id = Column(Integer)  # Links to RosterHistory if needed
    timestamp = Column(DateTime, default=datetime.utcnow)
    edit_type = Column(String(20), nullable=False)  # 'delete', 'swap', 'add'
    officer_id = Column(String(50))  # Primary officer involved
    officer_id_2 = Column(String(50))  # Second officer (for swap operations)
    counter_no = Column(Integer)  # Counter number (for add operations)
    slot_start = Column(Integer, nullable=False)  # Starting time slot (0-47)
    slot_end = Column(Integer, nullable=False)  # Ending time slot (0-47)
    time_start = Column(String(10))  # Human-readable start time (e.g., "10:00")
    time_end = Column(String(10))  # Human-readable end time (e.g., "10:45")
    notes = Column(Text)  # Additional notes or description


class Database:
    """Handles database connection and initialization using SQLAlchemy"""
    
    def __init__(self, db_path: str = "acroster.db"):
        """Initialize database with SQLAlchemy
        
        Args:
            db_path: Path to SQLite database file
        """
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._init_database()
    
    def _init_database(self):
        """Create tables and initialize default data"""
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        # Initialize last_inputs with default row if empty
        session = self.get_session()
        try:
            existing = session.query(LastInputs).filter_by(id=1).first()
            if not existing:
                default_inputs = LastInputs(
                    id=1,
                    main_officers='1-18',
                    gl_counters='4AC1, 8AC11, 12AC21, 16AC31',
                    handwritten_counters='3AC12,5AC13',
                    ot_counters='2,20,40',
                    ro_ra_officers='3RO2100, 11RO1700,15RO2130',
                    sos_timings='(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, 1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945',
                    beam_width=20
                )
                session.add(default_inputs)
                session.commit()
        finally:
            session.close()
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()


# Singleton instance
_db_instance = None


def get_db_instance(db_path: str = "acroster.db"):
    """Get or create database singleton instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance


def get_db_session(db_path: str = "acroster.db"):
    """Get a new database session (used by handlers)"""
    return get_db_instance(db_path).get_session()