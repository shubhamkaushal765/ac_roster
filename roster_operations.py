"""
Business logic for roster operations (generation, editing, visualization)
"""

import numpy as np
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from acroster.config import NUM_SLOTS, START_HOUR, MODE_CONFIG, OperationMode
from acroster.orchestrator_pipe import RosterAlgorithmOrchestrator
from acroster.plotter import Plotter
from acroster.statistics import StatisticsGenerator
from acroster.time_utils import hhmm_to_slot, extract_officer_timings
from acroster.db_handlers import save_last_inputs


class RosterOperations:
    """Core business logic for roster generation and editing"""
    
    def __init__(self):
        self.current_orchestrator: Optional[RosterAlgorithmOrchestrator] = None
        self.current_values: Optional[Dict[str, Any]] = None
        self.edited_schedule: Optional[Dict] = None
        self.timetable_history = []  # List of (fig, stats, timestamp, description)
        self.schedule_history = []   # List of (fig, stats, timestamp, description)
    
    def generate_schedule(self, values: Dict[str, Any]) -> Tuple:
        """
        Generate initial roster schedule
        
        Returns:
            Tuple of (counter_matrix, final_counter_matrix, officer_schedule, output_text)
        """
        orchestrator = RosterAlgorithmOrchestrator(
            mode=OperationMode(values['operation_mode'])
        )
        
        counter_matrix, final_counter_matrix, officer_schedule, output_text = orchestrator.run(
            main_officers_reported=values['main_officers'],
            report_gl_counters=values['gl_counters'],
            sos_timings="",
            ro_ra_officers=values['ro_ra_officers'],
            handwritten_counters=values['handwritten_counters'],
            ot_counters=values['ot_counters'],
        )
        
        # Store state
        self.current_orchestrator = orchestrator
        self.current_values = values
        self.edited_schedule = officer_schedule.copy()
        
        # Save inputs to DB
        save_last_inputs({
            'main_officers': values['main_officers'],
            'gl_counters': values['gl_counters'],
            'handwritten_counters': values['handwritten_counters'],
            'ot_counters': values['ot_counters'],
            'ro_ra_officers': values['ro_ra_officers'],
            'beam_width': values['beam_width'],
        })
        
        # Initialize history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._add_to_history(
            counter_matrix, 
            officer_schedule, 
            output_text[0],
            timestamp,
            "Initial schedule generated"
        )
        
        return counter_matrix, final_counter_matrix, officer_schedule, output_text
    
    def add_sos_officers(self, sos_timings: str) -> Tuple:
        """
        Re-run orchestrator with SOS officers
        
        Returns:
            Tuple of (counter_matrix, officer_schedule, stats, description)
        """
        orchestrator = RosterAlgorithmOrchestrator(
            mode=OperationMode(self.current_values['operation_mode'])
        )
        
        counter_matrix, final_counter_matrix, officer_schedule, output_text = orchestrator.run(
            main_officers_reported=self.current_values['main_officers'],
            report_gl_counters=self.current_values['gl_counters'],
            sos_timings=sos_timings,
            ro_ra_officers=self.current_values['ro_ra_officers'],
            handwritten_counters=self.current_values['handwritten_counters'],
            ot_counters=self.current_values['ot_counters'],
        )
        
        # Update state
        self.current_orchestrator = orchestrator
        self.edited_schedule = officer_schedule.copy()
        
        # Build description
        sos_officers = orchestrator.get_sos_officers()
        description = f"Added {len(sos_officers)} SOS officer(s):\n"
        for sos in sos_officers:
            description += f"• {sos.officer_key} at counter {sos.pre_assigned_counter}\n"
        
        # Add to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._add_to_history(
            final_counter_matrix, 
            officer_schedule, 
            output_text[0],
            timestamp,
            description
        )
        
        return final_counter_matrix, officer_schedule, output_text[0], description
    
    def swap_assignments(self, officer1: str, officer2: str, 
                        start_time: str, end_time: str) -> str:
        """
        Swap counter assignments between two officers
        
        Returns:
            Description of the swap operation
        """
        start_slot = hhmm_to_slot(start_time)
        end_slot = hhmm_to_slot(end_time)
        
        if start_slot >= end_slot:
            raise ValueError("Start time must be before end time")
        
        if officer1 not in self.edited_schedule or officer2 not in self.edited_schedule:
            raise ValueError("Officer not found in schedule")
        
        # Perform swap
        temp = self.edited_schedule[officer1][start_slot:end_slot].copy()
        self.edited_schedule[officer1][start_slot:end_slot] = \
            self.edited_schedule[officer2][start_slot:end_slot]
        self.edited_schedule[officer2][start_slot:end_slot] = temp
        
        description = f"Swapped officers {officer1} and {officer2} from {start_time} to {end_time}"
        
        # Update visualizations
        self._update_after_edit(description)
        
        return description
    
    def delete_assignment(self, officer: str, start_time: str, end_time: str) -> str:
        """
        Delete counter assignments for an officer
        
        Returns:
            Description of the delete operation
        """
        start_slot = hhmm_to_slot(start_time)
        end_slot = hhmm_to_slot(end_time)
        
        if start_slot >= end_slot:
            raise ValueError("Start time must be before end time")
        
        if officer not in self.edited_schedule:
            raise ValueError("Officer not found in schedule")
        
        # Delete assignments
        self.edited_schedule[officer][start_slot:end_slot] = 0
        
        description = f"Deleted officer {officer} from {start_time} to {end_time}"
        
        # Update visualizations
        self._update_after_edit(description)
        
        return description
    
    def _update_after_edit(self, description: str):
        """Update history after manual edit"""
        config = MODE_CONFIG[OperationMode(self.current_values['operation_mode'])]
        
        # Convert edited schedule back to counter matrix
        edited_counter_matrix = np.zeros((config['num_counters'], NUM_SLOTS), dtype=object)
        
        for officer_id, schedule in self.edited_schedule.items():
            for slot_idx, counter_no in enumerate(schedule):
                if counter_no != 0:
                    counter_idx = counter_no - 1
                    if 0 <= counter_idx < config['num_counters']:
                        edited_counter_matrix[counter_idx, slot_idx] = officer_id
        
        # Calculate stats
        stats_generator = StatisticsGenerator(OperationMode(self.current_values['operation_mode']))
        stats = stats_generator.generate_statistics(edited_counter_matrix)
        
        # Add to history
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._add_to_history(
            edited_counter_matrix,
            self.edited_schedule,
            stats,
            timestamp,
            description
        )
    
    def _add_to_history(self, counter_matrix, officer_schedule, stats, timestamp, description):
        """Add visualizations to history"""
        config = MODE_CONFIG[OperationMode(self.current_values['operation_mode'])]
        
        plotter = Plotter(
            num_slots=NUM_SLOTS,
            num_counters=config['num_counters'],
            start_hour=START_HOUR,
        )
        
        fig1 = plotter.plot_officer_timetable_with_labels(counter_matrix)
        fig2 = plotter.plot_officer_schedule_with_labels(officer_schedule)
        
        self.timetable_history.append((fig1, stats, timestamp, description))
        self.schedule_history.append((fig2, None, timestamp, description))
    
    def get_officer_counts(self) -> Dict[str, int]:
        """Get officer count metrics"""
        if not self.current_orchestrator:
            return {'main': 0, 'sos': 0, 'ot': 0, 'total': 0}
        return self.current_orchestrator.get_officer_counts()
    
    def extract_sos_from_text(self, raw_text: str) -> str:
        """Extract SOS timings from raw text"""
        return extract_officer_timings(raw_text)