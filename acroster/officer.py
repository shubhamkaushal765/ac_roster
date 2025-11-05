from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class Officer(ABC):
    """Abstract base class for all officer types"""

    def __init__(self, officer_id: int, num_slots: int = 48):
        self.officer_id = officer_id
        self.num_slots = num_slots
        self.schedule = np.zeros(num_slots, dtype=int)

    @property
    @abstractmethod
    def officer_key(self) -> str:
        """Return the string key for this officer (e.g., 'M1', 'S1', 'OT1')"""
        pass

    def assign_counter(self, slot: int, counter: int):
        """Assign a counter to a specific slot"""
        if 0 <= slot < self.num_slots:
            self.schedule[slot] = counter

    def assign_counter_range(
            self, start_slot: int, end_slot: int, counter: int
    ):
        """Assign a counter to a range of slots (inclusive)"""
        self.schedule[start_slot:end_slot + 1] = counter

    def clear_slot(self, slot: int):
        """Clear assignment at a specific slot"""
        self.schedule[slot] = 0

    def clear_range(self, start_slot: int, end_slot: int):
        """Clear assignments in a range of slots (inclusive)"""
        self.schedule[start_slot:end_slot + 1] = 0

    def get_schedule(self) -> np.ndarray:
        """Return the schedule array"""
        return self.schedule.copy()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.officer_key})"


class MainOfficer(Officer):
    """Main officer (M1-M40)"""

    def __init__(
            self, officer_id: int, roster_template: np.ndarray,
            num_slots: int = 48
    ):
        super().__init__(officer_id, num_slots)
        self.schedule = roster_template.copy()
        self.late_arrival_slot: Optional[int] = None
        self.early_departure_slot: Optional[int] = None

    @property
    def officer_key(self) -> str:
        return f"M{self.officer_id}"

    def apply_late_arrival(self, arrival_slot: int):
        """Clear schedule before arrival time (RA - Report After)"""
        self.late_arrival_slot = arrival_slot
        self.schedule[:arrival_slot] = 0

    def apply_early_departure(self, departure_slot: int):
        """Clear schedule after departure time (RO - Report Out)"""
        self.early_departure_slot = departure_slot
        self.schedule[departure_slot:] = 0

    def apply_last_counter(
            self, last_counter_start_slot: int, counter_no: int
    ):
        """Assign last counter before departure (for officers with id % 4 == 3)"""
        if self.officer_id % 4 == 3 and last_counter_start_slot >= 42:
            self.schedule[42:last_counter_start_slot] = counter_no

    def apply_ground_level_counter(self, counter_no: int):
        """Apply ground level counter to first 5 slots (for officers with id % 4 == 0)"""
        if self.officer_id % 4 == 0:
            self.schedule[0:5] = counter_no

    def apply_takeover_counter(self, counter_no: int):
        """Apply takeover counter to first 2 slots"""
        self.schedule[0:2] = counter_no


class SOSOfficer(Officer):
    """SOS officer (S1-Sn)"""

    def __init__(
            self, officer_id: int, availability_schedule: np.ndarray,
            num_slots: int = 48
    ):
        super().__init__(officer_id, num_slots)
        self.availability_schedule = availability_schedule.copy()  # 1 = available, 0 = not available
        self.break_schedules: List[
            np.ndarray] = []  # Multiple possible break schedules
        self.selected_schedule_index: Optional[int] = None
        self.pre_assigned_counter: Optional[int] = None

    @property
    def officer_key(self) -> str:
        return f"S{self.officer_id}"

    def set_pre_assigned_counter(self, counter_no: int):
        """Set a pre-assigned counter for this officer"""
        self.pre_assigned_counter = counter_no

    def add_break_schedule(self, break_schedule: np.ndarray):
        """Add a valid break schedule option (1 = working, 0 = break)"""
        self.break_schedules.append(break_schedule.copy())

    def select_schedule(self, schedule_index: int):
        """Select which break schedule to use"""
        if 0 <= schedule_index < len(self.break_schedules):
            self.selected_schedule_index = schedule_index
            self.schedule = self.break_schedules[schedule_index].copy()
        else:
            raise ValueError(f"Invalid schedule index {schedule_index}")

    def get_working_intervals(self) -> List[Tuple[int, int]]:
        """Get list of (start, end) intervals where officer is working"""
        if self.selected_schedule_index is None:
            return []

        intervals = []
        schedule = self.break_schedules[self.selected_schedule_index]
        i = 0
        while i < len(schedule):
            if schedule[i] == 1:
                start = i
                while i < len(schedule) and schedule[i] == 1:
                    i += 1
                end = i - 1
                intervals.append((start, end))
            else:
                i += 1
        return intervals


class OTOfficer(Officer):
    """Overtime officer (OT1-OTn)"""

    def __init__(self, officer_id: int, counter_no: int, num_slots: int = 48):
        super().__init__(officer_id, num_slots)
        self.counter_no = counter_no
        # OT officers work first 2 slots at assigned counter
        self.schedule[0:2] = counter_no

    @property
    def officer_key(self) -> str:
        return f"OT{self.officer_id}"
