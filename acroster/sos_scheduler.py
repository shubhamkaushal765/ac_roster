"""
SOS (Support Officer Scheduling) - Handles break schedule generation and management.

This module manages SOS officer availability, break schedule generation,
and constraint validation.
"""

import re
from typing import Dict, List, Tuple
import numpy as np

from acroster.officer import SOSOfficer
from acroster.time_utils import hhmm_to_slot
from acroster.config import NUM_SLOTS


class AvailabilityParser:
    """Parses and converts SOS officer availability strings."""

    @staticmethod
    def parse_availability(avail_str: str) -> np.ndarray:
        """
        Convert availability string into binary numpy array.

        Args:
            avail_str: String like "1000-1300, 2000-2200" or "1315-1430;2030-2200"

        Returns:
            Binary array where 1 = available, 0 = not available
        """
        schedule = np.zeros(NUM_SLOTS, dtype=int)

        # Support both semicolon and comma separators
        separator = ';' if ';' in avail_str else ','

        for rng in avail_str.split(separator):
            rng = rng.strip()

            # Skip empty ranges
            if not rng:
                continue

            # Validate format
            if '-' not in rng:
                print(f"Warning: Invalid range format '{rng}' - missing hyphen, skipping")
                continue

            parts = rng.split("-")
            if len(parts) != 2:
                print(f"Warning: Invalid range format '{rng}' - too many hyphens, skipping")
                continue

            start, end = parts[0].strip(), parts[1].strip()

            # Skip if either time is empty
            if not start or not end:
                print(f"Warning: Empty time value in range '{rng}', skipping")
                continue

            try:
                start_slot = hhmm_to_slot(start)
                end_slot = hhmm_to_slot(end)
                schedule[start_slot: end_slot + 1] = 1
            except ValueError as e:
                print(f"Warning: Could not parse range '{rng}': {e}")
                continue

        return schedule

    @staticmethod
    def convert_input(extracted_data: str) -> List[str]:
        """
        Convert extracted officer timing data to list of availability strings.

        Args:
            extracted_data: Comma-separated string like "1000-1300, 2000-2200, 1315-1430;2030-2200"

        Returns:
            List of timing strings for each officer
        """
        if isinstance(extracted_data, str):
            timings = [timing.strip() for timing in extracted_data.split(',') if timing.strip()]
            return timings

        # Fallback: Handle list input
        result = []
        for item in extracted_data:
            if isinstance(item, dict):
                timing = item.get('timing', '').strip()
            elif isinstance(item, str):
                timing = item.strip()
            else:
                timing = ''

            if timing:
                result.append(timing)

        return result


class SOSOfficerBuilder:
    """Builds SOS officer objects from input strings."""

    def __init__(self):
        """Initialize SOS officer builder."""
        self.parser = AvailabilityParser()

    def build_sos_officers(
            self,
            user_input: str
    ) -> Tuple[List[SOSOfficer], Dict[int, int]]:
        """
        Build SOSOfficer objects from user input.

        Args:
            user_input: String with officer timings and optional counter assignments
                       e.g., "(AC22)1000-1300, 2000-2200, (AC23)1000-1130"

        Returns:
            Tuple of:
                - List of SOSOfficer objects
                - Dict of pre-assigned counters {officer_index: counter_no}
        """
        input_avail = self.parser.convert_input(user_input)
        sos_officers: List[SOSOfficer] = []
        pre_assigned_counter_dict = {}

        for idx, avail in enumerate(input_avail):
            # Search for pre-assigned counters like (AC4)
            matches = re.findall(r"\(AC(\d+)\)", avail, flags=re.IGNORECASE)
            if matches:
                pre_assigned_counter_dict[idx] = int(matches[0])
                avail = re.sub(r"\(AC\d+\)", "", avail, flags=re.IGNORECASE)

            availability_schedule = self.parser.parse_availability(avail)
            officer = SOSOfficer(
                officer_id=idx + 1,
                availability_schedule=availability_schedule
            )

            if idx in pre_assigned_counter_dict:
                officer.set_pre_assigned_counter(pre_assigned_counter_dict[idx])

            sos_officers.append(officer)

        return sos_officers, pre_assigned_counter_dict


class BreakScheduleGenerator:
    """Generates valid break schedules for SOS officers with constraints."""

    MAX_CONSECUTIVE_SLOTS = 10  # Max consecutive working slots without break

    @staticmethod
    def is_valid_sliding_window(schedule: np.ndarray) -> bool:
        """
        Check if schedule satisfies sliding window constraint.

        Args:
            schedule: Binary array where 1 = working, 0 = break

        Returns:
            True if no more than MAX_CONSECUTIVE_SLOTS consecutive 1s
        """
        consec = 0
        for x in schedule:
            if x == 1:
                consec += 1
                if consec > BreakScheduleGenerator.MAX_CONSECUTIVE_SLOTS:
                    return False
            else:
                consec = 0
        return True

    def generate_break_schedules(
            self,
            sos_officers: List[SOSOfficer]
    ) -> List[SOSOfficer]:
        """
        Generate break schedules for all SOS officers.

        Args:
            sos_officers: List of SOSOfficer objects

        Returns:
            Updated list with break schedules generated
        """
        for officer in sos_officers:
            self._generate_officer_schedules(officer)

        return sos_officers

    def _generate_officer_schedules(self, officer: SOSOfficer):
        """Generate all valid break schedules for a single officer."""
        base = officer.availability_schedule.copy()
        work_slots = np.where(base == 1)[0]

        if len(work_slots) == 0:
            officer.add_break_schedule(base.copy())
            return

        # Build consecutive working stretches
        stretches = self._build_stretches(work_slots)

        # If all stretches ≤ MAX_CONSECUTIVE_SLOTS, schedule is valid
        if all(len(stretch) <= self.MAX_CONSECUTIVE_SLOTS for stretch in stretches):
            officer.add_break_schedule(base.copy())
            return

        # Generate schedules with breaks
        valid_schedules = []
        seen_schedules = set()

        self._place_breaks(
            base.copy(),
            stretches,
            valid_schedules,
            seen_schedules
        )

        if valid_schedules:
            for sched in valid_schedules:
                officer.add_break_schedule(sched)
        else:
            officer.add_break_schedule(base.copy())

    @staticmethod
    def _build_stretches(work_slots: np.ndarray) -> List[List[int]]:
        """Build list of consecutive working slot stretches."""
        stretches = []
        cur = [work_slots[0]]

        for s in work_slots[1:]:
            if s == cur[-1] + 1:
                cur.append(s)
            else:
                stretches.append(cur)
                cur = [s]
        stretches.append(cur)

        return stretches

    def _place_breaks(
            self,
            base_schedule: np.ndarray,
            stretches: List[List[int]],
            valid_schedules: List[np.ndarray],
            seen_schedules: set,
            stretch_idx: int = 0,
            last_break_end: int = -1,
            last_break_len: int = 0
    ):
        """
        Recursively place breaks in stretches to satisfy constraints.

        Uses depth-first search with backtracking to find valid break placements.
        """
        if stretch_idx >= len(stretches):
            self._finalize_schedule(
                base_schedule,
                valid_schedules,
                seen_schedules,
                last_break_end,
                last_break_len
            )
            return

        stretch = stretches[stretch_idx]
        min_slot, max_slot = stretch[0], stretch[-1]
        stretch_len = len(stretch)

        # If stretch is short enough, move to next
        if stretch_len <= self.MAX_CONSECUTIVE_SLOTS:
            self._place_breaks(
                base_schedule,
                stretches,
                valid_schedules,
                seen_schedules,
                stretch_idx + 1,
                last_break_end,
                last_break_len
            )
            return

        # Determine break pattern based on stretch length
        if stretch_len >= 36:
            pattern = [2, 3, 3]
        elif stretch_len >= 20:
            pattern = [2, 3]
        elif stretch_len >= 10:
            pattern = [2]
        else:
            pattern = [0]

        # Recursively insert breaks
        self._recurse_breaks(
            base_schedule,
            stretches,
            valid_schedules,
            seen_schedules,
            stretch_idx,
            min_slot,
            max_slot,
            pattern,
            last_break_end,
            last_break_len
        )

    def _recurse_breaks(
            self,
            schedule: np.ndarray,
            stretches: List[List[int]],
            valid_schedules: List[np.ndarray],
            seen_schedules: set,
            stretch_idx: int,
            min_slot: int,
            max_slot: int,
            blens: List[int],
            last_break_end: int,
            last_break_len: int
    ):
        """Recursively try break placements."""
        if not blens:
            self._place_breaks(
                schedule,
                stretches,
                valid_schedules,
                seen_schedules,
                stretch_idx + 1,
                last_break_end,
                last_break_len
            )
            return

        blen = blens[0]
        interval_start = min_slot if last_break_end < 0 else last_break_end + 1

        max_consec_start = interval_start + self.MAX_CONSECUTIVE_SLOTS
        max_allowed = min(max_consec_start, max_slot - blen - 3)

        for s in range(interval_start + 4, max_allowed + 1):
            # Check gap requirement
            required_gap = min(2 * last_break_len, 4) if last_break_end >= 0 else 0
            if s - last_break_end - 1 < required_gap:
                continue

            # Validate break can be placed
            if not np.all(schedule[s: s + blen] == 1):
                continue

            # Place break
            new_sched = schedule.copy()
            new_sched[s: s + blen] = 0

            self._recurse_breaks(
                new_sched,
                stretches,
                valid_schedules,
                seen_schedules,
                stretch_idx,
                min_slot,
                max_slot,
                blens[1:],
                s + blen,
                blen
            )

    def _finalize_schedule(
            self,
            schedule: np.ndarray,
            valid_schedules: List[np.ndarray],
            seen_schedules: set,
            last_break_end: int,
            last_break_len: int
    ):
        """Finalize schedule by checking validity and adding to results."""
        if self.is_valid_sliding_window(schedule):
            sig = schedule.tobytes()
            if sig not in seen_schedules:
                seen_schedules.add(sig)
                valid_schedules.append(schedule)
            return

        # Try inserting a 1-slot break as last resort
        for s in range(len(schedule)):
            if schedule[s] != 1:
                continue

            # Find interval boundaries
            next_break_index = s
            while next_break_index < len(schedule) and schedule[next_break_index] == 1:
                next_break_index += 1
            interval_end = next_break_index - 1

            prev_break_index = s
            while prev_break_index >= 0 and schedule[prev_break_index] == 1:
                prev_break_index -= 1
            interval_start = prev_break_index + 1

            # Check if break position is valid
            if s <= interval_start + 4 or s >= interval_end - 4:
                continue

            required_gap = min(2 * last_break_len, 4) if last_break_end >= 0 else 0
            if s - last_break_end - 1 < required_gap:
                continue

            # Try placing 1-slot break
            cand = schedule.copy()
            cand[s] = 0

            if self.is_valid_sliding_window(cand):
                sig = cand.tobytes()
                if sig not in seen_schedules:
                    seen_schedules.add(sig)
                    valid_schedules.append(cand)