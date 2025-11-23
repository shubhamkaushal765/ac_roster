"""
Main officer roster building and management.

Handles parsing, validation, and application of roster adjustments for main officers.
"""

import re
from typing import Dict, List, Set, Tuple
import numpy as np

from acroster.officer import MainOfficer
from acroster.time_utils import hhmm_to_slot
from acroster.counter import CounterMatrix
from acroster.config import NUM_SLOTS, MODE_CONFIG, OperationMode


class RosterBuilder:
    """Builds and manages main officer rosters."""

    def __init__(self, mode: OperationMode):
        """
        Initialize roster builder.

        Args:
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        self.mode = mode
        self.config = MODE_CONFIG[mode]
        self.roster_templates = self.config['roster_templates']

    def parse_reported_officers(self, officers_str: str) -> Set[int]:
        """
        Parse string of reported officers into a set of IDs.

        Args:
            officers_str: String like "1-18" or "1,3,5-10"

        Returns:
            Set of officer IDs that reported
        """
        reported_officers = set()
        parts = officers_str.split(",")

        for part in parts:
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                for i in range(int(start), int(end) + 1):
                    reported_officers.add(i)
            else:
                reported_officers.add(int(part))

        return reported_officers

    def validate_adjustments(
            self,
            input_str: str,
            reported_officers: Set[int]
    ) -> List[Tuple[int, str, str]]:
        """
        Validate and parse late arrival (RA) and early departure (RO) adjustments.

        Args:
            input_str: String like "3RO2100, 11RO1700, 15RA1030"
            reported_officers: Set of reported officer IDs

        Returns:
            List of validated (officer_id, adjustment_type, hhmm) tuples
        """
        valid_entries = []

        if not input_str.strip():
            return valid_entries

        entries = input_str.split(",")

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            # Validate format
            if not re.match(r"^\d+(RA|RO)\d{4}$", entry):
                print(f"⚠️ Skipping {entry}: invalid format")
                continue

            # Parse entry
            if "RA" in entry:
                idx = entry.index("RA")
                officer_id = int(entry[:idx])
                hhmm = entry[idx + 2:]
                adj_type = "RA"
            else:
                idx = entry.index("RO")
                officer_id = int(entry[:idx])
                hhmm = entry[idx + 2:]
                adj_type = "RO"

            # Validate officer is reported
            if officer_id not in reported_officers:
                print(f"⚠️ Skipping {entry}: officer {officer_id} not in reported list.")
                continue

            # Validate time range
            h = int(hhmm[:2])
            m = int(hhmm[2:])

            if not (10 <= h <= 22):
                print(f"⚠️ Skipping {entry}: hour {h} out of range (1000–2200)")
                continue

            if m not in (0, 15, 30, 45):
                print(f"⚠️ Skipping {entry}: minutes {m} must be 00, 15, 30, or 45")
                continue

            if h == 22 and m > 0:
                print(f"⚠️ Skipping {entry}: must not exceed 2200")
                continue

            valid_entries.append((officer_id, adj_type, hhmm))

        return valid_entries

    def apply_ground_level_counters(
            self,
            officers: Dict[str, MainOfficer],
            gl_counters_str: str
    ) -> Dict[str, MainOfficer]:
        """
        Apply ground level counter assignments to officers.

        Args:
            officers: Dictionary of MainOfficer objects
            gl_counters_str: String like "4AC1, 8AC11, 12AC21"

        Returns:
            Updated officers dictionary
        """
        report_list = [s.strip() for s in gl_counters_str.split(",") if s.strip()]

        for officer_counter in report_list:
            if "AC" in officer_counter:
                idx = officer_counter.index("AC")
                officer_id = int(officer_counter[:idx])
                counter_no = int(officer_counter[idx + 2:])

                officer_key = f"M{officer_id}"
                if officer_key in officers:
                    officers[officer_key].apply_ground_level_counter(counter_no)

        return officers

    def apply_takeover_counters(
            self,
            officers: Dict[str, MainOfficer],
            handwritten_counters: str
    ) -> Dict[str, MainOfficer]:
        """
        Apply takeover counter assignments to officers.

        Args:
            officers: Dictionary of MainOfficer objects
            handwritten_counters: String like "3AC12,5AC13"

        Returns:
            Updated officers dictionary
        """
        if not handwritten_counters.strip():
            return officers

        pairs = re.findall(r"(\d+)\s*[aA]\s*[cC]\s*(\d+)", handwritten_counters)

        for officer_str, counter_str in pairs:
            officer_key = f"M{officer_str}"
            new_counter = int(counter_str)

            if officer_key in officers:
                officers[officer_key].apply_takeover_counter(new_counter)

        return officers

    def build_main_officers(
            self,
            main_officers_reported: str,
            report_gl_counters: str,
            ro_ra_officers: str
    ) -> Tuple[Dict[str, MainOfficer], Set[int], List[Tuple]]:
        """
        Build MainOfficer objects with all adjustments applied.

        Args:
            main_officers_reported: String of reported officers (e.g., "1-18")
            report_gl_counters: String of ground level counters (e.g., "4AC1")
            ro_ra_officers: String of RA/RO adjustments (e.g., "3RO2100")

        Returns:
            Tuple of:
                - Dict of MainOfficer objects keyed by officer_key
                - Set of reported officer IDs
                - List of valid adjustments
        """
        # Parse reported officers
        reported_officers = self.parse_reported_officers(main_officers_reported)

        # Validate adjustments
        valid_adjustments = self.validate_adjustments(ro_ra_officers, reported_officers)

        # Build adjustment dict
        adjustments = {}
        for officer_id, adj_type, hhmm in valid_adjustments:
            slot = hhmm_to_slot(hhmm)
            adjustments[officer_id] = (adj_type, slot)

        # Build MainOfficer objects
        main_officers: Dict[str, MainOfficer] = {}

        for officer_id in reported_officers:
            if officer_id not in self.roster_templates:
                continue

            # Create MainOfficer with template
            officer = MainOfficer(
                officer_id=officer_id,
                roster_template=self.roster_templates[officer_id]
            )

            # Apply late arrival or early departure
            if officer_id in adjustments:
                adjustment_type, slot = adjustments[officer_id]
                if adjustment_type == "RA":
                    officer.apply_late_arrival(slot)
                elif adjustment_type == "RO":
                    officer.apply_early_departure(slot)

            main_officers[officer.officer_key] = officer

        # Apply ground level counters
        main_officers = self.apply_ground_level_counters(
            main_officers,
            report_gl_counters
        )

        return main_officers, reported_officers, valid_adjustments


class LastCounterAssigner:
    """Handles assignment of last counters to eligible officers."""

    def __init__(self, mode: OperationMode):
        """
        Initialize last counter assigner.

        Args:
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        self.mode = mode
        self.config = MODE_CONFIG[mode]

    def get_last_counter_slots(
            self,
            reported_officers: Set[int],
            ro_ra_officers: List[Tuple]
    ) -> Dict[int, int]:
        """
        Compute each eligible officer's last counter end slot.

        Args:
            reported_officers: Set of reported officer IDs
            ro_ra_officers: List of (officer_id, adjustment_type, hhmm) tuples

        Returns:
            Dict mapping officer_id to last counter end slot
        """
        officer_last_counter = {}

        for officer_id in reported_officers:
            if officer_id % 4 == 3:  # Eligible officers (3, 7, 11, 15, ...)
                found = False

                # Check if officer has RO adjustment
                for ro_ra_officer in ro_ra_officers:
                    if ro_ra_officer[0] == officer_id and ro_ra_officer[1] == "RO":
                        last_counter_end_slot = hhmm_to_slot(ro_ra_officer[2])
                        found = True
                        break

                if not found:
                    last_counter_end_slot = NUM_SLOTS

                officer_last_counter[officer_id] = last_counter_end_slot

        return officer_last_counter

    def find_empty_counters_from_slot(
            self,
            counter_matrix: CounterMatrix,
            from_slot: int = 42
    ) -> List[int]:
        """
        Find counters empty from a specific slot onwards.

        Args:
            counter_matrix: CounterMatrix object
            from_slot: Starting slot to check (default: 42, which is 2030)

        Returns:
            List of counter IDs sorted by priority
        """
        num_counters = self.config['num_counters']
        counter_priority_list = self.config['counter_priority_list']
        empty_counters = []

        for counter_id in range(1, num_counters + 1):
            counter = counter_matrix.get_counter(counter_id)
            if counter.is_empty(from_slot, NUM_SLOTS - 1):
                empty_counters.append(counter_id)

        # Sort by priority
        empty_counters.sort(
            key=lambda x: counter_priority_list.index(x)
            if x in counter_priority_list else float("inf")
        )

        return empty_counters

    def assign_last_counters(
            self,
            main_officers: Dict[str, MainOfficer],
            officer_last_counter: Dict[int, int],
            empty_counters: List[int]
    ) -> Dict[str, MainOfficer]:
        """
        Apply last counter assignments to eligible officers.

        Args:
            main_officers: Dict of MainOfficer objects
            officer_last_counter: Dict mapping officer_id to end slot
            empty_counters: List of available counter IDs

        Returns:
            Updated officers dictionary
        """
        empty_counters_copy = empty_counters.copy()

        for officer_key, officer in main_officers.items():
            if officer.officer_id in officer_last_counter:
                last_slot = officer_last_counter[officer.officer_id]

                if last_slot >= 42 and empty_counters_copy:
                    counter_no = empty_counters_copy[0]
                    officer.apply_last_counter(last_slot, counter_no)
                    empty_counters_copy.pop(0)

        return main_officers