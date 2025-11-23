"""
Counter assignment engine.

Handles assignment of officers to counters using gap-aware greedy algorithms
and priority-based selection.
"""

from typing import Dict, List, Tuple
import numpy as np

from acroster.officer import Officer, OTOfficer
from acroster.counter import CounterMatrix
from acroster.config import NUM_SLOTS, MODE_CONFIG, OperationMode


class CounterAssignmentEngine:
    """Assigns officers to counters using priority-based algorithms."""

    def __init__(self, mode: OperationMode):
        """
        Initialize counter assignment engine.

        Args:
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        self.mode = mode
        self.config = MODE_CONFIG[mode]
        self.num_counters = self.config['num_counters']
        self.counter_priority_list = self.config['counter_priority_list']

    def officers_to_counter_matrix(
            self,
            officers: Dict[str, Officer]
    ) -> CounterMatrix:
        """
        Convert officer schedules to CounterMatrix object.

        Args:
            officers: Dict of Officer objects keyed by officer_key

        Returns:
            CounterMatrix object with all officer assignments
        """
        counter_matrix = CounterMatrix(num_slots=NUM_SLOTS, mode=self.mode)

        for officer_key, officer in officers.items():
            for slot, counter in enumerate(officer.schedule):
                if counter != 0:
                    counter_matrix.assign_officer_to_counter(
                        counter_id=int(counter),
                        officer_key=officer_key,
                        start_slot=slot,
                        end_slot=slot
                    )

        return counter_matrix

    def assign_ot_officers(
            self,
            counter_matrix: CounterMatrix,
            ot_counters: str
    ) -> Tuple[CounterMatrix, List[OTOfficer]]:
        """
        Add OT (overtime) officers to counter matrix.

        Args:
            counter_matrix: Base counter matrix
            ot_counters: Comma-separated string of counter IDs (e.g., "2,20,40")

        Returns:
            Tuple of:
                - Updated CounterMatrix with OT officers
                - List of OTOfficer objects created
        """
        if len(ot_counters) == 0:
            return counter_matrix.copy(), []

        counter_matrix_w_ot = counter_matrix.copy()
        ot_list = [int(x.strip()) for x in ot_counters.split(",") if x.strip()]

        ot_officers = []
        for i, ot_counter in enumerate(ot_list):
            ot_officer = OTOfficer(officer_id=i + 1, counter_no=ot_counter)
            ot_officers.append(ot_officer)
            counter_matrix_w_ot.assign_officer_to_counter(
                ot_counter, ot_officer.officer_key, 0, 1
            )

        return counter_matrix_w_ot, ot_officers


class SOSAssignmentEngine:
    """
    Assigns SOS officers to counters using gap-aware greedy interval packing.

    Prioritizes:
    1. Pre-assigned counters
    2. Partial counters with connections
    3. Already used SOS counters with connections
    4. Priority list order
    """

    def __init__(self, mode: OperationMode):
        """
        Initialize SOS assignment engine.

        Args:
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        self.mode = mode
        self.config = MODE_CONFIG[mode]
        self.num_counters = self.config['num_counters']
        self.counter_priority_list = self.config['counter_priority_list']

    def assign_sos_officers(
            self,
            pre_assigned_counter_dict: Dict[int, int],
            schedule_intervals_to_officers: Dict[Tuple[int, int], List[int]],
            main_counter_matrix: CounterMatrix
    ) -> CounterMatrix:
        """
        Assign SOS officers to counters using gap-aware greedy interval packing.

        Args:
            pre_assigned_counter_dict: Dict of {officer_index: counter_no}
            schedule_intervals_to_officers: Dict of {(start, end): [officer_ids]}
            main_counter_matrix: CounterMatrix with main officer assignments

        Returns:
            CounterMatrix with SOS officer assignments
        """
        # Create copies for manipulation
        sos_main_counter_matrix = main_counter_matrix.copy()
        sos_counter_matrix = CounterMatrix(num_slots=NUM_SLOTS, mode=self.mode)

        # Sort intervals by start time, then by end time
        sorted_intervals = sorted(
            schedule_intervals_to_officers.items(),
            key=lambda x: (x[0][0], x[0][1])
        )

        print("++++++++++++++++++++sorted intervals++++++")
        print(sorted_intervals)

        for interval, officer_ids in sorted_intervals:
            start, end = interval

            for officer_id in officer_ids:
                best_counter = self._find_best_counter(
                    officer_id,
                    start,
                    end,
                    pre_assigned_counter_dict,
                    sos_main_counter_matrix,
                    sos_counter_matrix
                )

                if best_counter is None:
                    print(
                        f"ERROR: No available counter for officer {officer_id}, "
                        f"interval {interval}"
                    )
                    continue

                # Assign officer to counter
                officer_key = f"S{officer_id + 1}"
                sos_counter_matrix.assign_officer_to_counter(
                    best_counter, officer_key, start, end
                )
                sos_main_counter_matrix.assign_officer_to_counter(
                    best_counter, officer_key, start, end
                )

        # Print statistics
        self._print_statistics(sos_main_counter_matrix, sos_counter_matrix)

        return sos_counter_matrix

    def _find_best_counter(
            self,
            officer_id: int,
            start: int,
            end: int,
            pre_assigned_counter_dict: Dict[int, int],
            sos_main_counter_matrix: CounterMatrix,
            sos_counter_matrix: CounterMatrix
    ) -> int:
        """
        Find the best counter for an officer's interval.

        Priority order:
        1. Pre-assigned counter (if applicable)
        2. Partial counters with connections
        3. Already used SOS counters with connections
        4. Priority list order

        Args:
            officer_id: SOS officer index
            start: Start slot
            end: End slot
            pre_assigned_counter_dict: Pre-assigned counters
            sos_main_counter_matrix: Combined matrix
            sos_counter_matrix: SOS-only matrix

        Returns:
            Counter ID to assign, or None if none available
        """
        best_counter = None
        best_score = -1

        # Step 0: Check if counter is pre-assigned
        if (
                len(pre_assigned_counter_dict) > 0
                and officer_id in pre_assigned_counter_dict
                and start == 0
        ):
            return pre_assigned_counter_dict[officer_id]

        # Get current partial counters
        partial_counters = sos_main_counter_matrix.get_partial_empty_counters()

        # Step 1: Try partial counters first (must be CONNECTED)
        for counter_id in partial_counters:
            if sos_main_counter_matrix.is_interval_empty(counter_id, start, end):
                if sos_main_counter_matrix.is_interval_connected(counter_id, start, end):
                    score = 100  # Connected to existing assignment
                    if score > best_score:
                        best_score = score
                        best_counter = counter_id

        # Step 2: Try already used SOS counters (must be CONNECTED)
        if best_counter is None:
            used_sos_counters = sos_counter_matrix.get_used_counters(
                exclude_partial=True
            )
            for counter_id in used_sos_counters:
                if counter_id in partial_counters:
                    continue
                if sos_main_counter_matrix.is_interval_empty(counter_id, start, end):
                    if sos_main_counter_matrix.is_interval_connected(
                            counter_id, start, end
                    ):
                        score = 100
                        if score > best_score:
                            best_score = score
                            best_counter = counter_id

        # Step 3: Iterate through counter_priority_list in order
        if best_counter is None:
            for priority_counter in self.counter_priority_list:
                if sos_main_counter_matrix.is_interval_empty(
                        priority_counter, start, end
                ):
                    best_counter = priority_counter
                    break

        return best_counter

    def _print_statistics(
            self,
            sos_main_counter_matrix: CounterMatrix,
            sos_counter_matrix: CounterMatrix
    ):
        """Print assignment statistics."""
        final_partial = sos_main_counter_matrix.get_partial_empty_counters()
        used_counters = len(sos_counter_matrix.get_used_counters())

        print("\nAssignment Statistics:")
        print(f"  Partial counters at end: {len(final_partial)}")
        print(f"  Total counters used for SOS: {used_counters}/{self.num_counters}")


class MatrixConverter:
    """Converts between different matrix representations."""

    @staticmethod
    def counter_to_officer_schedule(counter_matrix: np.ndarray) -> Dict[str, List[int]]:
        """
        Convert counter_matrix back to officer schedules.

        Args:
            counter_matrix: 2D numpy array (num_counters x num_slots)

        Returns:
            Dict where keys are officer keys and values are counter assignments per slot
        """
        num_counters, num_slots = counter_matrix.shape
        officer_schedule = {}

        for counter_idx in range(num_counters):
            for slot in range(num_slots):
                officer_id = counter_matrix[counter_idx, slot]
                if officer_id == "0":
                    continue
                if officer_id not in officer_schedule:
                    officer_schedule[officer_id] = [0] * num_slots
                officer_schedule[officer_id][slot] = counter_idx + 1

        # Sort by officer type and number
        officer_schedule = MatrixConverter._sort_officer_schedule(officer_schedule)

        return officer_schedule

    @staticmethod
    def _sort_officer_schedule(officer_schedule: Dict) -> Dict:
        """Sort officer schedule by type (M, S, OT) and number."""
        prefix_order = {"M": 0, "S": 1, "OT": 2}

        def sort_key(k):
            k = str(k)
            if k.startswith("OT"):
                prefix = "OT"
                num_part = k[2:]
            else:
                prefix = k[0]
                num_part = k[1:]
            try:
                num_val = int(num_part)
            except ValueError:
                num_val = float("inf")
            return (prefix_order.get(prefix, 99), num_val)

        sorted_keys = sorted(officer_schedule.keys(), key=sort_key)
        return {k: officer_schedule[k] for k in sorted_keys}

    @staticmethod
    def merge_prefixed_matrices(
            counter_matrix: np.ndarray,
            sos_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Merge two prefixed matrices, keeping non-zero entries from sos_matrix.

        Args:
            counter_matrix: Original counter matrix (2D numpy array of strings)
            sos_matrix: SOS counter matrix (2D numpy array of strings)

        Returns:
            Merged matrix of the same shape
        """
        if counter_matrix.shape != sos_matrix.shape:
            raise ValueError("Both matrices must have the same shape")

        merged_matrix = np.where(sos_matrix != "0", sos_matrix, counter_matrix)
        return merged_matrix