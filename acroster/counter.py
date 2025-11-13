"""
Counter and CounterMatrix classes for managing counter assignments and officer scheduling.
"""
from typing import List
import numpy as np
from acroster.config import OperationMode, MODE_CONFIG

class Counter:
    """
    Represents a single counter with its time slot assignments.

    A counter can be assigned officers across 48 time slots (15-minute intervals).
    Each slot can contain an officer identifier (e.g., 'M1', 'S2', 'OT3') or '0' for empty.
    """

    def __init__(self, mode: OperationMode, counter_id: int, num_slots: int = 48):
        """
        Initialize a counter.

        Args:
            counter_id: Unique identifier for the counter
            num_slots: Number of time slots (default: 48)
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        max_counter = MODE_CONFIG[mode]['num_counters']
        
        if not 1 <= counter_id <= max_counter:
            raise ValueError(
                f"Counter ID must be between 1 and {max_counter} for {mode.value} mode, got {counter_id}"
            )

        self.counter_id = counter_id
        self.num_slots = num_slots
        self.mode = mode
        # Initialize all slots as empty ('0')
        self.slots = np.full(num_slots, "0", dtype=object)

    def assign_officer(self, officer_key: str, start_slot: int, end_slot: int):
        """
        Assign an officer to a range of time slots.

        Args:
            officer_key: Officer identifier (e.g., 'M1', 'S5', 'OT2')
            start_slot: Starting slot index (inclusive)
            end_slot: Ending slot index (inclusive)

        Raises:
            ValueError: If slot indices are out of range
        """
        if not 0 <= start_slot < self.num_slots:
            raise ValueError(
                f"Start slot {start_slot} out of range [0, {self.num_slots})"
            )
        if not 0 <= end_slot < self.num_slots:
            raise ValueError(
                f"End slot {end_slot} out of range [0, {self.num_slots})"
            )
        if start_slot > end_slot:
            raise ValueError(
                f"Start slot {start_slot} cannot be after end slot {end_slot}"
            )

        self.slots[start_slot:end_slot + 1] = officer_key

    def clear_slots(self, start_slot: int, end_slot: int):
        """
        Clear officer assignments in a range of time slots.

        Args:
            start_slot: Starting slot index (inclusive)
            end_slot: Ending slot index (inclusive)
        """
        self.slots[start_slot:end_slot + 1] = "0"

    def is_empty(self, start_slot: int, end_slot: int) -> bool:
        """
        Check if all slots in a range are empty.

        Args:
            start_slot: Starting slot index (inclusive)
            end_slot: Ending slot index (inclusive)

        Returns:
            True if all slots in range are '0', False otherwise
        """
        return np.all(self.slots[start_slot:end_slot + 1] == "0")

    def is_completely_empty(self) -> bool:
        """
        Check if the entire counter is empty.

        Returns:
            True if all slots are '0', False otherwise
        """
        return np.all(self.slots == "0")

    def is_completely_full(self) -> bool:
        """
        Check if the entire counter is full (no empty slots).

        Returns:
            True if no slots are '0', False otherwise
        """
        return np.all(self.slots != "0")

    def is_partially_empty(self) -> bool:
        """
        Check if the counter is partially empty (has both filled and empty slots).

        Returns:
            True if counter has at least one filled and one empty slot
        """
        has_filled = np.any(self.slots != "0")
        has_empty = np.any(self.slots == "0")
        return has_filled and has_empty

    def get_officer_at_slot(self, slot: int) -> str:
        """
        Get the officer assigned to a specific slot.

        Args:
            slot: Slot index

        Returns:
            Officer key or '0' if empty
        """
        return self.slots[slot]

    def is_connected(self, start_slot: int, end_slot: int) -> bool:
        """
        Check if an interval connects to existing assignments (adjacent non-zero slots).

        Args:
            start_slot: Starting slot index (inclusive)
            end_slot: Ending slot index (inclusive)

        Returns:
            True if the interval connects to previous or next non-zero slot
        """
        # Check connection to previous slot
        if start_slot > 0 and self.slots[start_slot - 1] != "0":
            return True

        # Check connection to next slot
        if end_slot < self.num_slots - 1 and self.slots[end_slot + 1] != "0":
            return True

        return False

    def get_filled_slots_count(self) -> int:
        """
        Get the number of filled (non-empty) slots.

        Returns:
            Count of slots with officer assignments
        """
        return np.sum(self.slots != "0")

    def get_empty_slots_count(self) -> int:
        """
        Get the number of empty slots.

        Returns:
            Count of empty slots
        """
        return np.sum(self.slots == "0")

    def get_slots_array(self) -> np.ndarray:
        """
        Get a copy of the slots array.

        Returns:
            Copy of the slots array
        """
        return self.slots.copy()

    def __repr__(self):
        status = "empty" if self.is_completely_empty() else \
            "full" if self.is_completely_full() else \
                "partial"
        return f"Counter({self.counter_id}, status={status}, filled={self.get_filled_slots_count()}/{self.num_slots})"

    def __str__(self):
        return f"Counter {self.counter_id}: {self.get_filled_slots_count()}/{self.num_slots} slots filled"


class CounterMatrix:
    """
    Manages all counters as a collective matrix structure.

    Provides operations on the entire set of counters including querying,
    assignment, merging, and conversion to/from numpy matrices.
    """

    def __init__(self, mode: OperationMode, num_slots: int = 48):
        """
        Initialize the counter matrix.

        Args:
            num_slots: Number of time slots (default: 48)
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        self.mode = mode
        self.num_counters = MODE_CONFIG[mode]['num_counters']
        self.num_slots = num_slots

        # Create dictionary of Counter objects (1-indexed)
        self.counters = {
            i: Counter(i, num_slots, mode) 
            for i in range(1, self.num_counters + 1)
        }

    def get_counter(self, counter_id: int) -> Counter:
        """
        Get a specific counter by ID.

        Args:
            counter_id: Counter identifier (1-41)

        Returns:
            Counter object

        Raises:
            ValueError: If counter_id is invalid
        """
        if counter_id not in self.counters:
            raise ValueError(
                f"Counter {counter_id} not found. Must be between 1 and {self.num_counters}"
            )
        return self.counters[counter_id]

    def assign_officer_to_counter(
            self,
            counter_id: int,
            officer_key: str,
            start_slot: int,
            end_slot: int
    ):
        """
        Assign an officer to a specific counter for a range of slots.

        Args:
            counter_id: Counter identifier (1-41)
            officer_key: Officer identifier (e.g., 'M1', 'S5')
            start_slot: Starting slot index (inclusive)
            end_slot: Ending slot index (inclusive)
        """
        counter = self.get_counter(counter_id)
        counter.assign_officer(officer_key, start_slot, end_slot)

    def get_empty_counters(self) -> List[int]:
        """
        Get list of completely empty counter IDs.

        Returns:
            List of counter IDs that are completely empty
        """
        return [
            counter_id for counter_id, counter in self.counters.items()
            if counter.is_completely_empty()
        ]

    def get_partial_empty_counters(self) -> List[int]:
        """
        Get list of partially empty counter IDs.

        A partial counter has at least one filled slot and at least one empty slot.

        Returns:
            List of counter IDs that are partially empty
        """
        return [
            counter_id for counter_id, counter in self.counters.items()
            if counter.is_partially_empty()
        ]

    def get_full_counters(self) -> List[int]:
        """
        Get list of completely full counter IDs.

        Returns:
            List of counter IDs that are completely full
        """
        return [
            counter_id for counter_id, counter in self.counters.items()
            if counter.is_completely_full()
        ]

    def get_counters_with_prefix(self, prefix: str) -> List[int]:
        """
        Get counters that have officers with a specific prefix (e.g., 'S' for SOS officers).

        Args:
            prefix: Officer prefix to search for (e.g., 'M', 'S', 'OT')

        Returns:
            List of counter IDs containing officers with the specified prefix
        """
        result = []
        for counter_id, counter in self.counters.items():
            if np.any(np.char.startswith(counter.slots.astype(str), prefix)):
                result.append(counter_id)
        return result

    def get_used_counters(self, exclude_partial: bool = False) -> List[int]:
        """
        Get list of counters that have any assignments.

        Args:
            exclude_partial: If True, exclude partially empty counters

        Returns:
            List of counter IDs with at least one assignment
        """
        used = []
        partial_set = set(
            self.get_partial_empty_counters()
        ) if exclude_partial else set()

        for counter_id, counter in self.counters.items():
            if exclude_partial and counter_id in partial_set:
                continue
            if not counter.is_completely_empty():
                used.append(counter_id)

        return used

    def is_interval_empty(
            self, counter_id: int, start_slot: int, end_slot: int
    ) -> bool:
        """
        Check if an interval is empty in a specific counter.

        Args:
            counter_id: Counter identifier
            start_slot: Starting slot index (inclusive)
            end_slot: Ending slot index (inclusive)

        Returns:
            True if all slots in the interval are empty
        """
        counter = self.get_counter(counter_id)
        return counter.is_empty(start_slot, end_slot)

    def is_interval_connected(
            self, counter_id: int, start_slot: int, end_slot: int
    ) -> bool:
        """
        Check if an interval connects to existing assignments in a counter.

        Args:
            counter_id: Counter identifier
            start_slot: Starting slot index (inclusive)
            end_slot: Ending slot index (inclusive)

        Returns:
            True if the interval connects to adjacent non-zero slots
        """
        counter = self.get_counter(counter_id)
        return counter.is_connected(start_slot, end_slot)

    def to_matrix(self) -> np.ndarray:
        """
        Convert the counter matrix to a 2D numpy array.

        Returns:
            2D numpy array of shape (num_counters, num_slots)
            Row i corresponds to counter i+1 (0-indexed array, 1-indexed counters)
        """
        matrix = np.full(
            (self.num_counters, self.num_slots), "0", dtype=object
        )
        for counter_id, counter in self.counters.items():
            matrix[counter_id - 1] = counter.get_slots_array()
        return matrix

    def from_matrix(self, matrix: np.ndarray):
        """
        Load counter assignments from a 2D numpy array.

        Args:
            matrix: 2D numpy array of shape (num_counters, num_slots)

        Raises:
            ValueError: If matrix shape doesn't match
        """
        if matrix.shape != (self.num_counters, self.num_slots):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match expected "
                f"({self.num_counters}, {self.num_slots})"
            )

        for counter_id in range(1, self.num_counters + 1):
            self.counters[counter_id].slots = matrix[counter_id - 1].copy()

    def merge_with(
            self, other: 'CounterMatrix', priority: str = 'other'
    ) -> 'CounterMatrix':
        """
        Merge this counter matrix with another, creating a new CounterMatrix.

        Args:
            other: Another CounterMatrix to merge with
            priority: Which matrix takes priority for conflicts
                     'other' = other matrix overwrites (for SOS overlay)
                     'self' = this matrix takes priority

        Returns:
            New CounterMatrix with merged assignments

        Raises:
            ValueError: If matrices have different dimensions
        """
        if self.num_counters != other.num_counters or self.num_slots != other.num_slots:
            raise ValueError("Cannot merge matrices with different dimensions")

        # Create new CounterMatrix
        merged = CounterMatrix(num_slots=self.num_slots, mode=self.mode)

        # Get matrix representations
        self_matrix = self.to_matrix()
        other_matrix = other.to_matrix()

        # Merge based on priority
        if priority == 'other':
            # Other matrix overwrites where it's non-zero
            merged_matrix = np.where(
                other_matrix != "0", other_matrix, self_matrix
            )
        elif priority == 'self':
            # This matrix overwrites where it's non-zero
            merged_matrix = np.where(
                self_matrix != "0", self_matrix, other_matrix
            )
        else:
            raise ValueError(
                f"Invalid priority '{priority}'. Use 'other' or 'self'"
            )

        # Load merged matrix
        merged.from_matrix(merged_matrix)

        return merged

    def copy(self) -> 'CounterMatrix':
        """
        Create a deep copy of this CounterMatrix.

        Returns:
            New CounterMatrix with copied data
        """
        new_matrix = CounterMatrix(num_slots=self.num_slots, mode=self.mode)
        new_matrix.from_matrix(self.to_matrix())
        return new_matrix

    def get_statistics_at_slot(self, slot: int) -> dict:
        """
        Get manning statistics for a specific time slot.

        Args:
            slot: Slot index

        Returns:
            Dictionary with statistics:
            - 'total_filled': Total number of counters with officers
            - 'by_row_group': Counts for counter groups [1-10, 11-20, 21-30, 31-40]
            - 'counter_41': Whether counter 41 is filled
        """
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"Slot {slot} out of range")

        matrix = self.to_matrix()

        # Get zone configuration for this mode
        cfg = MODE_CONFIG[self.mode]
        zones = [cfg['zone1'], cfg['zone2'], cfg['zone3'], cfg['zone4']]
        
        # Count filled slots in first car counters, e.g all except last counter
        car_ctr = np.sum(matrix[0::self.num_counters-1, slot] != "0")

        # Count filled in motor counter, which is the last counter
        motor_ctr = 1 if matrix[self.num_counters-1, slot] != "0" else 0

        # Count by row groups (1-10, 11-20, 21-30, 31-40)
        group_counts = [
            np.sum(matrix[zone[0]-1:zone[1]-1, slot] != "0") 
            for zone in zones
        ]
        return {
            'total_filled':  car_ctr + motor_ctr,
            'car_ctr': car_ctr,
            'motor_ctr':    motor_ctr,
            'by_row_group':  group_counts
        }

    def clear_all(self):
        """Clear all counter assignments."""
        for counter in self.counters.values():
            counter.slots[:] = "0"

    def __repr__(self):
        empty = len(self.get_empty_counters())
        partial = len(self.get_partial_empty_counters())
        full = len(self.get_full_counters())
        return (
            f"CounterMatrix({self.num_counters} counters, {self.num_slots} slots, "
            f"empty={empty}, partial={partial}, full={full})"
        )

    def __str__(self):
        used = len(self.get_used_counters())
        return f"CounterMatrix: {used}/{self.num_counters} counters in use"


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Counter Class ===")

    # Create a counter
    counter = Counter(1)
    print(f"Created: {counter}")

    # Assign officers
    counter.assign_officer("M1", 0, 5)
    counter.assign_officer("M2", 10, 15)
    print(f"After assignments: {counter}")
    print(f"Is partially empty: {counter.is_partially_empty()}")
    print(f"Is slot 8-9 empty: {counter.is_empty(8, 9)}")
    print(f"Is interval 8-9 connected: {counter.is_connected(8, 9)}")

    print("\n=== Testing CounterMatrix Class ===")

    # Create matrix
    matrix = CounterMatrix(num_counters=41, num_slots=48)
    print(f"Created: {matrix}")

    # Assign some officers
    matrix.assign_officer_to_counter(1, "M1", 0, 10)
    matrix.assign_officer_to_counter(1, "M2", 20, 30)
    matrix.assign_officer_to_counter(5, "S1", 0, 47)
    matrix.assign_officer_to_counter(10, "OT1", 0, 1)

    print(f"\nAfter assignments: {matrix}")
    print(f"Empty counters: {len(matrix.get_empty_counters())}")
    print(f"Partial counters: {matrix.get_partial_empty_counters()}")
    print(f"Full counters: {matrix.get_full_counters()}")
    print(f"Counters with 'S' prefix: {matrix.get_counters_with_prefix('S')}")

    # Test statistics
    stats = matrix.get_statistics_at_slot(5)
    print(f"\nStatistics at slot 5: {stats}")

    # Test merge
    print("\n=== Testing Merge ===")
    matrix2 = CounterMatrix(num_counters=41, num_slots=48)
    matrix2.assign_officer_to_counter(1, "S3", 11, 19)  # Fill gap in counter 1
    matrix2.assign_officer_to_counter(20, "S4", 0, 20)

    merged = matrix.merge_with(matrix2, priority='other')
    print(f"Merged matrix: {merged}")
    print(f"Counter 1 in merged: {merged.get_counter(1)}")

    # Convert to numpy array
    print("\n=== Testing Matrix Conversion ===")
    np_matrix = matrix.to_matrix()
    print(f"Numpy matrix shape: {np_matrix.shape}")
    print(f"Counter 1 slots 0-15: {np_matrix[0, 0:16]}")
