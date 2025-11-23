"""
ScheduleManager class - High-level interface for the officer scheduling system.

This class provides a simple, user-friendly interface to the refactored OOP
scheduling algorithm. It wraps the RosterAlgorithmOrchestrator and provides
additional convenience methods for accessing results.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

# Import new OOP architecture components
from acroster.algorithm_orchestrator import RosterAlgorithmOrchestrator
from acroster.officer import MainOfficer, OTOfficer, SOSOfficer
from acroster.counter import CounterMatrix
from acroster.config import OperationMode, MODE_CONFIG


class ScheduleManager:
    """
    High-level manager class for the officer scheduling system.

    Provides a simple interface to the scheduling algorithm with convenient
    methods for accessing officers, schedules, and statistics.

    This class acts as a facade over the RosterAlgorithmOrchestrator,
    providing state management and easy access to results.
    """

    def __init__(
        self,
        mode: OperationMode = OperationMode.ARRIVAL,
        num_slots: int = 48,
        start_hour: int = 10
    ):
        """
        Initialize the ScheduleManager.

        Args:
            mode: Operation mode (ARRIVAL or DEPARTURE)
            num_slots: Number of time slots per day (default: 48 for 15-min intervals)
            start_hour: Starting hour for the schedule (default: 10 for 10:00 AM)
        """
        self.mode = mode
        self.num_slots = num_slots
        self.start_hour = start_hour

        # Get configuration for this mode
        cfg = MODE_CONFIG[mode]
        self.num_counters = cfg["num_counters"]
        self.counter_priority_list = cfg["counter_priority_list"]

        # Initialize the orchestrator
        self.orchestrator = RosterAlgorithmOrchestrator(mode=mode)

        # Officer collections (populated after running algorithm)
        self.main_officers: Dict[str, MainOfficer] = {}
        self.sos_officers: List[SOSOfficer] = []
        self.ot_officers: List[OTOfficer] = []

        # Results (populated after running algorithm)
        self.main_counter_matrix_np: Optional[np.ndarray] = None
        self.final_counter_matrix_np: Optional[np.ndarray] = None
        self.officer_schedules: Optional[Dict[str, List[int]]] = None
        self.statistics: Optional[List[str]] = None
        self.optimization_penalty: Optional[float] = None

        # Processing state
        self._has_run = False

    def run_algorithm(
        self,
        main_officers_reported: str,
        report_gl_counters: str,
        sos_timings: str,
        ro_ra_officers: str,
        handwritten_counters: str,
        ot_counters: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]], List[str]]:
        """
        Execute the complete scheduling algorithm.

        Args:
            main_officers_reported: Comma-separated officer IDs or ranges (e.g., "1-18,20,22")
            report_gl_counters: Ground level counter assignments (e.g., "4AC1,8AC11")
            sos_timings: SOS officer timing specifications
            ro_ra_officers: Report Out/Report After adjustments (e.g., "3RO2100,11RO1700")
            handwritten_counters: Takeover counter assignments (e.g., "3AC12,5AC13")
            ot_counters: Overtime counter assignments (e.g., "2,20,40")

        Returns:
            Tuple containing:
                - main_counter_matrix_np: Numpy array of main + OT officers
                - final_counter_matrix_np: Numpy array with all officers including SOS
                - officer_schedules: Dict mapping officer keys to their schedules
                - statistics: List of statistics strings [main_stats, final_stats]
        """
        print("=" * 70)
        print(f"RUNNING SCHEDULING ALGORITHM - {self.mode.value.upper()} MODE")
        print("=" * 70)

        # Run the orchestrator
        results = self.orchestrator.run(
            main_officers_reported=main_officers_reported,
            report_gl_counters=report_gl_counters,
            sos_timings=sos_timings,
            ro_ra_officers=ro_ra_officers,
            handwritten_counters=handwritten_counters,
            ot_counters=ot_counters
        )

        # Unpack results
        (
            self.main_counter_matrix_np,
            self.final_counter_matrix_np,
            self.officer_schedules,
            self.statistics
        ) = results

        # Extract officer objects from orchestrator's internal state
        self._extract_officers_from_orchestrator()

        # Mark as run
        self._has_run = True

        print("\n" + "=" * 70)
        print("ALGORITHM COMPLETED SUCCESSFULLY")
        print("=" * 70)

        return results

    def _extract_officers_from_orchestrator(self):
        """
        Extract officer objects from the orchestrator's internal components.

        This is a helper method to populate the manager's officer collections
        after the algorithm has run.
        """
        # Note: Since the orchestrator doesn't expose officers directly,
        # we reconstruct them from the schedules
        # In a future enhancement, the orchestrator could be modified to expose these

        # For now, we'll populate based on the officer_schedules keys
        if self.officer_schedules:
            # Count officers by type
            main_count = sum(1 for k in self.officer_schedules.keys() if k.startswith('M'))
            sos_count = sum(1 for k in self.officer_schedules.keys() if k.startswith('S'))
            ot_count = sum(1 for k in self.officer_schedules.keys() if k.startswith('OT'))

            print(f"\nOfficers identified:")
            print(f"  - Main Officers: {main_count}")
            print(f"  - SOS Officers: {sos_count}")
            print(f"  - OT Officers: {ot_count}")

    def add_sos_officers_only(
        self,
        sos_timings: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]], List[str]]:
        """
        Add SOS officers to an existing schedule.

        This method allows adding SOS officers after the main algorithm has run,
        without re-running the entire process.

        Args:
            sos_timings: SOS officer timing specifications

        Returns:
            Tuple containing updated results

        Raises:
            RuntimeError: If main algorithm hasn't been run yet
        """
        if not self._has_run:
            raise RuntimeError(
                "Must run main algorithm first before adding SOS officers. "
                "Call run_algorithm() first."
            )

        if not sos_timings.strip():
            print("No SOS timings provided, returning existing results.")
            return (
                self.main_counter_matrix_np,
                self.final_counter_matrix_np,
                self.officer_schedules,
                self.statistics
            )

        print("\n" + "=" * 70)
        print("ADDING SOS OFFICERS TO EXISTING SCHEDULE")
        print("=" * 70)

        # This would require access to internal orchestrator methods
        # For now, we recommend re-running the full algorithm
        print("Note: For adding SOS officers, please re-run the full algorithm")
        print("with the SOS timings included in the original run_algorithm() call.")

        return (
            self.main_counter_matrix_np,
            self.final_counter_matrix_np,
            self.officer_schedules,
            self.statistics
        )

    def get_main_officers(self) -> Dict[str, MainOfficer]:
        """
        Get all main officer objects.

        Returns:
            Dictionary mapping officer keys (e.g., 'M1') to MainOfficer objects

        Note:
            In the refactored architecture, officer objects are internal to
            the orchestrator. This method returns an empty dict for now.
            Access schedules via get_officer_schedule() instead.
        """
        return self.main_officers

    def get_sos_officers(self) -> List[SOSOfficer]:
        """
        Get all SOS officer objects.

        Returns:
            List of SOSOfficer objects

        Note:
            In the refactored architecture, officer objects are internal to
            the orchestrator. This method returns an empty list for now.
            Access schedules via get_officer_schedule() instead.
        """
        return self.sos_officers

    def get_ot_officers(self) -> List[OTOfficer]:
        """
        Get all OT officer objects.

        Returns:
            List of OTOfficer objects

        Note:
            In the refactored architecture, officer objects are internal to
            the orchestrator. This method returns an empty list for now.
            Access schedules via get_officer_schedule() instead.
        """
        return self.ot_officers

    def get_officer_schedule(self, officer_key: str) -> Optional[List[int]]:
        """
        Get the schedule for a specific officer.

        Args:
            officer_key: Officer identifier (e.g., 'M1', 'S5', 'OT2')

        Returns:
            List of counter assignments per slot, or None if officer not found
        """
        if self.officer_schedules is None:
            return None
        return self.officer_schedules.get(officer_key)

    def get_all_officer_schedules(self) -> Optional[Dict[str, List[int]]]:
        """
        Get schedules for all officers.

        Returns:
            Dictionary mapping officer keys to their schedules, or None if not yet generated
        """
        return self.officer_schedules

    def get_statistics(self) -> Optional[List[str]]:
        """
        Get the generated statistics.

        Returns:
            List containing [main_stats, final_stats] or None if not yet generated
        """
        return self.statistics

    def get_optimization_penalty(self) -> Optional[float]:
        """
        Get the optimization penalty from SOS scheduling.

        Returns:
            Penalty value or None if SOS scheduling was not performed
        """
        return self.optimization_penalty

    def get_main_counter_matrix(self) -> Optional[np.ndarray]:
        """
        Get the main counter matrix (main + OT officers only).

        Returns:
            Numpy array of counter assignments or None if not yet generated
        """
        return self.main_counter_matrix_np

    def get_final_counter_matrix(self) -> Optional[np.ndarray]:
        """
        Get the final counter matrix (includes SOS officers if present).

        Returns:
            Numpy array of counter assignments or None if not yet generated
        """
        if self.final_counter_matrix_np is not None:
            return self.final_counter_matrix_np
        return self.main_counter_matrix_np

    def get_all_officers_count(self) -> Dict[str, int]:
        """
        Get count of each officer type from schedules.

        Returns:
            Dictionary with counts: {'main': X, 'sos': Y, 'ot': Z, 'total': T}
        """
        if self.officer_schedules is None:
            return {'main': 0, 'sos': 0, 'ot': 0, 'total': 0}

        main_count = sum(1 for k in self.officer_schedules.keys() if k.startswith('M'))
        sos_count = sum(1 for k in self.officer_schedules.keys() if k.startswith('S'))
        ot_count = sum(1 for k in self.officer_schedules.keys() if k.startswith('OT'))

        return {
            'main': main_count,
            'sos': sos_count,
            'ot': ot_count,
            'total': main_count + sos_count + ot_count
        }

    def get_officer_keys_by_type(self, officer_type: str) -> List[str]:
        """
        Get all officer keys of a specific type.

        Args:
            officer_type: Type of officers ('M', 'S', or 'OT')

        Returns:
            List of officer keys matching the type
        """
        if self.officer_schedules is None:
            return []

        prefix = officer_type.upper()
        return [k for k in self.officer_schedules.keys() if k.startswith(prefix)]

    def export_schedules_to_dict(self) -> Dict:
        """
        Export all scheduling data to a dictionary format.

        Returns:
            Dictionary containing all scheduling information
        """
        return {
            'mode': self.mode.value,
            'officer_schedules': self.officer_schedules,
            'statistics': self.statistics,
            'optimization_penalty': self.optimization_penalty,
            'officer_counts': self.get_all_officers_count(),
            'config': {
                'num_slots': self.num_slots,
                'num_counters': self.num_counters,
                'start_hour': self.start_hour
            }
        }

    def print_summary(self):
        """Print a detailed summary of the scheduling results."""
        if not self._has_run:
            print("Algorithm has not been run yet. Call run_algorithm() first.")
            return

        print("\n" + "=" * 70)
        print("SCHEDULING SUMMARY")
        print("=" * 70)

        # Officer counts
        counts = self.get_all_officers_count()
        print(f"\nOfficers Scheduled:")
        print(f"  Main Officers (M):   {counts['main']:3d}")
        print(f"  SOS Officers (S):    {counts['sos']:3d}")
        print(f"  OT Officers (OT):    {counts['ot']:3d}")
        print(f"  {'─' * 25}")
        print(f"  Total:               {counts['total']:3d}")

        # Optimization info
        if self.optimization_penalty is not None:
            print(f"\nOptimization Penalty: {self.optimization_penalty:.2f}")

        # Matrix info
        if self.main_counter_matrix_np is not None:
            print(f"\nCounter Matrix Shape: {self.main_counter_matrix_np.shape}")
            print(f"  Counters: {self.main_counter_matrix_np.shape[0]}")
            print(f"  Time Slots: {self.main_counter_matrix_np.shape[1]}")

        # Statistics preview
        if self.statistics:
            print("\n" + "─" * 70)
            print("Statistics Preview:")
            print("─" * 70)
            print(self.statistics[0][:400] + "...")

        print("\n" + "=" * 70)

    def reset(self):
        """
        Reset the manager to initial state, clearing all results.

        This allows reusing the same manager instance for multiple runs.
        """
        self.main_officers = {}
        self.sos_officers = []
        self.ot_officers = []
        self.main_counter_matrix_np = None
        self.final_counter_matrix_np = None
        self.officer_schedules = None
        self.statistics = None
        self.optimization_penalty = None
        self._has_run = False

        # Reset orchestrator
        self.orchestrator = RosterAlgorithmOrchestrator(mode=self.mode)

        print("ScheduleManager has been reset.")

    def __repr__(self):
        counts = self.get_all_officers_count()
        status = "initialized" if not self._has_run else "scheduled"
        return (
            f"ScheduleManager(mode={self.mode.value}, status={status}, "
            f"officers={counts['total']} "
            f"[M:{counts['main']}, S:{counts['sos']}, OT:{counts['ot']}])"
        )


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SCHEDULE MANAGER - EXAMPLE USAGE")
    print("=" * 70)

    # Test with ARRIVAL mode
    print("\n### Testing ARRIVAL Mode ###\n")

    manager = ScheduleManager(mode=OperationMode.ARRIVAL)
    print(f"Created: {manager}\n")

    # Example inputs
    main_officers_reported = "1-18"
    report_gl_counters = "4AC1, 8AC11, 12AC21, 16AC31"
    handwritten_counters = "3AC12,5AC13"
    ot_counters = "2,20,40"
    sos_timings = (
        "(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, "
        "(AC23)1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830"
    )
    ro_ra_officers = "3RO2100, 11RO1700,15RO2130"

    # Run the algorithm
    results = manager.run_algorithm(
        main_officers_reported,
        report_gl_counters,
        sos_timings,
        ro_ra_officers,
        handwritten_counters,
        ot_counters,
    )

    # Print summary
    manager.print_summary()

    # Test individual accessors
    print("\n" + "=" * 70)
    print("TESTING ACCESSOR METHODS")
    print("=" * 70)

    # Get specific officer schedule
    m1_schedule = manager.get_officer_schedule("M1")
    if m1_schedule:
        print(f"\nOfficer M1 schedule (first 10 slots): {m1_schedule[:10]}")

    # Get officers by type
    main_officer_keys = manager.get_officer_keys_by_type('M')
    print(f"\nMain officer keys (first 5): {main_officer_keys[:5]}")

    # Export to dict
    export_data = manager.export_schedules_to_dict()
    print(f"\nExported data keys: {list(export_data.keys())}")
    print(f"Configuration: {export_data['config']}")

    print("\n" + "=" * 70)
    print(f"Final State: {manager}")
    print("=" * 70)

    # Test DEPARTURE mode
    print("\n\n### Testing DEPARTURE Mode ###\n")

    manager_dep = ScheduleManager(mode=OperationMode.DEPARTURE)
    print(f"Created: {manager_dep}")

    # Note: You would need departure-compatible test data here
    # For now, just showing the manager can be created
    print("Departure mode manager created successfully.")
    print(f"Number of counters in DEPARTURE mode: {manager_dep.num_counters}")