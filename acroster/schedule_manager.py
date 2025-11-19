"""
ScheduleManager class for orchestrating the officer scheduling algorithm.

This class serves as the main entry point and coordinator for the scheduling system,
using existing functions from backend_algo.py
"""
#acroster/schedule_manager.py
from typing import Dict, List, Tuple, Optional
import numpy as np

# Import existing classes
from acroster.officer import MainOfficer, OTOfficer, SOSOfficer
from acroster.counter import CounterMatrix
from acroster.config import OperationMode, MODE_CONFIG


class ScheduleManager:
    """
    Main orchestrator class for the officer scheduling system.

    Coordinates the entire scheduling workflow from input parsing through
    optimization to final schedule generation and statistics.
    """

    def __init__(
        self,
        mode: OperationMode,
        num_slots: int = 48,
        start_hour: int = 10
    ):
        """
        Initialize the ScheduleManager.

        Args:
            num_slots: Number of time slots per day (default: 48 for 15-min intervals)
            num_counters: Total number of counters (default: 41)
            start_hour: Starting hour for the schedule (default: 10 for 10:00 AM)
        """
        self.num_slots = num_slots
        self.mode = mode
        self.start_hour = start_hour

        cfg = MODE_CONFIG[mode]
        self.num_counters = cfg["num_counters"]
        self.counter_priority_list = cfg["counter_priority_list"]

        # Officer collections
        self.main_officers: Dict[str, MainOfficer] = {}
        self.sos_officers: List[SOSOfficer] = []
        self.ot_officers: List[OTOfficer] = []

        # Counter matrices
        self.main_counter_matrix: Optional[CounterMatrix] = None
        self.sos_counter_matrix: Optional[CounterMatrix] = None
        self.final_counter_matrix: Optional[CounterMatrix] = None

        # Results
        self.officer_schedules: Optional[Dict[str, List[int]]] = None
        self.statistics: Optional[List[str]] = None
        self.optimization_penalty: Optional[float] = None

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
        # Import functions from backend_algo (assuming they exist)
        # In production, these would be imported at the top
        from backend_algo import (
            init_main_officers_template,
            generate_main_officers_schedule,
            officers_to_counter_matrix,
            get_officer_last_counter_and_empty_counters,
            update_main_officers_schedule_last_counter,
            add_takeover_ot_ctr,
            add_ot_counters,
            build_officer_schedules,
            generate_break_schedules,
            greedy_smooth_schedule_beam,
            add_sos_officers,
            merge_prefixed_matrices,
            counter_to_officer_schedule,
            generate_statistics,
        )

        # Step 1: Generate main officer schedules
        print("Step 1: Generating main officer schedules...")
        main_officers_template = init_main_officers_template(self.mode) 
        self.main_officers, reported_officers, valid_ro_ra = generate_main_officers_schedule(
            main_officers_template,
            main_officers_reported,
            report_gl_counters,
            ro_ra_officers,
        )

        print(f"DEBUG: Main officers created: {list(self.main_officers.keys())}")
        print(f"DEBUG: Reported officers: {reported_officers}")
        print(f"DEBUG: Template keys available: {list(main_officers_template.keys())}")
        print(f"\nDEBUG: Checking template for invalid counters (max: {self.num_counters})...")
        for officer_id, roster in main_officers_template.items():
            unique_counters = np.unique(roster[roster > 0])
            invalid_counters = unique_counters[unique_counters > self.num_counters]
            if len(invalid_counters) > 0:
                print(f"❌ Officer {officer_id} has invalid counters: {invalid_counters}")
                print(f"   Full roster: {roster}")
        # Step 2: Find empty counters and assign last counters
        print("Step 2: Finding empty counters and assigning last counters...")
        counter_matrix_wo_last = officers_to_counter_matrix(self.main_officers, mode = self.mode)
        officer_last_counter, empty_counters_2030 = get_officer_last_counter_and_empty_counters(
            reported_officers, valid_ro_ra, counter_matrix_wo_last, mode = self.mode
        )

        # Step 3: Apply last counters to main officers
        print("Step 3: Applying last counters...")
        self.main_officers = update_main_officers_schedule_last_counter(
            self.main_officers, officer_last_counter, empty_counters_2030
        )

        # Step 4: Apply takeover counters
        print("Step 4: Applying takeover counters...")
        self.main_officers = add_takeover_ot_ctr(self.main_officers, handwritten_counters)

        # Step 5: Convert main officers to counter matrix
        print("Step 5: Converting to counter matrix...")
        counter_matrix = officers_to_counter_matrix(self.main_officers,mode=self.mode)

        # Step 6: Add OT officers
        print("Step 6: Adding OT officers...")
        self.main_counter_matrix, self.ot_officers = add_ot_counters(
            counter_matrix, ot_counters
        )
        main_counter_matrix_np = self.main_counter_matrix.to_matrix()
        stats1 = generate_statistics(main_counter_matrix_np, mode=self.mode)

        # Step 7: Handle SOS officers if provided
        if len(sos_timings.strip()) > 0:
            print("Step 7: Processing SOS officers...")

            # Build SOS officer schedules
            print("  - Building SOS officer schedules...")
            self.sos_officers, pre_assigned_counter_dict = build_officer_schedules(
                sos_timings
            )

            # Generate break schedules
            print("  - Generating break schedules...")
            self.sos_officers = generate_break_schedules(self.sos_officers)

            # Optimize schedule selection
            print("  - Optimizing schedule selection...")
            chosen_schedule_indices, best_work_count, min_penalty = greedy_smooth_schedule_beam(
                self.sos_officers, self.main_officers, beam_width=20
            )
            self.optimization_penalty = min_penalty
            print(f"  - Optimization penalty: {min_penalty}")

            # Get working intervals for SOS officers
            schedule_intervals_to_officers = {}
            for officer in self.sos_officers:
                intervals = officer.get_working_intervals()
                for interval in intervals:
                    if interval not in schedule_intervals_to_officers:
                        schedule_intervals_to_officers[interval] = []
                    schedule_intervals_to_officers[interval].append(
                        officer.officer_id - 1
                    )

            print(f"  - Best work count: {best_work_count}")
            print(f"  - Schedule intervals: {len(schedule_intervals_to_officers)} intervals")

            # Add SOS officers to counter matrix
            print("  - Adding SOS officers to counter matrix...")
            self.sos_counter_matrix = add_sos_officers(
                pre_assigned_counter_dict,
                schedule_intervals_to_officers,
                self.main_counter_matrix,
                mode=self.mode
            )

            # Merge matrices
            sos_counter_matrix_np = self.sos_counter_matrix.to_matrix()
            final_counter_matrix_np = merge_prefixed_matrices(
                main_counter_matrix_np, sos_counter_matrix_np
            )

            # Convert to officer schedule format
            self.officer_schedules = counter_to_officer_schedule(final_counter_matrix_np)

            # Generate final statistics
            stats2 = generate_statistics(final_counter_matrix_np, mode=self.mode)
            self.statistics = [stats1, stats2]

            print("Step 8: Complete! Schedule generated successfully with SOS officers.")

            return (
                main_counter_matrix_np,
                final_counter_matrix_np,
                self.officer_schedules,
                self.statistics,
            )
        else:
            # No SOS officers - return main officer schedules only
            print("Step 7: No SOS officers provided. Finalizing main officers only...")

            # Convert main officers to schedule dict format
            self.officer_schedules = {
                k: v.schedule.tolist() for k, v in self.main_officers.items()
            }
            self.statistics = [stats1, stats1]

            print("Step 8: Complete! Schedule generated successfully.")

            return (
                main_counter_matrix_np,
                main_counter_matrix_np,
                self.officer_schedules,
                self.statistics,
            )

    def get_main_officers(self) -> Dict[str, MainOfficer]:
        """
        Get all main officer objects.

        Returns:
            Dictionary mapping officer keys (e.g., 'M1') to MainOfficer objects
        """
        return self.main_officers

    def get_sos_officers(self) -> List[SOSOfficer]:
        """
        Get all SOS officer objects.

        Returns:
            List of SOSOfficer objects
        """
        return self.sos_officers

    def get_ot_officers(self) -> List[OTOfficer]:
        """
        Get all OT officer objects.

        Returns:
            List of OTOfficer objects
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
        if self.main_counter_matrix is None:
            return None
        return self.main_counter_matrix.to_matrix()

    def get_final_counter_matrix(self) -> Optional[np.ndarray]:
        """
        Get the final counter matrix (includes SOS officers if present).

        Returns:
            Numpy array of counter assignments or None if not yet generated
        """
        if self.final_counter_matrix is not None:
            return self.final_counter_matrix.to_matrix()
        elif self.main_counter_matrix is not None:
            return self.main_counter_matrix.to_matrix()
        return None

    def get_all_officers_count(self) -> Dict[str, int]:
        """
        Get count of each officer type.

        Returns:
            Dictionary with counts: {'main': X, 'sos': Y, 'ot': Z, 'total': T}
        """
        main_count = len(self.main_officers)
        sos_count = len(self.sos_officers)
        ot_count = len(self.ot_officers)

        return {
            'main': main_count,
            'sos': sos_count,
            'ot': ot_count,
            'total': main_count + sos_count + ot_count
        }

    def export_schedules_to_dict(self) -> Dict:
        """
        Export all scheduling data to a dictionary format.

        Returns:
            Dictionary containing all scheduling information
        """
        return {
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

    def reset(self):
        """
        Reset the manager to initial state, clearing all officers and results.
        """
        self.main_officers = {}
        self.sos_officers = []
        self.ot_officers = []
        self.main_counter_matrix = None
        self.sos_counter_matrix = None
        self.final_counter_matrix = None
        self.officer_schedules = None
        self.statistics = None
        self.optimization_penalty = None

    def __repr__(self):
        counts = self.get_all_officers_count()
        status = "initialized" if counts['total'] == 0 else "scheduled"
        return (
            f"ScheduleManager(status={status}, "
            f"officers={counts['total']} "
            f"[M:{counts['main']}, S:{counts['sos']}, OT:{counts['ot']}])"
        )


# Example usage
if __name__ == "__main__":
    # Create schedule manager
    manager = ScheduleManager()

    # Example inputs
    main_officers_reported = "1-18"
    report_gl_counters = "4AC1, 8AC11, 12AC21, 16AC31"
    handwritten_counters = "3AC12,5AC13"
    ot_counters = "2,20,40"
    sos_timings = "(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200"
    ro_ra_officers = "3RO2100, 11RO1700,15RO2130"

    # Run the algorithm
    print("=" * 60)
    print("RUNNING SCHEDULING ALGORITHM")
    print("=" * 60)

    results = manager.run_algorithm(
        main_officers_reported,
        report_gl_counters,
        sos_timings,
        ro_ra_officers,
        handwritten_counters,
        ot_counters,
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Get officer counts
    counts = manager.get_all_officers_count()
    print(f"\nOfficers scheduled:")
    print(f"  - Main Officers: {counts['main']}")
    print(f"  - SOS Officers: {counts['sos']}")
    print(f"  - OT Officers: {counts['ot']}")
    print(f"  - Total: {counts['total']}")

    # Show optimization penalty if available
    if manager.get_optimization_penalty() is not None:
        print(f"\nOptimization Penalty: {manager.get_optimization_penalty():.2f}")

    # Show sample statistics
    stats = manager.get_statistics()
    if stats:
        print("\nStatistics Preview (first 200 chars):")
        print(stats[0][:200] + "...")

    print("\n" + "=" * 60)
    print(f"Manager State: {manager}")
    print("=" * 60)