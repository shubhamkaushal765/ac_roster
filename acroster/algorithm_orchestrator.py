"""
Algorithm Orchestrator - Main coordinator for roster generation.

This module orchestrates the entire roster generation process by coordinating
between all subsystems: roster building, SOS scheduling, optimization, and
counter assignment.
"""

from typing import Dict, List, Tuple
import numpy as np

from acroster.config import MODE_CONFIG, OperationMode
from acroster.officer import MainOfficer, SOSOfficer
from acroster.counter import CounterMatrix
from acroster.roster_builder import RosterBuilder, LastCounterAssigner
from acroster.sos_scheduler import SOSOfficerBuilder, BreakScheduleGenerator
from acroster.optimization import ScheduleOptimizer
from acroster.assignment_engine import (
    CounterAssignmentEngine,
    SOSAssignmentEngine,
    MatrixConverter
)
from acroster.statistics import StatisticsGenerator


class RosterAlgorithmOrchestrator:
    """
    Orchestrates the complete roster generation algorithm.

    Coordinates between:
    - Main officer roster building
    - SOS officer scheduling
    - Schedule optimization
    - Counter assignment
    - Statistics generation
    """

    def __init__(self, mode: OperationMode):
        """
        Initialize the orchestrator.

        Args:
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        self.mode = mode
        self.config = MODE_CONFIG[mode]

        # Initialize subsystems
        self.roster_builder = RosterBuilder(mode)
        self.last_counter_assigner = LastCounterAssigner(mode)
        self.sos_builder = SOSOfficerBuilder()
        self.break_generator = BreakScheduleGenerator()
        self.optimizer = ScheduleOptimizer(beam_width=20)
        self.counter_engine = CounterAssignmentEngine(mode)
        self.sos_engine = SOSAssignmentEngine(mode)
        self.stats_generator = StatisticsGenerator(mode)
        self.matrix_converter = MatrixConverter()

    def run(
            self,
            main_officers_reported: str,
            report_gl_counters: str,
            sos_timings: str,
            ro_ra_officers: str,
            handwritten_counters: str,
            ot_counters: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]], List[str]]:
        """
        Run the complete roster generation algorithm.

        Args:
            main_officers_reported: Officers who reported (e.g., "1-18")
            report_gl_counters: Ground level counters (e.g., "4AC1, 8AC11")
            sos_timings: SOS officer timings (e.g., "(AC22)1000-1300, 2000-2200")
            ro_ra_officers: Late/early adjustments (e.g., "3RO2100, 11RO1700")
            handwritten_counters: Takeover counters (e.g., "3AC12,5AC13")
            ot_counters: OT officer counters (e.g., "2,20,40")

        Returns:
            Tuple of:
                - Main counter matrix with OT (numpy array)
                - Final counter matrix (numpy array)
                - Officer schedule dict
                - List of statistics strings
        """
        # Step 1: Build main officer rosters
        print("=== Step 1: Building Main Officer Rosters ===")
        main_officers, reported_officers, valid_ro_ra = self._build_main_officers(
            main_officers_reported,
            report_gl_counters,
            ro_ra_officers,
            handwritten_counters
        )

        # Step 2: Assign last counters
        print("\n=== Step 2: Assigning Last Counters ===")
        main_officers = self._assign_last_counters(
            main_officers,
            reported_officers,
            valid_ro_ra
        )

        # Step 3: Convert to counter matrix and add OT officers
        print("\n=== Step 3: Adding OT Officers ===")
        counter_matrix_with_ot, ot_officers = self._add_ot_officers(
            main_officers,
            ot_counters
        )

        # Generate first statistics
        stats1 = self.stats_generator.generate_statistics(
            counter_matrix_with_ot.to_matrix()
        )

        # Step 4: Process SOS officers if provided
        if len(sos_timings) > 0:
            print("\n=== Step 4: Processing SOS Officers ===")
            final_matrix, officer_schedule, stats2 = self._process_sos_officers(
                sos_timings,
                main_officers,
                counter_matrix_with_ot
            )

            return (
                counter_matrix_with_ot.to_matrix(),
                final_matrix,
                officer_schedule,
                [stats1, stats2]
            )
        else:
            # No SOS officers - return main officers only
            officer_schedule = {
                k: v.schedule.tolist()
                for k, v in main_officers.items()
            }

            main_matrix = counter_matrix_with_ot.to_matrix()
            return (
                main_matrix,
                main_matrix,
                officer_schedule,
                [stats1, stats1]
            )

    def _build_main_officers(
            self,
            main_officers_reported: str,
            report_gl_counters: str,
            ro_ra_officers: str,
            handwritten_counters: str
    ) -> Tuple[Dict[str, MainOfficer], set, List[Tuple]]:
        """Build and configure main officers."""
        # Build main officers with RA/RO adjustments
        main_officers, reported_officers, valid_ro_ra = \
            self.roster_builder.build_main_officers(
                main_officers_reported,
                report_gl_counters,
                ro_ra_officers
            )

        # Apply takeover counters
        main_officers = self.roster_builder.apply_takeover_counters(
            main_officers,
            handwritten_counters
        )

        print(f"Built {len(main_officers)} main officers")
        print(f"Reported: {sorted(reported_officers)}")
        print(f"Adjustments: {valid_ro_ra}")

        return main_officers, reported_officers, valid_ro_ra

    def _assign_last_counters(
            self,
            main_officers: Dict[str, MainOfficer],
            reported_officers: set,
            valid_ro_ra: List[Tuple]
    ) -> Dict[str, MainOfficer]:
        """Assign last counters to eligible officers."""
        # Convert to counter matrix to find empty counters
        counter_matrix_wo_last = self.counter_engine.officers_to_counter_matrix(
            main_officers
        )

        # Get last counter slots and empty counters
        officer_last_counter = self.last_counter_assigner.get_last_counter_slots(
            reported_officers,
            valid_ro_ra
        )

        empty_counters = self.last_counter_assigner.find_empty_counters_from_slot(
            counter_matrix_wo_last,
            from_slot=42
        )

        print(f"Officers eligible for last counter: {list(officer_last_counter.keys())}")
        print(f"Empty counters from slot 42: {empty_counters}")

        # Apply last counters
        main_officers = self.last_counter_assigner.assign_last_counters(
            main_officers,
            officer_last_counter,
            empty_counters
        )

        return main_officers

    def _add_ot_officers(
            self,
            main_officers: Dict[str, MainOfficer],
            ot_counters: str
    ) -> Tuple[CounterMatrix, List]:
        """Add OT officers to counter matrix."""
        # Convert main officers to counter matrix
        counter_matrix = self.counter_engine.officers_to_counter_matrix(main_officers)

        # Add OT officers
        counter_matrix_with_ot, ot_officers = self.counter_engine.assign_ot_officers(
            counter_matrix,
            ot_counters
        )

        print(f"Added {len(ot_officers)} OT officers")

        return counter_matrix_with_ot, ot_officers

    def _process_sos_officers(
            self,
            sos_timings: str,
            main_officers: Dict[str, MainOfficer],
            counter_matrix_with_ot: CounterMatrix
    ) -> Tuple[np.ndarray, Dict, str]:
        """Process SOS officers: build, optimize, and assign."""
        # Build SOS officers
        sos_officers, pre_assigned_counter_dict = self.sos_builder.build_sos_officers(
            sos_timings
        )
        print(f"Built {len(sos_officers)} SOS officers")
        print(f"Pre-assigned counters: {pre_assigned_counter_dict}")

        # Generate break schedules
        sos_officers = self.break_generator.generate_break_schedules(sos_officers)

        # Optimize schedule selection
        chosen_indices, best_work_count, min_penalty = self.optimizer.optimize(
            sos_officers,
            main_officers
        )
        print(f"Optimization penalty: {min_penalty}")
        print(f"Best work count summary: min={best_work_count.min()}, "
              f"max={best_work_count.max()}, mean={best_work_count.mean():.1f}")

        # Get working intervals for SOS officers
        schedule_intervals_to_officers = self._extract_working_intervals(sos_officers)
        print(f"Schedule intervals to officers: {schedule_intervals_to_officers}")

        # Assign SOS officers to counters
        sos_counter_matrix = self.sos_engine.assign_sos_officers(
            pre_assigned_counter_dict,
            schedule_intervals_to_officers,
            counter_matrix_with_ot
        )

        # Merge matrices
        main_matrix = counter_matrix_with_ot.to_matrix()
        sos_matrix = sos_counter_matrix.to_matrix()

        final_matrix = self.matrix_converter.merge_prefixed_matrices(
            sos_matrix,
            main_matrix
        )

        # Convert to officer schedule format
        officer_schedule = self.matrix_converter.counter_to_officer_schedule(
            final_matrix
        )

        # Generate final statistics
        stats2 = self.stats_generator.generate_statistics(final_matrix)

        return final_matrix, officer_schedule, stats2

    def _extract_working_intervals(
            self,
            sos_officers: List[SOSOfficer]
    ) -> Dict[Tuple[int, int], List[int]]:
        """Extract working intervals from SOS officers."""
        schedule_intervals_to_officers = {}

        for officer in sos_officers:
            intervals = officer.get_working_intervals()
            for interval in intervals:
                if interval not in schedule_intervals_to_officers:
                    schedule_intervals_to_officers[interval] = []
                schedule_intervals_to_officers[interval].append(
                    officer.officer_id - 1
                )

        return schedule_intervals_to_officers


# Convenience function for backward compatibility
def run_algo(
        main_officers_reported: str,
        report_gl_counters: str,
        sos_timings: str,
        ro_ra_officers: str,
        handwritten_counters: str,
        ot_counters: str,
        mode: OperationMode = OperationMode.ARRIVAL
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]], List[str]]:
    """
    Run the roster generation algorithm (backward compatibility function).

    Args:
        main_officers_reported: Officers who reported
        report_gl_counters: Ground level counters
        sos_timings: SOS officer timings
        ro_ra_officers: Late/early adjustments
        handwritten_counters: Takeover counters
        ot_counters: OT officer counters
        mode: Operation mode (default: ARRIVAL)

    Returns:
        Tuple of (main_matrix, final_matrix, officer_schedule, statistics)
    """
    orchestrator = RosterAlgorithmOrchestrator(mode)
    return orchestrator.run(
        main_officers_reported,
        report_gl_counters,
        sos_timings,
        ro_ra_officers,
        handwritten_counters,
        ot_counters
    )


if __name__ == "__main__":
    # Test with default inputs
    print("=== Running Algorithm Test ===\n")

    main_officers_reported = "1-18"
    report_gl_counters = "4AC1, 8AC11, 12AC21, 16AC31"
    handwritten_counters = "3AC12,5AC13"
    ot_counters = "2,20,40"
    sos_timings = (
        "(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, "
        "(AC23)1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, "
        "1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945"
    )
    ro_ra_officers = "3RO2100, 11RO1700,15RO2130"

    results = run_algo(
        main_officers_reported,
        report_gl_counters,
        sos_timings,
        ro_ra_officers,
        handwritten_counters,
        ot_counters,
        mode=OperationMode.ARRIVAL
    )

    print("\n=== Results Summary ===")
    print(f"Main matrix shape: {results[0].shape}")
    print(f"Final matrix shape: {results[1].shape}")
    print(f"Officer schedules: {len(results[2])} officers")
    print(f"\nStatistics:\n{results[3][0]}")