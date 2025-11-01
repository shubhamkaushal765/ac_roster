"""
Object-Oriented Scheduling System for Officers and Counters

This module provides a modular, class-based approach to scheduling officers
across counters with support for breaks, overtime, and special assignments.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

import numpy as np
import plotly.graph_objects as go


# ============================================================================
# Configuration Class
# ============================================================================


class ScheduleConfig:
    """Global configuration for the scheduling system."""

    NUM_SLOTS = 48
    NUM_COUNTERS = 41
    START_HOUR = 10

    # Priority list for counter assignment
    COUNTER_PRIORITY_LIST = [41] + [
        n for offset in range(0, 10) for n in range(40 - offset, 0, -10)
    ]

    @staticmethod
    def hhmm_to_slot(hhmm: str) -> int:
        """Convert hhmm string to a slot index (0–47)."""
        t = int(hhmm)
        h = t // 100
        m = t % 100
        slot = (h - ScheduleConfig.START_HOUR) * 4 + (m // 15)
        return max(0, min(ScheduleConfig.NUM_SLOTS - 1, slot))

    @staticmethod
    def slot_to_hhmm(slot: int) -> str:
        """Convert slot index back to hhmm string."""
        h = ScheduleConfig.START_HOUR + slot // 4
        m = (slot % 4) * 15
        return f"{h:02d}{m:02d}"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Officer:
    """Represents an officer with their schedule."""

    officer_id: int
    schedule: np.ndarray = field(
        default_factory=lambda: np.zeros(ScheduleConfig.NUM_SLOTS)
    )
    is_main: bool = True

    def __post_init__(self):
        if not isinstance(self.schedule, np.ndarray):
            self.schedule = np.array(self.schedule)

    @property
    def key(self) -> str:
        """Return the officer's key identifier."""
        prefix = "M" if self.is_main else "S"
        return f"{prefix}{self.officer_id}"

    def apply_late_arrival(self, slot: int):
        """Mark officer as unavailable before the given slot."""
        self.schedule[:slot] = 0

    def apply_early_departure(self, slot: int):
        """Mark officer as unavailable after the given slot."""
        self.schedule[slot:] = 0

    def assign_counter(self, counter: int, start_slot: int, end_slot: int):
        """Assign a counter to specific time slots."""
        self.schedule[start_slot:end_slot] = counter


@dataclass
class Adjustment:
    """Represents a timing adjustment for an officer."""

    officer_id: int
    adjustment_type: str  # 'RA' (Report After) or 'RO' (Report Out)
    time_slot: int


# ============================================================================
# Template Manager
# ============================================================================


class TemplateManager:
    """Manages officer schedule templates."""

    @staticmethod
    def create_4main_roster(full_counters: List[int]) -> Tuple[List[int], ...]:
        """Create roster patterns for 4 main officers."""
        a = (
            [full_counters[0]] * 6
            + [0] * 2
            + [full_counters[1]] * 7
            + [0] * 3
            + [full_counters[2]] * 9
            + [0] * 3
            + [full_counters[0]] * 9
            + [0]
            + [full_counters[1]] * 8
        )

        b = (
            [full_counters[1]] * 8
            + [0] * 2
            + [full_counters[2]] * 8
            + [0] * 3
            + [full_counters[0]] * 9
            + [0] * 3
            + [full_counters[1]] * 7
            + [0]
            + [full_counters[2]] * 7
        )

        c = (
            [full_counters[2]] * 10
            + [0] * 2
            + [full_counters[0]] * 9
            + [0] * 3
            + [full_counters[1]] * 9
            + [0] * 3
            + [full_counters[2]] * 5
            + [0] * 7
        )

        d = (
            [0] * 6
            + [full_counters[0]] * 6
            + [0] * 2
            + [full_counters[1]] * 10
            + [0] * 3
            + [full_counters[2]] * 9
            + [0] * 3
            + [full_counters[0]] * 9
        )

        return (a, b, c, d)

    @staticmethod
    def initialize_main_officers_template() -> Dict[str, np.ndarray]:
        """Initialize template schedules for all main officers."""
        main_officers = {}

        # Hardcoded templates for officers 1-8
        main_officers[1] = (
            [41] * 6
            + [0] * 2
            + [30] * 7
            + [0] * 3
            + [20] * 9
            + [0] * 3
            + [40] * 9
            + [0]
            + [30] * 8
        )
        main_officers[2] = (
            [30] * 8
            + [0] * 2
            + [20] * 8
            + [0] * 3
            + [41] * 9
            + [0] * 3
            + [30] * 7
            + [0]
            + [20] * 7
        )
        main_officers[3] = (
            [20] * 10
            + [0] * 2
            + [41] * 9
            + [0] * 3
            + [30] * 9
            + [0] * 3
            + [20] * 5
            + [0] * 7
        )
        main_officers[4] = (
            [0] * 6
            + [40] * 6
            + [0] * 2
            + [30] * 10
            + [0] * 3
            + [20] * 9
            + [0] * 3
            + [41] * 9
        )
        main_officers[5] = (
            [40] * 6
            + [0] * 2
            + [9] * 7
            + [0] * 3
            + [29] * 9
            + [0] * 3
            + [41] * 9
            + [0]
            + [9] * 8
        )
        main_officers[6] = (
            [9] * 8
            + [0] * 2
            + [29] * 8
            + [0] * 3
            + [40] * 9
            + [0] * 3
            + [9] * 7
            + [0]
            + [29] * 7
        )
        main_officers[7] = (
            [29] * 10
            + [0] * 2
            + [40] * 9
            + [0] * 3
            + [9] * 9
            + [0] * 3
            + [29] * 5
            + [0] * 7
        )
        main_officers[8] = (
            [0] * 6
            + [41] * 6
            + [0] * 2
            + [9] * 10
            + [0] * 3
            + [29] * 9
            + [0] * 3
            + [40] * 9
        )

        # Define groups of officers and their roster counters
        groups = [
            ([9, 10, 11, 12], [19, 38, 10]),
            ([13, 14, 15, 16], [28, 17, 39]),
            ([17, 18, 19, 20], [7, 27, 18]),
            ([21, 22, 23, 24], [37, 8, 26]),
            ([25, 26, 27, 28], [15, 35, 5]),
            ([29, 30, 31, 32], [24, 16, 36]),
            ([33, 34, 35, 36], [6, 25, 13]),
            ([37, 38, 39, 40], [34, 3, 23]),
        ]

        # Generate schedules for grouped officers
        for m_no, roster in groups:
            results = TemplateManager.create_4main_roster(roster)
            for i, officer in enumerate(m_no):
                main_officers[officer] = results[i]

        # Convert to numpy arrays with M prefix
        return {f"M{i}": np.array(v) for i, v in main_officers.items()}


# ============================================================================
# Input Parser
# ============================================================================


class InputParser:
    """Parses and validates user input strings."""

    @staticmethod
    def parse_officer_range(input_str: str) -> Set[int]:
        """Parse officer IDs from range format (e.g., '1-10, 15, 20-25')."""
        reported_officers = set()
        parts = input_str.split(",")

        for part in parts:
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                for i in range(int(start), int(end) + 1):
                    reported_officers.add(i)
            else:
                reported_officers.add(int(part))

        return reported_officers

    @staticmethod
    def parse_adjustments(
        input_str: str, reported_officers: Set[int]
    ) -> List[Adjustment]:
        """
        Parse and validate officer timing adjustments.
        Format: 'officer_id(RA|RO)HHMM' (e.g., '3RO2100, 5RA1030')
        """
        valid_adjustments = []

        if not input_str.strip():
            return valid_adjustments

        entries = input_str.split(",")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            # Validate format
            if not re.match(r"^\d+(RA|RO)\d{4}$", entry):
                print(f"⚠️ Skipping {entry}: invalid format")
                continue

            # Extract components
            if "RA" in entry:
                idx = entry.index("RA")
                officer_id = int(entry[:idx])
                hhmm = entry[idx + 2 :]
                adj_type = "RA"
            else:
                idx = entry.index("RO")
                officer_id = int(entry[:idx])
                hhmm = entry[idx + 2 :]
                adj_type = "RO"

            # Validate officer is in reported list
            if officer_id not in reported_officers:
                print(f"⚠️ Skipping {entry}: officer {officer_id} not in reported list")
                continue

            # Validate time
            h = int(hhmm[:2])
            m = int(hhmm[2:])

            if not (10 <= h <= 22):
                print(f"⚠️ Skipping {entry}: hour {h} out of range (10-22)")
                continue
            if m not in (0, 15, 30, 45):
                print(f"⚠️ Skipping {entry}: minutes must be 00/15/30/45")
                continue
            if h == 22 and m > 0:
                print(f"⚠️ Skipping {entry}: cannot exceed 2200")
                continue

            slot = ScheduleConfig.hhmm_to_slot(hhmm)
            valid_adjustments.append(Adjustment(officer_id, adj_type, slot))

        return valid_adjustments

    @staticmethod
    def parse_counter_assignments(input_str: str) -> List[Tuple[int, int]]:
        """Parse counter assignments (e.g., '4AC1, 8AC11')."""
        assignments = []
        report_list = [s.strip() for s in input_str.split(",")]

        for assignment in report_list:
            if "AC" in assignment:
                idx = assignment.index("AC")
                officer_id = int(assignment[:idx])
                counter_no = int(assignment[idx + 2 :])
                assignments.append((officer_id, counter_no))

        return assignments


# ============================================================================
# Main Officer Scheduler
# ============================================================================


class MainOfficerScheduler:
    """Handles scheduling for main officers."""

    def __init__(self, template: Dict[str, np.ndarray]):
        self.template = template
        self.schedule: Dict[str, np.ndarray] = {}
        self.reported_officers: Set[int] = set()

    def generate_schedule(
        self,
        reported_officers_str: str,
        counter_assignments_str: str,
        adjustments_str: str,
    ) -> Tuple[Dict[str, np.ndarray], Set[int], List[Adjustment]]:
        """Generate the complete schedule for main officers."""

        # Parse inputs
        self.reported_officers = InputParser.parse_officer_range(reported_officers_str)
        adjustments = InputParser.parse_adjustments(
            adjustments_str, self.reported_officers
        )
        counter_assignments = InputParser.parse_counter_assignments(
            counter_assignments_str
        )

        # Build initial schedules from templates
        for officer_id in self.reported_officers:
            officer_key = f"M{officer_id}"
            if officer_key not in self.template:
                continue

            self.schedule[officer_key] = self.template[officer_key].copy()

        # Apply timing adjustments
        for adjustment in adjustments:
            officer_key = f"M{adjustment.officer_id}"
            if officer_key in self.schedule:
                if adjustment.adjustment_type == "RA":
                    self.schedule[officer_key][: adjustment.time_slot] = 0
                elif adjustment.adjustment_type == "RO":
                    self.schedule[officer_key][adjustment.time_slot :] = 0

        # Apply counter assignments
        for officer_id, counter_no in counter_assignments:
            # Only apply to officers divisible by 4
            if officer_id % 4 == 0:
                officer_key = f"M{officer_id}"
                if officer_key in self.schedule:
                    # Assign counter to first 5 slots (0-4 inclusive)
                    for slot in range(0, 5):
                        self.schedule[officer_key][slot] = counter_no

        return self.schedule, self.reported_officers, adjustments

    def update_last_counter_assignments(
        self, officer_last_counter: Dict[str, int], empty_counters: List[int]
    ):
        """Update the last counter assignments based on algorithm results."""
        for officer_key, counter in officer_last_counter.items():
            if officer_key in self.schedule:
                # Find last work slot and update
                schedule = self.schedule[officer_key]
                last_work_slot = None
                for i in range(len(schedule) - 1, -1, -1):
                    if schedule[i] != 0:
                        last_work_slot = i
                        break

                if last_work_slot is not None:
                    self.schedule[officer_key][last_work_slot] = counter

    def add_handwritten_counters(self, handwritten_str: str):
        """Add handwritten counter assignments to first 2 slots."""
        if not handwritten_str.strip():
            return

        # Find all officer-counter pairs using regex
        pairs = re.findall(r"(\d+)\s*[aA]\s*[cC]\s*(\d+)", handwritten_str)

        for officer_str, counter_str in pairs:
            officer_key = f"M{officer_str}"
            new_counter = int(counter_str)

            if officer_key in self.schedule:
                self.schedule[officer_key][0:2] = [new_counter, new_counter]


# ============================================================================
# Counter Matrix Manager
# ============================================================================


class CounterMatrix:
    """Manages the counter-to-officer assignment matrix."""

    def __init__(self, num_counters: int = 41, num_slots: int = 48):
        self.num_counters = num_counters
        self.num_slots = num_slots
        self.matrix = np.zeros((num_counters, num_slots), dtype=object)

    @classmethod
    def from_officer_schedule(
        cls, officer_schedule: Dict[str, np.ndarray]
    ) -> "CounterMatrix":
        """Create a counter matrix from an officer schedule."""
        instance = cls()

        # Initialize counters dictionary
        counters = {
            i: [0] * instance.num_slots for i in range(1, instance.num_counters + 1)
        }

        for officer_key, schedule in officer_schedule.items():
            for slot_idx, counter_assigned in enumerate(schedule):
                if counter_assigned != 0:
                    counter_idx = int(counter_assigned)
                    if 1 <= counter_idx <= instance.num_counters:
                        counters[counter_idx][slot_idx] = officer_key

        # Convert to matrix format
        for counter_num in range(1, instance.num_counters + 1):
            instance.matrix[counter_num - 1, :] = counters[counter_num]

        return instance

    def add_overtime_counters(self, ot_counters_str: str):
        """Add overtime (OT) designations to specified counters."""
        if not ot_counters_str.strip():
            return

        self.matrix = self.matrix.astype(object)
        ot_list = [int(x.strip()) for x in ot_counters_str.split(",") if x.strip()]

        for i, ot_counter in enumerate(ot_list):
            ot_id = f"OT{i + 1}"
            self.matrix[ot_counter - 1, 0:2] = [ot_id, ot_id]

        # Replace remaining zeros with '0' string
        self.matrix[self.matrix == 0] = "0"

    def to_officer_schedule(self) -> Dict[str, np.ndarray]:
        """Convert counter matrix back to officer schedule format."""
        officer_schedule = defaultdict(lambda: np.zeros(self.num_slots, dtype=int))

        for counter_idx in range(self.num_counters):
            for slot_idx in range(self.num_slots):
                officer_key = self.matrix[counter_idx, slot_idx]
                if officer_key not in ["0", 0, ""]:
                    counter_num = counter_idx + 1
                    officer_schedule[officer_key][slot_idx] = counter_num

        return dict(officer_schedule)


# ============================================================================
# Statistics Generator
# ============================================================================


class StatisticsGenerator:
    """Generates statistical summaries of counter assignments."""

    @staticmethod
    def generate(counter_matrix: np.ndarray) -> str:
        """Generate statistics text from counter matrix."""
        statistics_list = []
        counter_matrix = np.array(counter_matrix)
        num_rows, num_slots = counter_matrix.shape

        # Define row groups for second line
        row_groups = [range(0, 10), range(10, 20), range(20, 30), range(30, 40)]

        for slot in range(num_slots):
            # First line: count of non-zero main counters / non-zero other counters
            count1 = np.sum(counter_matrix[0:40, slot] != "0")
            count2 = np.sum(counter_matrix[40:, slot] != "0")
            first_line = f"{ScheduleConfig.slot_to_hhmm(slot)}: "
            first_line2 = f"{count1:02d}/{count2:02d}"

            # Second line: counts per row group
            group_counts = []
            for g in row_groups:
                group_counts.append(str(np.sum(counter_matrix[g, slot] != "0")))
            second_line = "/".join(group_counts)

            statistics_list.append((first_line, first_line2, second_line))

        # Filter statistics (keep every 4th or when counts change)
        stats = []
        for i, t in enumerate(statistics_list):
            if i % 4 == 0:
                stats.append(t)
            elif i % 2 == 0 and t[2] != stats[-1][2]:
                stats.append(t)

        # Format output
        output_text = "ACar \n\n"
        for stat in stats:
            output_text += f"{stat[0]}{stat[1]}\n{stat[2]}\n\n"

        return output_text


# ============================================================================
# Visualization
# ============================================================================


class ScheduleVisualizer:
    """Creates visual representations of schedules."""

    @staticmethod
    def plot_officer_timetable(officer_schedule: Dict[str, np.ndarray]) -> go.Figure:
        """Generate a heatmap visualization of officer schedules."""
        num_slots = ScheduleConfig.NUM_SLOTS

        # Sort officers
        sorted_officers = sorted(
            officer_schedule.keys(), key=lambda x: (x[0], int(x[1:]))
        )

        # Build matrix
        matrix_data = []
        for officer_key in sorted_officers:
            matrix_data.append(officer_schedule[officer_key])

        matrix_data = np.array(matrix_data)

        # Create heatmap
        heatmap = go.Heatmap(
            z=matrix_data,
            x=list(range(num_slots)),
            y=sorted_officers,
            colorscale="Viridis",
            showscale=True,
        )

        # Add annotations and shapes for counter labels
        annotations = []
        shapes = []

        for i, officer_id in enumerate(sorted_officers):
            t = 0
            while t < num_slots:
                counter = officer_schedule[officer_id][t]
                if counter != 0:
                    # Find end of this counter block
                    t_end = t
                    while (
                        t_end < num_slots
                        and officer_schedule[officer_id][t_end] == counter
                    ):
                        t_end += 1

                    # Add label at center
                    center_x = (t + t_end - 1) / 2
                    annotations.append(
                        dict(
                            x=center_x,
                            y=officer_id,
                            text=f"C{counter}",
                            showarrow=False,
                            font=dict(color="black", size=18),
                        )
                    )

                    # Add border
                    shapes.append(
                        dict(
                            type="rect",
                            x0=t - 0.5,
                            x1=t_end - 0.5,
                            y0=i - 0.5,
                            y1=i + 0.5,
                            line=dict(color="black", width=1),
                            fillcolor="rgba(0,0,0,0)",
                        )
                    )

                    t = t_end
                else:
                    t += 1

        # Build figure
        fig = go.Figure(data=[heatmap])
        fig.update_layout(
            title="Officer Timetable (Counter Assignments)",
            xaxis_title="Time Slot",
            yaxis_title="Officer",
            width=900,
            height=900,
            annotations=annotations,
            shapes=shapes,
            yaxis_autorange="reversed",
            dragmode=False,
        )

        return fig


# ============================================================================
# SOS Officer Scheduling (Placeholder for extended functionality)
# ============================================================================


class SOSOfficerScheduler:
    """Handles scheduling for SOS (special) officers."""

    def __init__(self):
        self.schedules = {}

    def parse_timings(self, sos_timings_str: str):
        """Parse SOS officer timing strings."""
        # This would contain the complex parsing logic from the original
        # Left as a placeholder for the full implementation
        pass

    def generate_schedules(self):
        """Generate optimized SOS officer schedules."""
        # This would contain the beam search and optimization logic
        # Left as a placeholder for the full implementation
        pass


# ============================================================================
# Main Scheduling Engine
# ============================================================================


class SchedulingEngine:
    """Main engine that coordinates all scheduling components."""

    def __init__(self):
        self.config = ScheduleConfig()
        self.template_manager = TemplateManager()
        self.main_scheduler = None
        self.counter_matrix = None

    def run(
        self,
        main_officers_reported: str,
        report_gl_counters: str,
        sos_timings: str,
        ro_ra_officers: str,
        handwritten_counters: str,
        ot_counters: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict, List[str]]:
        """
        Run the complete scheduling algorithm.

        Args:
            main_officers_reported: Range of main officers (e.g., "1-18")
            report_gl_counters: Counter assignments (e.g., "4AC1, 8AC11")
            sos_timings: SOS officer timings
            ro_ra_officers: Late/early adjustments (e.g., "3RO2100")
            handwritten_counters: Manual counter overrides (e.g., "3AC12")
            ot_counters: Overtime counter list (e.g., "2,20,40")

        Returns:
            Tuple of (main_matrix, final_matrix, officer_schedule, stats_list)
        """

        # Initialize template
        template = self.template_manager.initialize_main_officers_template()

        # Create main officer scheduler
        self.main_scheduler = MainOfficerScheduler(template)

        # Generate main officer schedules
        main_schedule, reported_officers, valid_adjustments = (
            self.main_scheduler.generate_schedule(
                main_officers_reported, report_gl_counters, ro_ra_officers
            )
        )

        # Convert to counter matrix
        counter_matrix_wo_last = CounterMatrix.from_officer_schedule(main_schedule)

        # Get last counter assignments (simplified - full logic would go here)
        officer_last_counter, empty_counters = self._get_last_counter_info(
            reported_officers, valid_adjustments, counter_matrix_wo_last.matrix
        )

        # Update last counter assignments
        self.main_scheduler.update_last_counter_assignments(
            officer_last_counter, empty_counters
        )

        # Add handwritten counter overrides
        self.main_scheduler.add_handwritten_counters(handwritten_counters)

        # Create final counter matrix
        self.counter_matrix = CounterMatrix.from_officer_schedule(
            self.main_scheduler.schedule
        )

        # Add overtime counters
        self.counter_matrix.add_overtime_counters(ot_counters)

        # Generate statistics
        stats1 = StatisticsGenerator.generate(self.counter_matrix.matrix)

        # Handle SOS officers if provided
        if len(sos_timings) > 0:
            # SOS logic would go here
            # For now, return main matrix as final matrix
            stats2 = stats1
            final_matrix = self.counter_matrix.matrix
        else:
            stats2 = stats1
            final_matrix = self.counter_matrix.matrix

        officer_schedule = self.counter_matrix.to_officer_schedule()

        return (
            self.counter_matrix.matrix,
            final_matrix,
            officer_schedule,
            [stats1, stats2],
        )

    def _get_last_counter_info(
        self,
        reported_officers: Set[int],
        adjustments: List[Adjustment],
        counter_matrix: np.ndarray,
    ) -> Tuple[Dict[str, int], List[int]]:
        """Get officer last counter and empty counters (simplified)."""
        # This is a simplified version - full logic would analyze
        # the counter matrix to determine optimal last counter assignments
        officer_last_counter = {}
        empty_counters = []

        return officer_last_counter, empty_counters


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for testing."""

    # Test inputs
    main_officers_reported = "1-18"
    report_gl_counters = "4AC1, 8AC11, 12AC21, 16AC31"
    handwritten_counters = "3AC12,5AC13"
    ot_counters = "2,20,40"
    sos_timings = ""  # Simplified for initial testing
    ro_ra_officers = "3RO2100, 11RO1700,15RO2130"

    # Create and run engine
    engine = SchedulingEngine()
    results = engine.run(
        main_officers_reported,
        report_gl_counters,
        sos_timings,
        ro_ra_officers,
        handwritten_counters,
        ot_counters,
    )

    main_matrix, final_matrix, officer_schedule, stats = results

    print("=== Statistics ===")
    print(stats[0])
    print("\n=== Matrix Shape ===")
    print(f"Main Matrix: {main_matrix.shape}")
    print(f"Final Matrix: {final_matrix.shape}")


if __name__ == "__main__":
    main()
