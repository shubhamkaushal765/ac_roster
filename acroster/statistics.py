"""
Statistics generation for roster manning levels.

Generates formatted statistics showing counter manning levels
by time slot and zone.
"""

import numpy as np
from typing import List, Tuple

from acroster.time_utils import slot_to_hhmm
from acroster.config import MODE_CONFIG, OperationMode


class StatisticsGenerator:
    """Generates manning statistics from counter matrix."""

    def __init__(self, mode: OperationMode):
        """
        Initialize statistics generator.

        Args:
            mode: Operation mode (ARRIVAL or DEPARTURE)
        """
        self.mode = mode
        self.config = MODE_CONFIG[mode]

    def generate_statistics(self, counter_matrix: np.ndarray) -> str:
        """
        Generate manning statistics from counter matrix.

        Args:
            counter_matrix: 2D numpy array (num_counters x num_slots)

        Returns:
            Formatted statistics string
        """
        counter_matrix = np.array(counter_matrix)
        num_rows, num_slots = counter_matrix.shape

        # Get configuration
        num_counters = self.config['num_counters']
        num_car_counters = num_counters - 1  # All except motor counter

        # Get zone ranges from config
        zone1 = self.config['zone1']
        zone2 = self.config['zone2']
        zone3 = self.config['zone3']
        zone4 = self.config['zone4']

        row_groups = [
            range(zone1[0], zone1[1]),
            range(zone2[0], zone2[1]),
            range(zone3[0], zone3[1]),
            range(zone4[0], zone4[1])
        ]

        # Generate statistics for each slot
        statistics_list = []
        for slot in range(num_slots):
            stats = self._compute_slot_statistics(
                counter_matrix,
                slot,
                num_car_counters,
                num_counters,
                row_groups
            )
            statistics_list.append(stats)

        # Filter statistics for output
        filtered_stats = self._filter_statistics(statistics_list)

        # Format output
        output_text = self._format_output(filtered_stats)

        return output_text

    def _compute_slot_statistics(
            self,
            counter_matrix: np.ndarray,
            slot: int,
            num_car_counters: int,
            num_counters: int,
            row_groups: List[range]
    ) -> Tuple[str, str, str]:
        """
        Compute statistics for a single time slot.

        Args:
            counter_matrix: Counter matrix
            slot: Time slot index
            num_car_counters: Number of car counters
            num_counters: Total number of counters
            row_groups: List of zone ranges

        Returns:
            Tuple of (time_label, count_label, zone_counts)
        """
        # Count car counters and motor counter
        count1 = np.sum(counter_matrix[0:num_car_counters, slot] != "0")
        count2 = np.sum(counter_matrix[num_car_counters:num_counters, slot] != "0")

        time_label = f"{slot_to_hhmm(slot)}: "
        count_label = f"{count1:02d}/{count2:02d}"

        # Count by zone
        group_counts = []
        for g in row_groups:
            group_counts.append(str(np.sum(counter_matrix[g, slot] != "0")))
        zone_counts = "/".join(group_counts)

        return (time_label, count_label, zone_counts)

    def _filter_statistics(
            self,
            statistics_list: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """
        Filter statistics to show only relevant time slots.

        Shows:
        - Every 4th slot (hourly)
        - Every 2nd slot if zone distribution changed

        Args:
            statistics_list: Full list of statistics

        Returns:
            Filtered list of statistics
        """
        stats = []
        for i, t in enumerate(statistics_list):
            if i % 4 == 0:
                stats.append(t)
            elif i % 2 == 0 and t[2] != stats[-1][2]:
                stats.append(t)

        return stats

    def _format_output(self, stats: List[Tuple[str, str, str]]) -> str:
        """
        Format statistics for output.

        Args:
            stats: List of statistics tuples

        Returns:
            Formatted string
        """
        output_text = f"{self.config['stats_label']} \n\n"

        for stat in stats:
            output_text += f"{stat[0]}{stat[1]}\n{stat[2]}\n\n"

        return output_text


# Convenience function for backward compatibility
def generate_statistics(
        counter_matrix: np.ndarray,
        mode: OperationMode = None
) -> str:
    """
    Generate manning statistics from counter matrix.

    Args:
        counter_matrix: 2D numpy array
        mode: Operation mode (required)

    Returns:
        Formatted statistics string
    """
    if mode is None:
        raise ValueError("mode parameter is required")

    generator = StatisticsGenerator(mode)
    return generator.generate_statistics(counter_matrix)