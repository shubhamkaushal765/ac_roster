"""
Schedule optimization algorithms.

Implements beam search and scoring functions to optimize SOS officer
break schedule selection for smooth manning levels.
"""

from copy import deepcopy
from typing import List, Dict, Optional, Tuple
import numpy as np

from acroster.officer import SOSOfficer, MainOfficer
from acroster.config import NUM_SLOTS


class SegmentTree:
    """
    Helper class for computing schedule optimization scores.

    Tracks work count across all time slots and computes
    penalty and reward metrics for optimization.
    """

    def __init__(self, work_count: np.ndarray):
        """
        Initialize segment tree with work count array.

        Args:
            work_count: Array of worker count per time slot
        """
        self.work_count = work_count.copy()

    def update_delta(self, delta_indices: np.ndarray, delta: int):
        """
        Update work count at specific indices.

        Args:
            delta_indices: Indices to update
            delta: Value to add to work count
        """
        for i in delta_indices:
            self.work_count[i] += delta

    def compute_penalty(self) -> int:
        """
        Compute penalty as number of changes between consecutive slots.

        Lower penalty means smoother manning levels.

        Returns:
            Number of transitions between different manning levels
        """
        diffs = np.diff(self.work_count)
        return int(np.sum(diffs != 0))

    def compute_reward(self) -> int:
        """
        Compute reward based on maintaining high manning levels.

        Reward per slot = max(work_count) - work_count[t]
        Lower reward (closer to max) is better.

        Returns:
            Sum of gaps from maximum manning level
        """
        max_work_count = np.max(self.work_count)
        return int(np.sum(max_work_count - self.work_count))

    def compute_score(self, alpha: float = 1.0, beta: float = 1.0) -> float:
        """
        Compute combined optimization score.

        Score = alpha * penalty - beta * reward
        Lower score is better.

        Args:
            alpha: Weight for penalty term
            beta: Weight for reward term

        Returns:
            Combined score (lower is better)
        """
        penalty = self.compute_penalty()
        reward = self.compute_reward()
        return alpha * penalty - beta * reward


class ScheduleOptimizer:
    """
    Optimizes SOS officer break schedule selection using beam search.

    Selects the best combination of break schedules across all SOS officers
    to minimize manning fluctuations while maximizing coverage.
    """

    def __init__(
            self,
            beam_width: int = 50,
            alpha: float = 0.1,
            beta: float = 1.0
    ):
        """
        Initialize schedule optimizer.

        Args:
            beam_width: Number of top candidates to keep at each step
            alpha: Weight for smoothness penalty
            beta: Weight for coverage reward
        """
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta

    def optimize(
            self,
            sos_officers: List[SOSOfficer],
            main_officers: Optional[Dict[str, MainOfficer]] = None
    ) -> Tuple[List[int], np.ndarray, float]:
        """
        Select best break schedule for each SOS officer using beam search.

        Args:
            sos_officers: List of SOS officers with multiple break schedules
            main_officers: Optional dict of main officers for baseline count

        Returns:
            Tuple of:
                - List of selected schedule indices for each officer
                - Best work count array
                - Best score achieved
        """
        # Build initial work count
        initial_work_count = self._build_initial_work_count(
            sos_officers,
            main_officers
        )

        # Initialize beam with empty state
        initial_tree = SegmentTree(initial_work_count)
        initial_score = initial_tree.compute_score(self.alpha, self.beta)
        beam = [(initial_tree, initial_score, [])]

        # Iterate over each SOS officer
        for officer in sos_officers:
            beam = self._expand_beam_for_officer(officer, beam)

        # Get best result
        best_tree, best_score, chosen_indices = min(beam, key=lambda x: x[1])

        # Apply selected schedules to officers
        for i, officer in enumerate(sos_officers):
            if chosen_indices[i] is not None:
                officer.select_schedule(chosen_indices[i])

        return chosen_indices, best_tree.work_count, best_score

    def _build_initial_work_count(
            self,
            sos_officers: List[SOSOfficer],
            main_officers: Optional[Dict[str, MainOfficer]]
    ) -> np.ndarray:
        """
        Build initial work count array from all officers.

        Args:
            sos_officers: List of SOS officers
            main_officers: Optional dict of main officers

        Returns:
            Work count array (number of workers per slot)
        """
        work_count = np.zeros(NUM_SLOTS, dtype=int)

        # Add SOS officer availability
        for officer in sos_officers:
            work_count += officer.availability_schedule

        # Add main officer schedules
        if main_officers is not None:
            for officer in main_officers.values():
                work_count += np.where(officer.schedule != 0, 1, 0)

        return work_count

    def _expand_beam_for_officer(
            self,
            officer: SOSOfficer,
            beam: List[Tuple[SegmentTree, float, List]]
    ) -> List[Tuple[SegmentTree, float, List]]:
        """
        Expand beam by trying all break schedules for an officer.

        Args:
            officer: SOS officer to process
            beam: Current beam state

        Returns:
            Updated beam with top candidates
        """
        new_beam = []

        # Handle officers with no break schedules
        if len(officer.break_schedules) == 0:
            for tree, score, indices in beam:
                new_beam.append((tree, score, indices + [None]))
            return new_beam

        # Try each break schedule candidate
        for tree, score, indices in beam:
            for idx, candidate in enumerate(officer.break_schedules):
                # Find slots where breaks are taken
                delta_indices = np.where(
                    (officer.availability_schedule == 1) & (candidate == 0)
                )[0]

                # Create new tree with updated work count
                new_tree = deepcopy(tree)
                if len(delta_indices) > 0:
                    new_tree.update_delta(delta_indices, -1)

                new_score = new_tree.compute_score(self.alpha, self.beta)
                new_beam.append((new_tree, new_score, indices + [idx]))

        # Keep only top beam_width candidates
        new_beam = sorted(new_beam, key=lambda x: x[1])[:self.beam_width]

        return new_beam


def greedy_smooth_schedule_beam(
        sos_officers: List[SOSOfficer],
        main_officers: Optional[Dict[str, MainOfficer]],
        beam_width: int = 50,
        alpha: float = 0.1,
        beta: float = 1.0,
) -> Tuple[List[int], np.ndarray, float]:
    """
    Convenience function for backward compatibility.

    Select best break schedule for each SOS officer using beam search.

    Args:
        sos_officers: List of SOS officers
        main_officers: Optional dict of main officers
        beam_width: Number of candidates to keep
        alpha: Smoothness weight
        beta: Coverage weight

    Returns:
        Tuple of (chosen_indices, best_work_count, best_score)
    """
    optimizer = ScheduleOptimizer(beam_width, alpha, beta)
    return optimizer.optimize(sos_officers, main_officers)