import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from acroster.config import OperationMode as AcrosterOperationMode
from acroster.roster_builder import RosterBuilder, LastCounterAssigner
from acroster.sos_scheduler import SOSOfficerBuilder, BreakScheduleGenerator
from acroster.optimization import ScheduleOptimizer
from acroster.assignment_engine import (
    CounterAssignmentEngine,
    SOSAssignmentEngine,
    MatrixConverter
)
from acroster.statistics import StatisticsGenerator
from ..schemas.roster import OperationMode, OfficerCounts

logger = logging.getLogger(__name__)


class RosterGenerationService:
    def __init__(self):
        self.matrix_converter = MatrixConverter()

    async def generate_roster(
            self,
            mode: OperationMode,
            main_officers_reported: str,
            report_gl_counters: str,
            handwritten_counters: str,
            ot_counters: str,
            ro_ra_officers: str,
            sos_timings: str,
            beam_width: int = 20
    ) -> Tuple[Dict, OfficerCounts, float | None, Tuple[str, str]]:

        acroster_mode = AcrosterOperationMode.ARRIVAL if mode == OperationMode.ARRIVAL else AcrosterOperationMode.DEPARTURE

        roster_builder = RosterBuilder(acroster_mode)
        last_counter_assigner = LastCounterAssigner(acroster_mode)
        counter_engine = CounterAssignmentEngine(acroster_mode)
        sos_engine = SOSAssignmentEngine(acroster_mode)
        stats_generator = StatisticsGenerator(acroster_mode)

        logger.info(
            f"Starting roster generation with mode={mode.value}, beam_width={beam_width}"
        )

        main_officers, reported_officers, valid_ro_ra = roster_builder.build_main_officers(
            main_officers_reported,
            report_gl_counters,
            ro_ra_officers
        )

        main_officers = roster_builder.apply_takeover_counters(
            main_officers,
            handwritten_counters
        )

        logger.info(f"Built {len(main_officers)} main officers")

        counter_matrix_wo_last = counter_engine.officers_to_counter_matrix(
            main_officers
        )

        officer_last_counter = last_counter_assigner.get_last_counter_slots(
            reported_officers,
            valid_ro_ra
        )

        empty_counters = last_counter_assigner.find_empty_counters_from_slot(
            counter_matrix_wo_last,
            from_slot=42
        )

        main_officers = last_counter_assigner.assign_last_counters(
            main_officers,
            officer_last_counter,
            empty_counters
        )

        counter_matrix = counter_engine.officers_to_counter_matrix(
            main_officers
        )
        counter_matrix_with_ot, ot_officers = counter_engine.assign_ot_officers(
            counter_matrix,
            ot_counters
        )

        logger.info(f"Added {len(ot_officers)} OT officers")

        stats1 = stats_generator.generate_statistics(
            counter_matrix_with_ot.to_matrix()
        )

        optimization_penalty = None

        if sos_timings and sos_timings.strip():
            logger.info("Processing SOS officers")

            sos_builder = SOSOfficerBuilder()
            break_generator = BreakScheduleGenerator()
            optimizer = ScheduleOptimizer(beam_width=beam_width)

            sos_officers, pre_assigned_counter_dict = sos_builder.build_sos_officers(
                sos_timings
            )
            sos_officers = break_generator.generate_break_schedules(
                sos_officers
            )

            logger.info(f"Built {len(sos_officers)} SOS officers")

            chosen_indices, best_work_count, optimization_penalty = optimizer.optimize(
                sos_officers,
                main_officers
            )

            logger.info(f"Optimization penalty: {optimization_penalty}")

            schedule_intervals_to_officers = {}
            for officer in sos_officers:
                intervals = officer.get_working_intervals()
                for interval in intervals:
                    if interval not in schedule_intervals_to_officers:
                        schedule_intervals_to_officers[interval] = []
                    schedule_intervals_to_officers[interval].append(
                        officer.officer_id - 1
                    )

            sos_counter_matrix = sos_engine.assign_sos_officers(
                pre_assigned_counter_dict,
                schedule_intervals_to_officers,
                counter_matrix_with_ot
            )

            main_matrix = counter_matrix_with_ot.to_matrix()
            sos_matrix = sos_counter_matrix.to_matrix()

            final_matrix = self.matrix_converter.merge_prefixed_matrices(
                main_matrix,
                sos_matrix
            )

            officer_schedule = self.matrix_converter.counter_to_officer_schedule(
                final_matrix
            )

            stats2 = stats_generator.generate_statistics(final_matrix)

            logger.info(
                f"Generated roster with {len(officer_schedule)} total officers"
            )
        else:
            logger.info("No SOS officers, using main officers only")
            officer_schedule = {k: v.schedule.tolist() for k, v in
                                main_officers.items()}
            final_matrix = counter_matrix_with_ot.to_matrix()
            stats2 = stats1

        officer_counts = self._count_officers(officer_schedule)

        roster_data = {
            "officer_schedules": officer_schedule,
            "counter_matrix":    final_matrix.tolist(),
            "mode":              mode.value
        }

        return roster_data, officer_counts, optimization_penalty, (
            stats1, stats2)

    def _count_officers(self, officer_schedule: Dict) -> OfficerCounts:
        main_count = sum(
            1 for k in officer_schedule.keys() if k.startswith('M')
        )
        sos_count = sum(
            1 for k in officer_schedule.keys() if k.startswith('S')
        )
        ot_count = sum(
            1 for k in officer_schedule.keys() if k.startswith('OT')
        )

        return OfficerCounts(
            main=main_count,
            sos=sos_count,
            ot=ot_count,
            total=main_count + sos_count + ot_count
        )


roster_generation_service = RosterGenerationService()
