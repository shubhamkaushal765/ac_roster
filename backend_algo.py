import re
from collections import defaultdict
from copy import deepcopy
from acroster import Plotter
from acroster import Officer, MainOfficer, OTOfficer, SOSOfficer
from typing import Dict, List, Tuple, Optional

import numpy as np

NUM_SLOTS = 48
NUM_COUNTERS = 41
START_HOUR = 10
counter_priority_list = [41] + [
    n for offset in range(0, 10) for n in range(40 - offset, 0, -10)
]


def hhmm_to_slot(hhmm: str) -> int:
    """Convert hhmm string to a slot index (0–47)."""
    t = int(hhmm)
    h = t // 100
    m = t % 100
    slot = (h - START_HOUR) * 4 + (m // 15)
    return max(0, min(NUM_SLOTS - 1, slot))


def slot_to_hhmm(slot: int) -> str:
    """Convert slot index back to hhmm string."""
    h = START_HOUR + slot // 4
    m = (slot % 4) * 15
    return f"{h:02d}{m:02d}"


# Initialize counters as a dict (keys 1–41)
counters = {i: [0] * NUM_SLOTS for i in range(1, 42)}


# Function to get a specific counter
def get_counter(counter_number: int):
    if counter_number in counters:
        return counters[counter_number]
    else:
        raise ValueError("Counter number must be between 1 and 40")


# # Update an interval
# counters[5][10] = 1
# print(get_counter(5)[10])  # 1


def add_4main_roster(full_counters):
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
            + [0]
            + [0] * 6
    )
    d = (
            [0] * 5
            + [0] * 1
            + [full_counters[0]] * 6
            + [0] * 2
            + [full_counters[1]] * 10
            + [0] * 3
            + [full_counters[2]] * 9
            + [0] * 3
            + [full_counters[0]] * 9
    )
    return (a, b, c, d)


def init_main_officers_template(main_total=24, exclude_main: list = None) -> \
        Dict[int, np.ndarray]:
    """Generate roster templates for main officers (unchanged, just returns templates)"""
    main_officers = {}
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
            + [0]
            + [0] * 6
    )
    main_officers[4] = (
            [0] * 5
            + [0] * 1
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
            + [0]
            + [0] * 6
    )
    main_officers[8] = (
            [0] * 5
            + [0] * 1
            + [41] * 6
            + [0] * 2
            + [9] * 10
            + [0] * 3
            + [29] * 9
            + [0] * 3
            + [40] * 9
    )

    # Define groups of officers and their rosters
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

    # Loop through each group and assign
    for m_no, roster in groups:
        results = add_4main_roster(roster)
        for i, officer in enumerate(m_no):
            main_officers[officer] = results[i]

    # Convert to numpy arrays
    main_officers = {i: np.array(v) for i, v in main_officers.items()}
    return main_officers


def generate_main_officers_schedule(
        main_officers_template: Dict[int, np.ndarray],
        main_officers_reported: str,
        report_gl_counters: str,
        main_officers_report_late_or_leave_early: str,
) -> Tuple[Dict[str, MainOfficer], set, List[Tuple]]:
    """
    Generate MainOfficer objects with schedules

    Returns:
        - Dict of MainOfficer objects keyed by officer_key (e.g., 'M1')
        - Set of reported officer IDs
        - List of valid adjustments
    """
    # Parse which officers reported
    reported_officers = set()
    parts = main_officers_reported.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            for i in range(int(start), int(end) + 1):
                reported_officers.add(i)
        else:
            reported_officers.add(int(part))

    # Validation function
    def validate_adjustments(input_str):
        valid_entries = []
        if not input_str.strip():
            return valid_entries

        entries = input_str.split(",")
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            if not re.match(r"^\d+(RA|RO)\d{4}$", entry):
                print(f"⚠️ Skipping {entry}: invalid format")
                continue

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

            if officer_id not in reported_officers:
                print(
                    f"⚠️ Skipping {entry}: officer {officer_id} not in reported list."
                )
                continue

            h = int(hhmm[:2])
            m = int(hhmm[2:])
            if not (10 <= h <= 22):
                print(
                    f"⚠️ Skipping {entry}: hour {h} out of range (1000–2200)"
                )
                continue
            if m not in (0, 15, 30, 45):
                print(
                    f"⚠️ Skipping {entry}: minutes {m} must be 00, 15, 30, or 45"
                )
                continue
            if h == 22 and m > 0:
                print(f"⚠️ Skipping {entry}: must not exceed 2200")
                continue

            valid_entries.append((officer_id, adj_type, hhmm))
        return valid_entries

    # Validate and parse adjustments
    valid_adjustments = validate_adjustments(
        main_officers_report_late_or_leave_early
    )
    adjustments = {}
    for officer_id, adj_type, hhmm in valid_adjustments:
        slot = hhmm_to_slot(hhmm)
        adjustments[officer_id] = (adj_type, slot)

    # Build MainOfficer objects
    main_officers: Dict[str, MainOfficer] = {}
    for officer_id in reported_officers:
        if officer_id not in main_officers_template:
            continue

        # Create MainOfficer with template
        officer = MainOfficer(
            officer_id=officer_id,
            roster_template=main_officers_template[officer_id]
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
    report_list = [s.strip() for s in report_gl_counters.split(",") if
                   s.strip()]
    for officer_counter in report_list:
        if "AC" in officer_counter:
            idx = officer_counter.index("AC")
            officer_id = int(officer_counter[:idx])
            counter_no = int(officer_counter[idx + 2:])

            officer_key = f"M{officer_id}"
            if officer_key in main_officers:
                main_officers[officer_key].apply_ground_level_counter(
                    counter_no
                )

    return main_officers, reported_officers, valid_adjustments


def get_officer_last_counter_and_empty_counters(
        reported_officers: set,
        ro_ra_officers: List[Tuple],
        counter_matrix: np.ndarray
) -> Tuple[Dict[int, int], List[int]]:
    """
    Compute each officer's last counter start slot (for officers with id % 4 == 3)
    and find counters that are empty from slot 42 onward.
    """
    officer_last_counter = {}

    # Step 1: Assign last counter start slots for eligible officers
    for officer_id in reported_officers:
        if officer_id % 4 == 3:
            found = False
            for ro_ra_officer in ro_ra_officers:
                if ro_ra_officer[0] == officer_id and ro_ra_officer[1] == "RO":
                    last_counter_end_slot = hhmm_to_slot(ro_ra_officer[2])
                    found = True
                    break
            if not found:
                last_counter_end_slot = 48
            officer_last_counter[officer_id] = last_counter_end_slot

    # Step 2: Identify counters empty from slot 42 onwards
    empty_counters_2030 = [
        row_idx
        for row_idx in range(counter_matrix.shape[0])
        if np.all(counter_matrix[row_idx, 42:] == 0)
    ]

    # Step 3: Sort counters by priority order
    empty_counters_2030.sort(
        key=lambda x: counter_priority_list.index(x + 1)
        if (x + 1) in counter_priority_list
        else float("inf")
    )
    return officer_last_counter, empty_counters_2030


def update_main_officers_schedule_last_counter(
        main_officers: Dict[str, MainOfficer],
        officer_last_counter: Dict[int, int],
        empty_counters_2030: List[int]
) -> Dict[str, MainOfficer]:
    """
    Apply last counter assignments to main officers
    """
    empty_counters_copy = empty_counters_2030.copy()

    for officer_key, officer in main_officers.items():
        if officer.officer_id in officer_last_counter:
            last_slot = officer_last_counter[officer.officer_id]
            if last_slot >= 42 and empty_counters_copy:
                counter_no = empty_counters_copy[
                                 0] + 1  # Convert 0-indexed to 1-indexed
                officer.apply_last_counter(last_slot, counter_no)
                empty_counters_copy.pop(0)

    return main_officers


def add_takeover_ot_ctr(
        main_officers: Dict[str, MainOfficer],
        handwritten_counters: str
) -> Dict[str, MainOfficer]:
    """
    Apply takeover counters to main officers
    """
    if not handwritten_counters.strip():
        return main_officers

    pairs = re.findall(r"(\d+)\s*[aA]\s*[cC]\s*(\d+)", handwritten_counters)

    for officer_str, counter_str in pairs:
        officer_key = f"M{officer_str}"
        new_counter = int(counter_str)

        if officer_key in main_officers:
            main_officers[officer_key].apply_takeover_counter(new_counter)

    return main_officers


def build_officer_schedules(user_input: str) -> Tuple[
    List[SOSOfficer], Dict[int, int]]:
    """
    Build SOSOfficer objects from user input

    Returns:
        - List of SOSOfficer objects
        - Dict of pre-assigned counters {officer_index: counter_no}
    """
    input_avail = convert_input(user_input)
    sos_officers: List[SOSOfficer] = []
    pre_assigned_counter_dict = {}

    for idx, avail in enumerate(input_avail):
        # Search for pre-assigned counters like (AC4)
        matches = re.findall(r"\(AC(\d+)\)", avail, flags=re.IGNORECASE)
        if matches:
            pre_assigned_counter_dict[idx] = int(matches[0])
            avail = re.sub(r"\(AC\d+\)", "", avail, flags=re.IGNORECASE)

        availability_schedule = parse_availability(avail)
        officer = SOSOfficer(
            officer_id=idx + 1,
            availability_schedule=availability_schedule
        )

        if idx in pre_assigned_counter_dict:
            officer.set_pre_assigned_counter(pre_assigned_counter_dict[idx])

        sos_officers.append(officer)

    return sos_officers, pre_assigned_counter_dict


def generate_break_schedules(sos_officers: List[SOSOfficer]) -> List[
    SOSOfficer]:
    """
    Generate break schedules for all SOS officers and store them in the officer objects
    """

    def sliding_window_ok(schedule):
        consec = 0
        for x in schedule:
            if x == 1:
                consec += 1
                if consec > 10:
                    return False
            else:
                consec = 0
        return True

    for officer in sos_officers:
        base = officer.availability_schedule.copy()
        work_slots = np.where(base == 1)[0]

        if len(work_slots) == 0:
            officer.add_break_schedule(base.copy())
            continue

        # Build original consecutive working stretches
        stretches = []
        cur = [work_slots[0]]
        for s in work_slots[1:]:
            if s == cur[-1] + 1:
                cur.append(s)
            else:
                stretches.append(cur)
                cur = [s]
        stretches.append(cur)

        # If all stretches ≤10, store schedule as valid directly
        if all(len(stretch) <= 10 for stretch in stretches):
            officer.add_break_schedule(base.copy())
            continue

        valid_schedules = []
        seen_schedules = set()

        def finalize_schedule(schedule, last_break_end, last_break_len):
            if sliding_window_ok(schedule):
                sig = schedule.tobytes()
                if sig not in seen_schedules:
                    seen_schedules.add(sig)
                    valid_schedules.append(schedule)
                return

            # Try inserting a 1-slot break
            for s in range(len(schedule)):
                if schedule[s] != 1:
                    continue

                next_break_index = s
                while next_break_index < len(schedule) and schedule[
                    next_break_index] == 1:
                    next_break_index += 1
                interval_end = next_break_index - 1

                prev_break_index = s
                while prev_break_index >= 0 and schedule[
                    prev_break_index] == 1:
                    prev_break_index -= 1
                interval_start = prev_break_index + 1

                if s <= interval_start + 4 or s >= interval_end - 4:
                    continue

                required_gap = min(
                    2 * last_break_len, 4
                ) if last_break_end >= 0 else 0
                if s - last_break_end - 1 < required_gap:
                    continue

                cand = schedule.copy()
                cand[s] = 0
                if sliding_window_ok(cand):
                    sig = cand.tobytes()
                    if sig not in seen_schedules:
                        seen_schedules.add(sig)
                        valid_schedules.append(cand)

        def place_breaks(
                schedule, stretch_idx=0, last_break_end=-1, last_break_len=0
        ):
            if stretch_idx >= len(stretches):
                finalize_schedule(schedule, last_break_end, last_break_len)
                return

            stretch = stretches[stretch_idx]
            min_slot, max_slot = stretch[0], stretch[-1]
            stretch_len = len(stretch)

            if stretch_len <= 10:
                place_breaks(
                    schedule, stretch_idx + 1, last_break_end, last_break_len
                )
                return

            if stretch_len >= 36:
                pattern = [2, 3, 3]
            elif stretch_len >= 20:
                pattern = [2, 3]
            elif stretch_len >= 10:
                pattern = [2]
            else:
                pattern = [0]

            def recurse(schedule, blens, last_break_end, last_break_len):
                if not blens:
                    place_breaks(
                        schedule, stretch_idx + 1, last_break_end,
                        last_break_len
                    )
                    return

                blen = blens[0]
                interval_start = min_slot
                if last_break_end >= 0:
                    interval_start = last_break_end + 1

                max_consec_start = interval_start + 10
                max_allowed = min(max_consec_start, max_slot - blen - 3)

                for s in range(interval_start + 4, max_allowed + 1):
                    required_gap = min(
                        2 * last_break_len, 4
                    ) if last_break_end >= 0 else 0
                    if s - last_break_end - 1 < required_gap:
                        continue

                    if not np.all(schedule[s: s + blen] == 1):
                        continue

                    new_sched = schedule.copy()
                    new_sched[s: s + blen] = 0
                    recurse(new_sched, blens[1:], s + blen, blen)

            recurse(schedule, pattern, last_break_end, last_break_len)

        place_breaks(base.copy())

        if valid_schedules:
            for sched in valid_schedules:
                officer.add_break_schedule(sched)
        else:
            officer.add_break_schedule(base.copy())

    return sos_officers


def greedy_smooth_schedule_beam(
        sos_officers: List[SOSOfficer],
        main_officers: Optional[Dict[str, MainOfficer]],
        beam_width: int = 50,
        alpha: float = 0.1,
        beta: float = 1.0,
) -> Tuple[List[int], np.ndarray, float]:
    """
    Select best break schedule for each SOS officer using beam search

    Returns:
        - List of selected schedule indices for each officer
        - Best work count array
        - Best score
    """
    # Build initial work count
    initial_work_count = np.zeros(NUM_SLOTS, dtype=int)
    for officer in sos_officers:
        initial_work_count += officer.availability_schedule

    if main_officers is not None:
        for officer in main_officers.values():
            initial_work_count += np.where(officer.schedule != 0, 1, 0)

    # Initialize beam
    stree = SegmentTree(initial_work_count)
    beam = [(stree, stree.compute_score(alpha, beta), [])]

    # Iterate over each SOS officer
    for officer in sos_officers:
        new_beam = []

        if len(officer.break_schedules) == 0:
            for stree, score, indices in beam:
                new_beam.append((stree, score, indices + [None]))
            beam = new_beam
            continue

        for stree, score, indices in beam:
            for idx, candidate in enumerate(officer.break_schedules):
                delta_indices = np.where(
                    (officer.availability_schedule == 1) & (candidate == 0)
                )[0]
                new_stree = deepcopy(stree)
                if len(delta_indices) > 0:
                    new_stree.update_delta(delta_indices, -1)
                new_score = new_stree.compute_score(alpha, beta)
                new_beam.append((new_stree, new_score, indices + [idx]))

        beam = sorted(new_beam, key=lambda x: x[1])[:beam_width]

    best_stree, best_score, chosen_indices = min(beam, key=lambda x: x[1])

    # Apply selected schedules to officers
    for i, officer in enumerate(sos_officers):
        if chosen_indices[i] is not None:
            officer.select_schedule(chosen_indices[i])

    return chosen_indices, best_stree.work_count, best_score


def officer_to_counter_matrix(officers: Dict[str, Officer]) -> np.ndarray:
    """
    Convert officer schedules to counter matrix

    Args:
        officers: Dict of Officer objects keyed by officer_key

    Returns:
        counter_matrix: (41, 48) array where each row is a counter
    """
    counter_matrix = np.zeros((NUM_COUNTERS, NUM_SLOTS), dtype=object)

    for officer_key, officer in officers.items():
        for slot, counter in enumerate(officer.schedule):
            if counter != 0:
                counter_matrix[counter - 1, slot] = officer_key

    return counter_matrix


def add_ot_counters(
        counter_matrix: np.ndarray,
        OT_counters: str
) -> Tuple[np.ndarray, List[OTOfficer]]:
    """
    Add OT officers to counter matrix

    Returns:
        - Updated counter matrix
        - List of OTOfficer objects created
    """
    if len(OT_counters) == 0:
        return counter_matrix, []

    counter_matrix_w_OT = counter_matrix.copy().astype(object)
    OT_list = [int(x.strip()) for x in OT_counters.split(",") if x.strip()]

    ot_officers = []
    for i, OT_counter in enumerate(OT_list):
        ot_officer = OTOfficer(officer_id=i + 1, counter_no=OT_counter)
        ot_officers.append(ot_officer)
        counter_matrix_w_OT[OT_counter - 1, 0:2] = [ot_officer.officer_key,
                                                    ot_officer.officer_key]

    counter_matrix_w_OT = counter_matrix_w_OT.astype(object)
    counter_matrix_w_OT[counter_matrix_w_OT == 0] = "0"

    return counter_matrix_w_OT, ot_officers


def counter_to_officer_schedule(counter_matrix: np.ndarray) -> Dict[
    str, List[int]]:
    """
    Convert counter_matrix back to officer schedules

    Returns:
        Dict where keys are officer keys (e.g., 'M1', 'S2') and values are counter assignments per slot
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
    sorted_schedule = {k: officer_schedule[k] for k in sorted_keys}

    return sorted_schedule


# Update run_algo function to use new OOP structure
def run_algo(
        main_officers_reported: str,
        report_gl_counters: str,
        sos_timings: str,
        ro_ra_officers: str,
        handwritten_counters: str,
        OT_counters: str,
):
    """Main algorithm using OOP structure"""

    # Generate main officers
    main_officers_template = init_main_officers_template()
    main_officers, reported_officers, valid_ro_ra = generate_main_officers_schedule(
        main_officers_template,
        main_officers_reported,
        report_gl_counters,
        ro_ra_officers,
    )

    # Convert to counter matrix to find empty counters
    counter_matrix_wo_last = officer_to_counter_matrix(main_officers)
    officer_last_counter, empty_counters_2030 = get_officer_last_counter_and_empty_counters(
        reported_officers, valid_ro_ra, counter_matrix_wo_last
    )

    # Apply last counters
    main_officers = update_main_officers_schedule_last_counter(
        main_officers, officer_last_counter, empty_counters_2030
    )

    # Apply takeover counters
    main_officers = add_takeover_ot_ctr(main_officers, handwritten_counters)

    # Convert to counter matrix
    counter_matrix = officer_to_counter_matrix(main_officers)

    # Add OT officers
    main_counter_matrix_w_OT, ot_officers = add_ot_counters(
        counter_matrix, OT_counters
    )
    stats1 = generate_statistics(main_counter_matrix_w_OT)

    if len(sos_timings) > 0:
        # Build SOS officers
        sos_officers, pre_assigned_counter_dict = build_officer_schedules(
            sos_timings
        )

        # Generate break schedules
        sos_officers = generate_break_schedules(sos_officers)

        # Optimize schedule selection
        chosen_schedule_indices, best_work_count, min_penalty = greedy_smooth_schedule_beam(
            sos_officers, main_officers, beam_width=20
        )
        print(f"Optimization penalty: {min_penalty}")

        # Get working intervals for SOS officers
        schedule_intervals_to_officers = {}
        for officer in sos_officers:
            intervals = officer.get_working_intervals()
            for interval in intervals:
                if interval not in schedule_intervals_to_officers:
                    schedule_intervals_to_officers[interval] = []
                schedule_intervals_to_officers[interval].append(
                    officer.officer_id - 1
                )

        print("=== best work count ===")
        print(best_work_count)
        print("===schedule_intervals_to_officers===")
        print(schedule_intervals_to_officers)

        # Add SOS officers to counter matrix
        sos_counter_matrix = add_sos_officers(
            pre_assigned_counter_dict,
            schedule_intervals_to_officers,
            main_counter_matrix_w_OT,
        )

        final_counter_matrix = merge_prefixed_matrices(
            sos_counter_matrix, main_counter_matrix_w_OT
        )

        # Convert back to officer schedule format
        officer_schedule = counter_to_officer_schedule(final_counter_matrix)

        stats2 = generate_statistics(final_counter_matrix)
        return (
            main_counter_matrix_w_OT,
            final_counter_matrix,
            officer_schedule,
            [stats1, stats2],
        )
    else:
        # Convert main officers to schedule dict format
        officer_schedule = {k: v.schedule.tolist() for k, v in
                            main_officers.items()}
        return (
            main_counter_matrix_w_OT,
            main_counter_matrix_w_OT,
            officer_schedule,
            [stats1, stats1],
        )


def find_partial_availability(counter_matrix):
    """
    Find intervals of 0 in each row of the counter matrix if row has less than 48 zeros.
    Only include rows with at least one zero interval.
    """
    counter_w_partial_availability = {}
    num_rows, num_cols = counter_matrix.shape

    for row_idx in range(num_rows):
        row = counter_matrix[row_idx]
        zero_indices = np.where(row == 0)[0]

        if len(zero_indices) < num_cols and len(zero_indices) > 0:
            intervals = []
            start = int(zero_indices[0])
            prev = int(zero_indices[0])

            for idx in zero_indices[1:]:
                idx = int(idx)
                if idx == prev + 1:
                    prev = idx
                else:
                    intervals.append((start, prev))
                    start = idx
                    prev = idx
            intervals.append((start, prev))  # add the last interval

            if intervals:  # only include non-empty intervals
                counter_w_partial_availability[row_idx + 1] = intervals

    return counter_w_partial_availability


def find_consecutive_intervals(intervals, selected_index):
    """
    Find a consecutive path of intervals from start_index to end_index using DFS.
    Intervals are consecutive if next.start > prev.end (no overlap).

    Parameters:
    -----------
    intervals : list of tuples (int, int)
    selected_index = (start_index : int, end_index : int)

    Returns:
    --------
    path : list of tuples or None
        Consecutive intervals from start_index to end_index
    unused_intervals : list of tuples
        Intervals not used in the path (duplicates handled correctly)
    """
    start_index, end_index = selected_index
    n = len(intervals)
    used = [False] * n
    path_result = None
    used_indices = []

    def dfs(current_index, path, used_indices_local):
        nonlocal path_result, used_indices
        if path_result is not None:
            return  # already found

        if current_index >= end_index:
            path_result = path[:]
            used_indices = used_indices_local[:]
            return

        for i, (s, e) in enumerate(intervals):
            if not used[i] and s == current_index + 1:
                used[i] = True
                dfs(e, path + [(s, e)], used_indices_local + [i])
                used[i] = False

    dfs(start_index - 1, [], [])

    if path_result is None:
        return None, intervals

    # Verify the last interval reaches exactly end_index
    if path_result and path_result[-1][1] != end_index:
        return None, intervals

    # Compute unused intervals by index
    unused_intervals = [iv for i, iv in enumerate(intervals) if
                        i not in used_indices]

    return path_result, unused_intervals


def parse_availability(avail_str: str) -> np.ndarray:
    """
    Convert availability string (e.g., '1000-1200,2030-2200')
    into a binary numpy array of length NUM_SLOTS.
    """
    schedule = np.zeros(NUM_SLOTS, dtype=int)

    for rng in avail_str.split(","):
        start, end = rng.split("-")
        start_slot = hhmm_to_slot(start)
        end_slot = hhmm_to_slot(end)

        # Fill working slots; end is inclusive now
        schedule[start_slot: end_slot + 1] = 1

    return schedule


def convert_input(user_input: str):
    # Split by main delimiter (comma) → separates elements
    raw_elements = [elem.strip() for elem in user_input.split(",")]

    result = []
    for elem in raw_elements:
        # Replace ';' with ', ' only inside each element
        if ";" in elem:
            elem = ", ".join([part.strip() for part in elem.split(";")])
        result.append(elem)

    return result


class SegmentTree:
    def __init__(self, work_count):
        self.work_count = work_count.copy()

    def update_delta(self, delta_indices, delta):
        for i in delta_indices:
            self.work_count[i] += delta

    def compute_penalty(self):
        """Penalty = number of changes between consecutive slots."""
        diffs = np.diff(self.work_count)
        return int(np.sum(diffs != 0))

    def compute_reward(self):
        """Reward per slot = max(work_count) - work_count[t]."""
        max_work_count = np.max(self.work_count)
        return int(np.sum(max_work_count - self.work_count))

    def compute_score(self, alpha=1.0, beta=1.0):
        """Combined score = alpha * penalty - beta * reward (lower is better)."""
        penalty = self.compute_penalty()
        reward = self.compute_reward()
        return alpha * penalty - beta * reward


def generate_sos_schedule_matrix(
        saved_indices, all_break_schedules, officer_names
):
    """
    Generate a 2D matrix of officers' schedules based on selected indices.

    Args:
        saved_indices (list of int): Selected schedule index per officer.
        all_break_schedules (dict): officer -> list of 0/1 np.arrays (schedules).
        officer_names (list of str): List of officer names in the same order.

    Returns:
        np.ndarray: 2D array (num_officers x num_slots), 1=working, 0=break
    """
    num_officers = len(saved_indices)
    num_slots = len(
        next(iter(all_break_schedules[officer_names[0]]))
    )  # assume all schedules same length

    sos_schedule_matrix = np.zeros((num_officers, num_slots), dtype=object)
    sos_schedule_matrix[sos_schedule_matrix == 0] = "0"

    for i, officer in enumerate(officer_names):
        idx = saved_indices[i]
        sos_schedule_matrix[i] = all_break_schedules[officer][idx]

    return sos_schedule_matrix


# sos_schedule_matrix = generate_sos_schedule_matrix(chosen_schedule_indices, all_break_schedules, officer_names)


def get_intervals_from_schedule(sos_schedule_matrix):
    interval_dict = defaultdict(list)
    sos_schedule_matrix = np.array(
        sos_schedule_matrix
    )  # Ensure it's a NumPy array

    for row_idx, row in enumerate(sos_schedule_matrix):
        n = len(row)
        i = 0
        while i < n:
            if row[i] == 1:
                start = i
                # Find the end of consecutive 1's
                while i < n and row[i] == 1:
                    i += 1
                end = i - 1  # end index is inclusive
                interval_dict[(start, end)].append(row_idx)
            else:
                i += 1

    schedule_intervals = []
    for interval, rows in interval_dict.items():
        schedule_intervals.extend([interval] * len(rows))
    return dict(interval_dict), schedule_intervals


def split_full_partial_paths(paths, target_length=48):
    """
    Splits a list of paths into full paths (covering target_length exactly)
    and partial paths (covering less than target_length).

    Args:
        paths (list of list of tuples): Each path is a list of intervals (start, end).
        target_length (int): The length to consider a path as full.

    Returns:
        tuple: (full_paths, partial_paths)
    """
    full_paths = []
    partial_paths = []

    for path in paths:
        total_length = sum(end - start + 1 for start, end in path)
        if total_length == target_length:
            full_paths.append(path)
        else:
            partial_paths.append(path)

    return full_paths, partial_paths


def prefix_non_zero(counter_matrix, prefix):
    # Create an empty array of same shape, dtype=object to hold strings
    result = np.empty(counter_matrix.shape, dtype=object)

    # Fill zeros as string "0"
    result[counter_matrix == 0] = "0"

    # Fill non-zero elements with "M" prefix
    result[counter_matrix != 0] = [
        prefix + str(x) for x in counter_matrix[counter_matrix != 0]
    ]

    return result


def add_sos_officers(
        pre_assigned_counter_dict, schedule_intervals_to_officers,
        main_counter_matrix
):
    """
    Assign officers to counters using gap-aware greedy interval packing.

    Args:
        schedule_intervals_to_officers: dict {(start,end): [officer_ids]}
        main_counter_matrix: np.array of shape (total_counters, total_slots)


    Returns:
        sos_counter_matrix: np.array of shape (total_counters, total_slots)
                        each element is officer_id or '0'
    """
    # Create duplicate matrix
    sos_main_counter_matrix = main_counter_matrix.copy()
    sos_counter_matrix = np.full((NUM_COUNTERS, NUM_SLOTS), "0", dtype=object)

    # Priority list for step 3 (1-indexed: 1 to 41)
    counter_priority_list = [41] + [
        n for offset in range(0, 10) for n in range(40 - offset, 0, -10)
    ]

    # Sort intervals by start time, then by end time
    sorted_intervals = sorted(
        schedule_intervals_to_officers.items(),
        key=lambda x: (x[0][0], x[0][1])
    )

    # Helper: check if interval is empty in a counter
    def is_interval_empty(counter_id, start, end):
        """Check if all slots in [start, end] are '0' in sos_main_counter_matrix"""
        return np.all(
            sos_main_counter_matrix[counter_id - 1, start: end + 1] == "0"
        )

    # Helper: check if interval connects to existing assignments
    def is_connected(counter_id, start, end):
        """Check if interval connects to previous or next non-zero slot"""
        # Check connection to previous slot
        if start > 0 and sos_counter_matrix[counter_id - 1, start - 1] != "0":
            return True
        # Check connection to next slot
        if end < NUM_SLOTS - 1 and sos_counter_matrix[
            counter_id - 1, end + 1] != "0":
            return True
        return False

    # Helper: get partial empty counters based on current state
    def get_partial_empty_counters():
        """Get counters that are partially empty (at least 1 non-zero and at least 1 zero)"""
        partial_counters = []
        for counter_id in range(1, NUM_COUNTERS + 1):
            row = sos_main_counter_matrix[counter_id - 1]
            # Counter is partially empty if at least 1 element is non-zero and at least 1 is zero
            if np.any(row != "0") and np.any(row == "0"):
                partial_counters.append(counter_id)
        return partial_counters

    # Helper: get counters with SOS assignments (not partial, not empty)
    def get_used_sos_counters(partial_counters):
        """Get counters that have SOS assignments but aren't partial or empty"""
        used_counters = []
        for counter_id in range(1, NUM_COUNTERS + 1):
            # Skip if partial
            if counter_id in partial_counters:
                continue
            # Check if this counter has any SOS assignments
            if np.any(sos_counter_matrix[counter_id - 1] != "0"):
                used_counters.append(counter_id)
        return used_counters

    print("++++++++++++++++++++sorted intervals++++++")
    print(sorted_intervals)
    for interval, officer_ids in sorted_intervals:
        start, end = interval

        for officer_id in officer_ids:
            best_counter = None
            best_score = -1

            # Step 0: check if first counter is already pre-assigned
            if (
                    len(pre_assigned_counter_dict) > 0
                    and officer_id in pre_assigned_counter_dict
                    and start == 0
            ):
                best_counter = pre_assigned_counter_dict[officer_id]

            # Get current partial counters
            partial_counters = get_partial_empty_counters()

            # Step 1: try partial counters first (must be CONNECTED to previous or next)
            if best_counter is None:
                for counter_id in partial_counters:
                    if is_interval_empty(counter_id, start, end):
                        if is_connected(counter_id, start, end):
                            score = 100  # Connected to existing assignment
                            if score > best_score:
                                best_score = score
                                best_counter = counter_id

            # Step 2: try already used SOS counters (must be CONNECTED)
            if best_counter is None:
                used_sos_counters = get_used_sos_counters(partial_counters)
                for counter_id in used_sos_counters:
                    if is_interval_empty(counter_id, start, end):
                        if is_connected(counter_id, start, end):
                            score = 100
                            if score > best_score:
                                best_score = score
                                best_counter = counter_id

            # Step 3: iterate through counter_priority_list in order
            if best_counter is None:
                for priority_counter in counter_priority_list:
                    if is_interval_empty(priority_counter, start, end):
                        best_counter = priority_counter
                        break

                if best_counter is None:
                    print(
                        f"ERROR: No available counter for officer {officer_id}, interval {interval}"
                    )
                    continue

            # Assign officer to counter
            sos_counter_matrix[best_counter - 1,
            start: end + 1] = f"S{officer_id + 1}"
            sos_main_counter_matrix[best_counter - 1, start: end + 1] = (
                f"S{officer_id + 1}"
            )

    # Print statistics
    final_partial = get_partial_empty_counters()
    used_counters = sum(
        1
        for c in range(1, NUM_COUNTERS + 1)
        if np.any(sos_counter_matrix[c - 1] != "0")
    )

    print("\nAssignment Statistics:")
    print(f"  Partial counters at end: {len(final_partial)}")
    print(f"  Total counters used for SOS: {used_counters}/{NUM_COUNTERS}")

    return sos_counter_matrix


def merge_prefixed_matrices(counter_matrix, sos_matrix):
    """
    Merge two prefixed matrices of the same shape, keeping non-zero entries from sos_matrix
    and filling the rest from counter_matrix.

    Args:
        counter_matrix (np.ndarray): Original counter matrix (2D array of strings)
        sos_matrix (np.ndarray): SOS counter manning matrix (2D array of strings)

    Returns:
        np.ndarray: Merged matrix of the same shape
    """
    # Ensure both matrices have the same shape
    if counter_matrix.shape != sos_matrix.shape:
        raise ValueError("Both matrices must have the same shape")

    # Merge: take sos_matrix where it's non-zero, else take counter_matrix
    merged_matrix = np.where(sos_matrix != "0", sos_matrix, counter_matrix)

    return merged_matrix


def generate_statistics(counter_matrix: np.ndarray):
    statistics_list = []
    counter_matrix = np.array(counter_matrix)  # ensure numpy array
    num_rows, num_slots = counter_matrix.shape

    # define row groups for second line
    row_groups = [range(0, 10), range(10, 20), range(20, 30), range(30, 40)]

    for slot in range(num_slots):
        # First line: sum of rows 0-39 not '0' + '/' + sum of remaining rows not '0'
        count1 = np.sum(counter_matrix[0:40, slot] != "0")
        count2 = np.sum(counter_matrix[40:, slot] != "0")
        first_line = f"{slot_to_hhmm(slot)}: "
        first_line2 = f"{count1:02d}/{count2:02d}"

        # Second line: sum per row group
        group_counts = []
        for g in row_groups:
            group_counts.append(str(np.sum(counter_matrix[g, slot] != "0")))
        second_line = "/".join(group_counts)

        statistics_list.append((first_line, first_line2, second_line))
    stats = []
    for i, t in enumerate(statistics_list):
        if i % 4 == 0:
            stats.append(t)
        elif i % 2 == 0 and t[2] != stats[-1][2]:
            stats.append(t)
    output_text = "ACar \n\n"
    for stat in stats:
        output_text += f"{stat[0]}{stat[1]}\n{stat[2]}\n\n"
    return output_text


# ================================================================

if __name__ == "__main__":
    # Default test inputs (same as Streamlit defaults)
    main_officers_reported = "1-18"
    report_gl_counters = "4AC1, 8AC11, 12AC21, 16AC31"
    handwritten_counters = "3AC12,5AC13"
    OT_counters = "2,20,40"
    sos_timings = "(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, 1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945"
    ro_ra_officers = "3RO2100, 11RO1700,15RO2130"
    results = run_algo(
        main_officers_reported,
        report_gl_counters,
        sos_timings,
        ro_ra_officers,
        handwritten_counters,
        OT_counters,
    )
    print("Results:", results[-1][0])
    # === Show plots ===
    plotter = Plotter(
        num_slots=NUM_SLOTS, num_counters=NUM_COUNTERS, start_hour=START_HOUR
    )

    fig1 = plotter.plot_officer_timetable_with_labels(results[0])
    fig2 = plotter.plot_officer_timetable_with_labels(results[1])
    fig3 = plotter.plot_officer_schedule_with_labels(results[2])

    # explicitly show them (works in script mode)
    fig1.show()
    fig2.show()
    fig3.show()
