import numpy as np
from copy import deepcopy
from collections import defaultdict
import re
import copy
from bisect import bisect_left
import plotly.graph_objects as go
import IPython

NUM_SLOTS = 48
NUM_COUNTERS = 41
START_HOUR = 10
counter_priority_list = [41] + [n for offset in range(0,10) for n in range(40 - offset, 0, -10)]


def hhmm_to_slot(hhmm: str) -> int:
        """Convert hhmm string to a slot index (0–47)."""
        t = int(hhmm)
        h = t // 100
        m = t % 100
        slot = (h - START_HOUR) * 4 + (m // 15)
        return max(0, min(NUM_SLOTS - 1, slot))

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
    a = [full_counters[0]]*6 + [0]*2 + [full_counters[1]]*7 + [0]*3 + [full_counters[2]]*9 + [0]*3 + [full_counters[0]]*9 + [0] + [full_counters[1]]*8
    b = [full_counters[1]]*8 + [0]*2 + [full_counters[2]]*8 + [0]*3 + [full_counters[0]]*9 + [0]*3 + [full_counters[1]]*7 + [0] + [full_counters[2]]*7
    c = [full_counters[2]]*10+ [0]*2 + [full_counters[0]]*9 + [0]*3 + [full_counters[1]]*9 + [0]*3 + [full_counters[2]]*5 + [0] + [0               ]*6
    d = [0               ]*5 + [0]*1 + [full_counters[0]]*6 + [0]*2 + [full_counters[1]]*10+ [0]*3 + [full_counters[2]]*9 + [0]*3+[full_counters[0]]*9
    return (a,b,c,d)

def init_main_officers_template(main_total = 24, exclude_main:list = None):
    main_officers = {}
    main_officers[1] = [41]*6 + [0]*2 + [30]*7 + [0]*3 + [20]*9 + [0]*3 + [40]*9 + [0] + [30]*8
    main_officers[2] = [30]*8 + [0]*2 + [20]*8 + [0]*3 + [41]*9 + [0]*3 + [30]*7 + [0] + [20]*7
    main_officers[3] = [20]*10+ [0]*2 + [41]*9 + [0]*3 + [30]*9 + [0]*3 + [20]*5 + [0] + [0 ]*6
    main_officers[4] = [0] *5 + [0]*1 + [40]*6 + [0]*2 + [30]*10+ [0]*3 + [20]*9 + [0]*3+[41]*9
    main_officers[5] = [40]*6 + [0]*2 + [9 ]*7 + [0]*3 + [29]*9 + [0]*3 + [41]*9 + [0] + [9 ]*8
    main_officers[6] = [9 ]*8 + [0]*2 + [29]*8 + [0]*3 + [40]*9 + [0]*3 + [9 ]*7 + [0] + [29]*7
    main_officers[7] = [29]*10+ [0]*2 + [40]*9 + [0]*3 + [9 ]*9 + [0]*3 + [29]*5 + [0] + [0 ]*6
    main_officers[8] = [0] *5 + [0]*1 + [41]*6 + [0]*2 + [9 ]*10+ [0]*3 + [29]*9 + [0]*3+[40]*9

    # Define groups of officers and their rosters
    groups = [
        ([9, 10, 11, 12],  [19, 38, 10]),
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

    main_officers = {f"M{i}": np.array(v) for i, v in main_officers.items()}
    return(main_officers)

# main_officers_template = init_main_officers_template()
# main_officers_template


def generate_main_officers_schedule(main_officers_template, main_officers_reported, report_gl_counters,main_officers_report_late_or_leave_early):

    # Parse which officers reported
    reported_officers = set()
    parts = main_officers_reported.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            for i in range(int(start), int(end) + 1):
                reported_officers.add(i)
        else:
            reported_officers.add(int(part))

    # --- Validation function (skip invalids) ---
    def validate_adjustments(input_str):
        valid_entries = []
        if not input_str.strip():
            return valid_entries  # no adjustments

        entries = input_str.split(',')
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            # Must match officer_id + RA/RO + 4 digits
            if not re.match(r"^\d+(RA|RO)\d{4}$", entry):
                print(f"⚠️ Skipping {entry}: invalid format")
                continue

            # Extract parts
            if "RA" in entry:
                idx = entry.index("RA")
                officer_id = int(entry[:idx])
                hhmm = entry[idx+2:]
                adj_type = "RA"
            else:
                idx = entry.index("RO")
                officer_id = int(entry[:idx])
                hhmm = entry[idx+2:]
                adj_type = "RO"

            # Officer must be in reported_officers
            if officer_id not in reported_officers:
                print(f"⚠️ Skipping {entry}: officer {officer_id} not in reported list.")
                continue

            # Validate HHMM
            h = int(hhmm[:2])
            m = int(hhmm[2:])
            if not (10 <= h <= 22):
                print(f"⚠️ Skipping {entry}: hour {h} out of range (1000–2200)")
                continue
            if m not in (0, 15, 30, 45):
                print(f"⚠️ Skipping {entry}: minutes {m} must be 00, 15, 30, or 45")
                continue
            if h == 22 and m > 0:
                print(f"⚠️ Skipping {entry}: must not exceed 2200")
                continue

            valid_entries.append((officer_id, adj_type, hhmm))
        return valid_entries

    # Validate and parse adjustments
    valid_adjustments = validate_adjustments(main_officers_report_late_or_leave_early)
    adjustments = {}
    for officer_id, adj_type, hhmm in valid_adjustments:
        slot = hhmm_to_slot(hhmm)
        adjustments[officer_id] = (adj_type, slot)

    # Build the schedule
    main_officers_schedule = {}
    for officer_id in reported_officers:
        officer_key = f'M{officer_id}'
        if officer_key not in main_officers_template:
            continue

        schedule = main_officers_template[officer_key].copy()

        if officer_id in adjustments:
            adjustment_type, slot = adjustments[officer_id]
            if adjustment_type == 'RA':
                schedule[:slot] = 0
            elif adjustment_type == 'RO':
                schedule[slot:] = 0
        main_officers_schedule[officer_key] = schedule
    
    report_list = [s.strip() for s in report_gl_counters.split(',')]

    for officer_counter in report_list:
        if "AC" in officer_counter:
            idx = officer_counter.index("AC")
            officer_id = int(officer_counter[:idx])
            counter_no = int(officer_counter[idx + 2:])

            # Only apply to officers divisible by 4
            if officer_id % 4 == 0:
                officer_key = "M" + str(officer_id)
                if officer_key in main_officers_schedule:
                    # Assign counter_no to slots 0 to 4 inclusive
                    for slot in range(0, 5):
                        main_officers_schedule[officer_key][slot] = counter_no


    return main_officers_schedule, reported_officers, valid_adjustments
def get_officer_last_counter_and_empty_counters(
    reported_officers, ro_ra_officers, counter_matrix
):
    """
    Compute each officer's last counter start slot (for officers with id % 4 == 3)
    and find counters that are empty from slot 42 onward.

    Args:
        reported_officers: list[int] — all officer IDs.
        ro_ra_officers: list[tuple] — e.g. [(officer_id, 'RO', '2030'), ...].
        counter_matrix: np.ndarray of shape (41, 48) — counter schedule.
        counter_priority_list: list[int] — ordered list of counters (1-indexed).
        hhmm_to_slot: function — converts HHMM (string/int) to slot index.

    Returns:
        officer_last_counter: dict {officer_id: last_counter_start_slot}
        empty_counters_2030: list[int] of row indices (0-indexed)
    """
    officer_last_counter = {}

    # Step 1: Assign last counter start slots for eligible officers
    for officer_id in reported_officers:
        if officer_id % 4 == 3:  # only if remainder is 3
            found = False
            for ro_ra_officer in ro_ra_officers:
                #print(ro_ra_officer)
                if ro_ra_officer[0] == officer_id and ro_ra_officer[1] == "RO":
                    last_counter_end_slot = hhmm_to_slot(ro_ra_officer[2])
                    found = True
                    break
            if not found:
                last_counter_end_slot = 48  # default fallback
            officer_last_counter[officer_id] = last_counter_end_slot

    # Step 2: Identify counters empty from slot 42 onwards
    empty_counters_2030 = [
        row_idx for row_idx in range(counter_matrix.shape[0])
        if np.all(counter_matrix[row_idx, 42:] == 0)
    ]
    # Step 3: Sort counters by priority order (convert to 0-indexed)
    empty_counters_2030.sort(
        key=lambda x: counter_priority_list.index(x + 1)
        if (x + 1) in counter_priority_list else float('inf')
    )
    return officer_last_counter, empty_counters_2030

def update_main_officers_schedule_last_counter(main_officers_schedule, officer_last_counter, empty_counters_2030):
    """
    For each officer in officer_last_counter, set their schedule from their
    last counter slot onwards to 0.

    Args:
        main_officers_schedule: dict {officer_id: np.array of schedule}
        officer_last_counter: dict {officer_id: last_counter_start_slot}

    Returns:
        updated_main_officers_schedule: dict with modified schedules
    """
    updated_schedule = {}

    for officer_id, schedule in main_officers_schedule.items():
        # Convert officer key if needed (e.g., 'M1', 'M2' etc.)
        # If main_officers_schedule keys are officer_id directly, skip this
        key_id = officer_id
        if isinstance(officer_id, str) and officer_id.startswith('M'):
            key_id = int(officer_id[1:])

        # Copy schedule
        updated_schedule[officer_id] = schedule.copy()

        # Update from last counter slot onwards if officer is in officer_last_counter
        if key_id in officer_last_counter:
            last_slot = officer_last_counter[key_id]
            if last_slot >= 42:
                updated_schedule[officer_id][42:last_slot] = empty_counters_2030[0]
                empty_counters_2030.pop(0)

    return updated_schedule


def officer_to_counter_matrix(officer_matrix):
    """
    Convert officer_matrix (dict of officer → array of counter assignments per slot)
    to a counter_matrix (num_counters x num_slots), where each row represents a counter 1..41.
    
    Parameters:
    -----------
    officer_matrix : dict
        Keys = officer names (like 'M1', 'M2', ...), values = np.array of counter assignments per slot
    
    Returns:
    --------
    counter_matrix : np.array
        Shape = (41, num_slots)
        Rows = counters 1..41
        Values = officer_id assigned to that counter in that slot (0 = no officer)
    
    counter_matrix_row_names : list of str
        Names of counters corresponding to each row, e.g., ['C1', 'C2', ..., 'C41']
    """
    num_counters = 41  # counters 1..41
    num_slots = len(next(iter(officer_matrix.values())))
    counter_matrix = np.zeros((num_counters, num_slots), dtype=int)
    
    for officer_idx, arr in enumerate(officer_matrix.values(), start=1):
        for slot, counter in enumerate(arr):
            if counter != 0:
                # subtract 1 since row 0 = counter 1
                counter_matrix[counter-1, slot] = officer_idx
    
    counter_matrix_row_names = [f'{i+1}' for i in range(num_counters)]
    
    return counter_matrix, counter_matrix_row_names

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
    unused_intervals = [iv for i, iv in enumerate(intervals) if i not in used_indices]

    return path_result, unused_intervals

def parse_availability(avail_str: str) -> np.ndarray:
    """
    Convert availability string (e.g., '1000-1200,2030-2200')
    into a binary numpy array of length NUM_SLOTS.
    """
    schedule = np.zeros(NUM_SLOTS, dtype=int)

    for rng in avail_str.split(','):
        start, end = rng.split('-')
        start_slot = hhmm_to_slot(start)
        end_slot = hhmm_to_slot(end)

        # Fill working slots; end is inclusive now
        schedule[start_slot:end_slot + 1] = 1

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

def build_officer_schedules(user_input):
    """
    Build (officer_names, base_schedules_matrix).
    officer_names: list of officer IDs
    base_schedules: 2D numpy array (num_officers, NUM_SLOTS)
    """
    input_avail = convert_input(user_input)
    officer_names = [f"O{i+1}" for i in range(len(input_avail))]
    schedules = [parse_availability(avail) for avail in input_avail]
    base_schedules = np.vstack(schedules)
    return officer_names, base_schedules

def generate_break_schedules(base_schedules, officer_names):
    def sliding_window_ok(schedule):
        """Check that no more than 10 consecutive 1s exist."""
        consec = 0
        for x in schedule:
            if x == 1:
                consec += 1
                if consec > 10:
                    return False
            else:
                consec = 0
        return True

    all_schedules = {}

    for idx, officer in enumerate(officer_names):
        base = base_schedules[idx].copy()
        work_slots = np.where(base == 1)[0]

        if len(work_slots) == 0:
            all_schedules[officer] = [base.copy()]
            #print(f"[{officer}] No working slots. Stored as-is: {base}")
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
            all_schedules[officer] = [base.copy()]
            #print(f"[{officer}] All stretches ≤10 slots. Stored as valid: {base}")
            continue

        #print(f"[{officer}] Some stretches >10 slots. Executing mandatory-break placement.")

        valid_schedules = []
        seen_schedules = set()

        def finalize_schedule(schedule, last_break_end, last_break_len):
            """Try 1-slot breaks if sliding-window violated after mandatory breaks."""
            if sliding_window_ok(schedule):
                sig = schedule.tobytes()
                if sig not in seen_schedules:
                    seen_schedules.add(sig)
                    valid_schedules.append(schedule)
                    #print(f"[{officer}] Schedule valid after mandatory breaks: {schedule}")
                return

            # Try inserting a 1-slot break in all working intervals
            for s in range(len(schedule)):
                if schedule[s] != 1:
                    continue

                # Determine current working interval dynamically
                next_break_index = s
                while next_break_index < len(schedule) and schedule[next_break_index] == 1:
                    next_break_index += 1
                interval_end = next_break_index - 1

                prev_break_index = s
                while prev_break_index >= 0 and schedule[prev_break_index] == 1:
                    prev_break_index -= 1
                interval_start = prev_break_index + 1

                # First/last 4 slots rule
                if s <= interval_start + 4 or s >= interval_end - 4:
                    continue

                # Spacing rule
                required_gap = min(2 * last_break_len, 4) if last_break_end >= 0 else 0
                if s - last_break_end - 1 < required_gap:
                    continue

                cand = schedule.copy()
                cand[s] = 0
                if sliding_window_ok(cand):
                    sig = cand.tobytes()
                    if sig not in seen_schedules:
                        seen_schedules.add(sig)
                        valid_schedules.append(cand)
                        #print(f"[{officer}] 1-slot break placed at {s} → schedule OK: {cand}")

            #if not valid_schedules:
                #print(f"[{officer}] No feasible 1-slot break placement, rejecting schedule: {schedule}")

        def place_breaks(schedule, stretch_idx=0, last_break_end=-1, last_break_len=0):
            """Recursive placement of mandatory breaks."""
            if stretch_idx >= len(stretches):
                finalize_schedule(schedule, last_break_end, last_break_len)
                return

            stretch = stretches[stretch_idx]
            min_slot, max_slot = stretch[0], stretch[-1]
            stretch_len = len(stretch)

            # Skip small stretches ≤10
            if stretch_len <= 10:
                place_breaks(schedule, stretch_idx + 1, last_break_end, last_break_len)
                return

            # Determine mandatory break pattern
            if stretch_len >= 36:
                pattern = [2, 3, 3]
            elif stretch_len >= 20:
                pattern = [2,3]
            elif stretch_len >=10:
                pattern = [2]
            else:  # 10-19
                pattern = [0]

            def recurse(schedule, blens, last_break_end, last_break_len):
                if not blens:
                    place_breaks(schedule, stretch_idx + 1, last_break_end, last_break_len)
                    return

                blen = blens[0]

                # Determine the start of the current working interval
                interval_start = min_slot
                if last_break_end >= 0:
                    interval_start = last_break_end + 1

                # Maximum allowed start to ensure no >10 consecutive slots
                max_consec_start = interval_start + 10
                max_allowed = min(max_consec_start, max_slot - blen - 3)
                # also respect last 4 slots

                for s in range(interval_start + 4, max_allowed + 1):  # respect first 4 slots
                    # Spacing rule
                    required_gap = min(2 * last_break_len, 4) if last_break_end >= 0 else 0
                    if s - last_break_end - 1 < required_gap:
                        continue

                    # Only place break if all slots are working
                    if not np.all(schedule[s:s + blen] == 1):
                        continue

                    new_sched = schedule.copy()
                    new_sched[s:s + blen] = 0
                    #print(f"[{officer}] Placing mandatory break {blen} at {s}-{s + blen - 1}")
                    #print(f"Partial schedule: {new_sched}")

                    recurse(new_sched, blens[1:], s + blen, blen)


            recurse(schedule, pattern, last_break_end, last_break_len)

        # Run recursion
        place_breaks(base.copy())

        all_schedules[officer] = valid_schedules if valid_schedules else [base.copy()]
        #print(f"[{officer}] Finished. Number of valid schedules: {len(valid_schedules)}\n")

    return all_schedules

# all_break_schedules = generate_break_schedules(base_schedules, officer_names)


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
    
def greedy_smooth_schedule_beam(sos_schedule_matrix, main_officers_schedule, all_break_schedule,
                                beam_width=50, alpha=0.1, beta=1.0):
    I, L = sos_schedule_matrix.shape

    # initial work count
    initial_work_count = sos_schedule_matrix.sum(axis=0)
    if main_officers_schedule is not None:
        main_officers_binary = np.vstack([np.where(v != 0, 1, 0) for v in main_officers_schedule.values()])
        initial_work_count += main_officers_binary.sum(axis=0)

    # Initialize beam: (SegmentTree, score, chosen_indices)
    stree = SegmentTree(initial_work_count)
    beam = [(stree, stree.compute_score(alpha, beta), [])]

    # Iterate over each SOS officer
    for officer in range(I):
        officer_key = f'O{officer+1}'
        candidates = all_break_schedule.get(officer_key, [])
        new_beam = []

        if not candidates:
            # No candidates — carry forward
            for stree, score, indices in beam:
                new_beam.append((stree, score, indices + [None]))
            beam = new_beam
            continue

        for stree, score, indices in beam:
            for idx, candidate in enumerate(candidates):
                delta_indices = np.where((sos_schedule_matrix[officer] == 1) & (candidate == 0))[0]
                new_stree = deepcopy(stree)
                if len(delta_indices) > 0:
                    new_stree.update_delta(delta_indices, -1)
                new_score = new_stree.compute_score(alpha, beta)
                new_beam.append((new_stree, new_score, indices + [idx]))

        # Keep only top-K by total score (lower = better)
        beam = sorted(new_beam, key=lambda x: x[1])[:beam_width]

    # Return best candidate
    best_stree, best_score, chosen_indices = min(beam, key=lambda x: x[1])
    return chosen_indices, best_stree.work_count, best_score

# chosen_schedule_indices, best_work_count, min_penalty = greedy_smooth_schedule_beam(
#     base_schedules,all_break_schedules,beam_width=20  # tune beam width
# )

# print(chosen_schedule_indices)
# print(best_work_count)
# print(min_penalty)

def generate_sos_schedule_matrix(saved_indices, all_break_schedules, officer_names):
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
    num_slots = len(next(iter(all_break_schedules[officer_names[0]])))  # assume all schedules same length
    
    sos_schedule_matrix = np.zeros((num_officers, num_slots), dtype=int)
    
    for i, officer in enumerate(officer_names):
        idx = saved_indices[i]
        sos_schedule_matrix[i] = all_break_schedules[officer][idx]
        
    return sos_schedule_matrix

#sos_schedule_matrix = generate_sos_schedule_matrix(chosen_schedule_indices, all_break_schedules, officer_names)

from collections import defaultdict
import numpy as np

def get_intervals_from_schedule(sos_schedule_matrix):
    interval_dict = defaultdict(list)
    sos_schedule_matrix = np.array(sos_schedule_matrix)  # Ensure it's a NumPy array

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

def greedy_longest_partition_inclusive(intervals):
    """
    Partition inclusive intervals into disjoint paths.
    Always pick the longest available path first.
    
    Inclusive intervals: (start, end) means both start and end are included.
    """
    intervals = intervals[:]  # copy
    paths = []

    def build_longest_path(remaining):
        # Build adjacency: start -> intervals
        start_map = defaultdict(list)
        for s, e in remaining:
            start_map[s].append((s, e))

        best_path = []

        def dfs(path, current_end, visited):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = path[:]

            next_start = current_end + 1  # for inclusive intervals
            if next_start not in start_map:
                return

            for nxt in start_map[next_start]:
                if nxt not in visited:
                    visited.add(nxt)
                    dfs(path + [nxt], nxt[1], visited)
                    visited.remove(nxt)

        # Try starting from every interval
        for interval in remaining:
            dfs([interval], interval[1], {interval})

        return best_path

    # Keep extracting longest paths until no intervals remain
    while intervals:
        longest = build_longest_path(intervals)
        if not longest:  # safety check
            break
        paths.append(longest)
        # Remove used intervals
        for iv in longest:
            intervals.remove(iv)

    return paths

def max_coverage_paths_inclusive(chains):
    """
    chains: list of chains, each chain = list of inclusive intervals [(start, end), ...]
    Returns flattened paths covering maximum ranges using inclusive logic.
    """
    # Assign unique indices to chains
    chain_indices = list(range(len(chains)))
    remaining_chains = set(chain_indices)
    all_paths = []

    while remaining_chains:
        best_path = []
        best_coverage = -1

        def dfs(path, coverage_end, used_chains):
            nonlocal best_path, best_coverage

            # Update best path if current coverage is better
            if coverage_end > best_coverage:
                best_coverage = coverage_end
                best_path = path[:]

            for idx in list(remaining_chains):
                if idx in used_chains:
                    continue
                chain = chains[idx]
                chain_start = chain[0][0]
                if chain_start >= coverage_end + 1:  # inclusive: next interval can start at coverage_end + 1
                    dfs(path + [chain], chain[-1][1], used_chains | {idx})

        # Run DFS starting with empty path
        dfs([], -1, set())  # start coverage at -1 to allow starting at 0

        # Commit the best path found
        all_paths.append(best_path)
        for chain in best_path:
            # Find its index and remove from remaining_chains
            for i in remaining_chains:
                if chains[i] == chain:
                    remaining_chains.remove(i)
                    break

    # Flatten the paths
    flattened_paths = [sum(path, []) for path in all_paths]
    return flattened_paths

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
        total_length = sum(end - start +1 for start, end in path)
        if total_length == target_length:
            full_paths.append(path)
        else:
            partial_paths.append(path)

    return full_paths, partial_paths


def fill_sos_counter_manning(counter_matrix, paths, schedule_intervals_to_officers):
    # Deep copy so the original dict is not modified
    schedule_copy = copy.deepcopy(schedule_intervals_to_officers)
    
    # Find empty counters
    zero_rows = np.where(np.all(counter_matrix == 0, axis=1))[0]
    empty_counters = (zero_rows + 1).tolist()
    
    # Sort empty_counters according to the order in counter_priority_list
    empty_counters.sort(key=lambda x: counter_priority_list.index(x + 1) if (x + 1) in counter_priority_list else float('inf'))
    print(empty_counters)
    # Initialize sos_counter_manning (41 rows, 48 columns)
    sos_counter_manning = np.zeros((NUM_COUNTERS, NUM_SLOTS), dtype=int)
    
    for i, each_path in enumerate(paths):
        if not empty_counters:
            print("No available counters left for path", i)
            break
        
        # Pick the first available counter (highest priority)
        priority_counter = empty_counters.pop(0)
        #print(f"Assigning path {i} to counter {priority_counter}")
        
        # Fill the intervals in the chosen counter
        for interval in each_path:
            start_index, end_index = interval
            
            if interval in schedule_copy and schedule_copy[interval]:
                officer_id = schedule_copy[interval].pop(0)
                #print(f"Interval {interval} -> Officer {officer_id}")
                
                # Fill in the sos_counter_manning row for this interval
                sos_counter_manning[priority_counter-1, start_index:end_index+1] = officer_id
                
            else:
                print(f"Cannot find officer for interval {interval}")
    return sos_counter_manning

def prefix_non_zero(counter_matrix, prefix):
    # Create an empty array of same shape, dtype=object to hold strings
    result = np.empty(counter_matrix.shape, dtype=object)
    
    # Fill zeros as string "0"
    result[counter_matrix == 0] = "0"
    
    # Fill non-zero elements with "M" prefix
    result[counter_matrix != 0] = [prefix + str(x) for x in counter_matrix[counter_matrix != 0]]
    
    return result

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
    merged_matrix = np.where(sos_matrix != '0', sos_matrix, counter_matrix)
    
    return merged_matrix

def format_slots_with_sep(row, sep_every=4):
    formatted = []
    for i, x in enumerate(row):
        formatted.append(f"{x:4}" if str(x) != '0' else " .  ")
        if (i + 1) % sep_every == 0 and (i + 1) != len(row):
            formatted.append("|")  # add separator
    return ' '.join(formatted)

def slot_officers_matrix_gap_aware(schedule_intervals_to_officers, partial_empty_rows):
    """
    Assign officers to counters using gap-aware greedy interval packing.

    Args:
        schedule_intervals_to_officers: dict {(start,end): [officer_ids]}
        partial_empty_rows: dict {counter_id: [(avail_start, avail_end), ...]}
        NUM_COUNTERS: number of counters
        NUM_SLOTS: number of time slots

    Returns:
        counter_matrix: np.array of shape (total_counters, total_slots)
                        each element is officer_id or '0'
    """
    counter_matrix = np.full((NUM_COUNTERS, NUM_SLOTS), '0', dtype=object)
    
    # Track occupied intervals per counter: counter_id -> sorted list of (start,end,officer_id)
    # ONLY initialize counters we can actually use
    counter_occupied = {}
    
    # CRITICAL FIX: Initialize partial counters with RESERVED blocks for unavailable ranges
    for counter_id, avail_ranges in partial_empty_rows.items():
        reserved = []
        sorted_ranges = sorted(avail_ranges)
        
        # Mark everything before first available range as RESERVED
        if sorted_ranges[0][0] > 0:
            reserved.append((0, sorted_ranges[0][0] - 1, 'RESERVED'))
        
        # Mark gaps between available ranges as RESERVED
        for i in range(len(sorted_ranges) - 1):
            gap_start = sorted_ranges[i][1] + 1
            gap_end = sorted_ranges[i + 1][0] - 1
            if gap_start <= gap_end:
                reserved.append((gap_start, gap_end, 'RESERVED'))
        
        # Mark everything after last available range as RESERVED
        if sorted_ranges[-1][1] < NUM_SLOTS - 1:
            reserved.append((sorted_ranges[-1][1] + 1, NUM_SLOTS - 1, 'RESERVED'))
        
        counter_occupied[counter_id] = reserved
    
    # empty_counters is 1-indexed (counter names), convert to 0-indexed (counter IDs)
    empty_counters = [8, 37, 36, 26, 16, 6, 35, 25, 15, 5, 34, 24, 14, 4, 33, 23, 13, 3, 32, 22, 12, 2, 31, 21, 11, 1]
    empty_counters = [c - 1 for c in empty_counters]  # Convert to 0-indexed
    # Filter out any counters that are already partial
    empty_counters = [c for c in empty_counters if c not in partial_empty_rows]
    print(f"Available empty counters (0-indexed) after filtering: {empty_counters}")
    print(f"Available empty counters (display names): {['C' + str(c+1) for c in empty_counters]}")
    print(f"Partial counters: {list(partial_empty_rows.keys())}")
    empty_idx = 0
    
    # Sort intervals by start time, then by end time
    sorted_intervals = sorted(schedule_intervals_to_officers.items(), 
                             key=lambda x: (x[0][0], x[0][1]))
    
    # Helper: check if interval can be placed in occupied list at a position
    def can_place(occupied, interval):
        """
        Check if interval can fit without overlap.
        Returns (can_fit: bool, insert_idx: int)
        """
        start, end = interval
        
        # Use only start times for binary search
        occupied_starts = [s for s, e, _ in occupied]
        idx = bisect_left(occupied_starts, start)
        
        # Check overlap with previous interval
        if idx > 0:
            prev_start, prev_end, _ = occupied[idx - 1]
            # Inclusive intervals: [a,b] and [c,d] overlap if max(a,c) <= min(b,d)
            if max(prev_start, start) <= min(prev_end, end):
                return False, idx
        
        # Check overlap with next interval
        if idx < len(occupied):
            next_start, next_end, _ = occupied[idx]
            if max(next_start, start) <= min(next_end, end):
                return False, idx
        
        return True, idx
    
    for interval, officer_ids in sorted_intervals:
        start, end = interval
        
        for officer_id in officer_ids:
            best_counter = None
            best_score = -1
            best_insert_idx = -1
            
            # Step 1: try partial counters first (RESERVED blocks will prevent invalid placement)
            for counter_id in partial_empty_rows.keys():
                occupied = counter_occupied[counter_id]
                can_fit, insert_idx = can_place(occupied, interval)
                if can_fit:
                    # Score: bonus for consecutive placement
                    score = 50  # base bonus for partial counter
                    if insert_idx > 0 and occupied[insert_idx-1][1] + 1 == start and occupied[insert_idx-1][2] != 'RESERVED':
                        score += 100  # connects to previous (not RESERVED)
                    if insert_idx < len(occupied) and end + 1 == occupied[insert_idx][0] and occupied[insert_idx][2] != 'RESERVED':
                        score += 100  # connects to next (not RESERVED)
                    
                    if score > best_score:
                        best_score = score
                        best_counter = counter_id
                        best_insert_idx = insert_idx
            
            # Step 2: try already used empty counters (gap allowed)
            if best_counter is None:
                for counter_id in list(counter_occupied.keys()):
                    # Skip partial counters (already tried in step 1)
                    if counter_id in partial_empty_rows:
                        continue
                    
                    occupied = counter_occupied[counter_id]
                    can_fit, insert_idx = can_place(occupied, interval)
                    if can_fit:
                        score = 10  # base score for reusing counter
                        # bonus if connects to previous or next
                        if insert_idx > 0 and occupied[insert_idx-1][1] + 1 == start:
                            score += 100
                        if insert_idx < len(occupied) and end + 1 == occupied[insert_idx][0]:
                            score += 100
                        
                        if score > best_score:
                            best_score = score
                            best_counter = counter_id
                            best_insert_idx = insert_idx
            
            # Step 3: use a new empty counter
            if best_counter is None:
                if empty_idx < len(empty_counters):
                    best_counter = empty_counters[empty_idx]
                    #print(f"  Assigning officer {officer_id} interval {interval} to NEW counter {best_counter} (C{best_counter+1})")
                    counter_occupied[best_counter] = []  # Initialize this counter
                    empty_idx += 1
                    best_insert_idx = 0
                else:
                    #print(f"ERROR: No available counter for officer {officer_id}, interval {interval}")
                    continue
            
            # Place officer in counter
            counter_occupied[best_counter].insert(best_insert_idx, (start, end, officer_id))
            # Fill the matrix (inclusive range)
            counter_matrix[best_counter, start:end+1] = f"S{officer_id}"
    
    # Print statistics
    used_counters = sum(1 for c in counter_occupied if any(o[2] != 'RESERVED' for o in counter_occupied[c]))
    partial_used = sum(1 for c in partial_empty_rows if any(o[2] != 'RESERVED' for o in counter_occupied[c]))
    new_empty_used = used_counters - partial_used
    
    print(f"\nAssignment Statistics:")
    print(f"  Partial counters used: {partial_used}/{len(partial_empty_rows)}")
    print(f"  New empty counters used: {new_empty_used}")
    print(f"  Total counters used: {used_counters}/{NUM_COUNTERS}")
    
    # Print partial counter validation
    print(f"\nPartial Counter Validation:")
    for counter_id, avail_ranges in partial_empty_rows.items():
        assignments = [o for o in counter_occupied[counter_id] if o[2] != 'RESERVED']
        print(f"  Counter {counter_id}: available {avail_ranges}, assigned {len(assignments)} intervals")
        for s, e, oid in assignments:
            # Verify each assignment is within available ranges
            in_range = any(s >= as_ and e <= ae for as_, ae in avail_ranges)
            status = "✓" if in_range else "✗ VIOLATION"
            print(f"    ({s},{e}) officer {oid} {status}")
    
    return counter_matrix

def plot_officer_timetable_with_labels(counter_matrix):
    """
    Plots an interactive timetable with officer IDs inside each cell.
    Consecutive cells with the same officer are merged with one label and thick border.
    
    Parameters:
    - counter_matrix: 2D numpy array or list of lists
        Shape: (NUM_SLOTS, NUM_COUNTERS) = (48, 41)
        counter_matrix[i, j] = officer at time_slot i, counter j
    
    Returns:
    - fig: Plotly Figure object
    """
    counter_matrix = np.array(counter_matrix)
    time_slots, num_counters = counter_matrix.shape  # (48, 41)
    
    # Create numeric matrix for colors
    color_matrix = np.zeros((time_slots, num_counters), dtype=int)
    for i in range(time_slots):
        for j in range(num_counters):
            val = str(counter_matrix[i, j])
            if val.startswith('M'):
                color_matrix[i, j] = 1
            elif val.startswith('S'):
                color_matrix[i, j] = 2
            else:
                color_matrix[i, j] = 0
    
    # Create heatmap
    heatmap = go.Heatmap(
        z=color_matrix,
        y=[f'C{t+1}' for t in range(time_slots)],
        x=[f'T{c}' for c in range(num_counters)],
        showscale=False,
        colorscale=[
            [0, '#2E2C2C'],
            [0.5, '#a2d2ff'],
            [1, '#ffc6d9']
        ]
    )
    
    # Find merged regions (consecutive horizontal cells with same officer)
    annotations = []
    shapes = []
    
    for i in range(time_slots):
        j = 0
        while j < num_counters:
            officer = str(counter_matrix[i, j])
            if officer != '0':
                # Find the end of this consecutive block
                j_end = j
                while j_end < num_counters and str(counter_matrix[i, j_end]) == officer:
                    j_end += 1
                
                # Add annotation at the center of the merged region
                center_x = (j + j_end - 1) / 2
                annotations.append(
                    dict(
                        x=center_x,
                        y=i,
                        text=officer,
                        showarrow=False,
                        font=dict(color='black', size=10)
                    )
                )
                
                # Add border around the merged region
                shapes.append(
                    dict(
                        type='rect',
                        x0=j - 0.5,
                        x1=j_end - 0.5,
                        y0=i - 0.5,
                        y1=i + 0.5,
                        line=dict(color='black', width=2),
                        fillcolor='rgba(0,0,0,0)'
                    )
                )
                
                j = j_end
            else:
                j += 1
    
    # Create figure
    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title='Officer Timetable',
        xaxis_title='Time Slot',
        yaxis_title='Counter',
        width=1400,
        height=900,
        annotations=annotations,
        shapes=shapes,
        yaxis_autorange='reversed'
    )
    return fig
# Assuming your matrix is called `merged`
#plot_officer_timetable_with_labels(merged)


import numpy as np

def find_empty_rows(counter_matrix):
    counter_matrix = np.array(counter_matrix)
    empty_rows = []
    partial_empty_rows = {}

    for i, row in enumerate(counter_matrix):
        zero_indices = np.where(row == 0)[0]
        zero_indices = zero_indices.astype(int)  # convert to native int
        if len(zero_indices) == 0:
            continue  # no zeros, skip
        if len(zero_indices) == len(row):
            empty_rows.append(i)  # entire row is zero
        else:
            # find consecutive zero ranges
            ranges = []
            start = zero_indices[0]
            for j in range(1, len(zero_indices)):
                if zero_indices[j] != zero_indices[j-1] + 1:
                    end = zero_indices[j-1]
                    ranges.append((int(start), int(end)))
                    start = zero_indices[j]
            ranges.append((int(start), int(zero_indices[-1])))  # last range
            partial_empty_rows[i] = ranges
    partial_empty_rows_index = list({t for ranges in partial_empty_rows.values() for t in ranges})
    # Optional: sort by first element for readability
    partial_empty_rows_index.sort()
    empty_rows.sort(key=lambda x: counter_priority_list.index(x + 1) if (x + 1) in counter_priority_list else float('inf'))
    return empty_rows, partial_empty_rows, partial_empty_rows_index

# ================================================================
