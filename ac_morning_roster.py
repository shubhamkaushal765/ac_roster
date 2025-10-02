import numpy as np
import heapq
from copy import deepcopy
from collections import defaultdict
import random
import time
import copy

NUM_SLOTS = 48
START_HOUR = 10

# Number of intervals per counter (12 hours × 4 = 48)
intervals_per_shift = 48

def hhmm_to_slot(hhmm: str) -> int:
        """Convert hhmm string to a slot index (0–47)."""
        t = int(hhmm)
        h = t // 100
        m = t % 100
        slot = (h - START_HOUR) * 4 + (m // 15)
        return max(0, min(NUM_SLOTS - 1, slot))

# Initialize counters as a dict (keys 1–41)
counters = {i: [0] * intervals_per_shift for i in range(1, 42)}

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
import re
import numpy as np

def generate_main_officers_schedule(main_officers_template, main_officers_reported, main_officers_report_late_or_leave_early):
    START_HOUR = 10
    NUM_SLOTS = 48

    

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

    return main_officers_schedule

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


def build_officer_schedules(input_avail):
    """
    Build (officer_names, base_schedules_matrix).
    
    officer_names: list of officer IDs
    base_schedules: 2D numpy array (num_officers, NUM_SLOTS)
    """
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
        diffs = np.diff(self.work_count)
        return int(np.sum(diffs != 0))  # cast to int to avoid numpy.int64 issues

def greedy_smooth_schedule_beam(sos_schedule_matrix, main_officers_schedule, all_break_schedule, beam_width=10):
    I, L = sos_schedule_matrix.shape
    
    initial_work_count = sos_schedule_matrix.sum(axis=0) 
    if main_officers_schedule is not None:
        main_officers_binary = np.vstack([np.where(v != 0, 1, 0) for v in main_officers_schedule.values()])
        initial_work_count += main_officers_binary.sum(axis=0)
    
    # Beam elements: (penalty, SegmentTree, chosen_indices)
    beam = [(SegmentTree(initial_work_count), np.sum(np.diff(initial_work_count) != 0), [])]

    for officer in range(I):
        officer_key = f'O{officer+1}'
        candidates = all_break_schedule.get(officer_key, [])
        new_beam = []

        if not candidates:
            # No break schedules: extend beam with None
            for stree, pen, indices in beam:
                new_beam.append((stree, pen, indices + [None]))
            beam = new_beam
            continue

        for stree, pen, indices in beam:
            for idx, candidate in enumerate(candidates):
                delta_indices = np.where((sos_schedule_matrix[officer] == 1) & (candidate == 0))[0]
                new_stree = deepcopy(stree)
                if len(delta_indices) > 0:
                    new_stree.update_delta(delta_indices, -1)
                new_penalty = new_stree.compute_penalty()
                new_beam.append((new_stree, new_penalty, indices + [idx]))

        # Keep top-K by penalty
        beam = sorted(new_beam, key=lambda x: x[1])[:beam_width]

    # Return best solution
    best_stree, min_penalty, chosen_indices = min(beam, key=lambda x: x[1])
    best_work_count = best_stree.work_count
    return chosen_indices, best_work_count, min_penalty

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


def fill_sos_counter_manning(counter_matrix, counter_priority_list, paths, schedule_intervals_to_officers):
    # Deep copy so the original dict is not modified
    schedule_copy = copy.deepcopy(schedule_intervals_to_officers)
    
    # Find empty counters
    zero_rows = np.where(np.all(counter_matrix == 0, axis=1))[0]
    empty_counters = (zero_rows + 1).tolist()
    
    # Sort empty_counters according to the order in counter_priority_list
    empty_counters.sort(key=lambda x: counter_priority_list.index(x + 1) if (x + 1) in counter_priority_list else float('inf'))
    print(empty_counters)
    # Initialize sos_counter_manning (41 rows, 48 columns)
    sos_counter_manning = np.zeros((41, 48), dtype=int)
    
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


# ================================================================
counter_priority_list = [41] + [n for offset in range(0,10) for n in range(40 - offset, 0, -10)]
# === Example usage ===
input_avail = [
    '1000-1200','2000-2200','1300-1430,2030-2200','1300-1430,2030-2200',
    '1300-1430,2030-2200,1000-1130','1000-1600','1000-1600','1030-1900',
    '1030-1900','1030-1900','1030-1900','1030-1900','1100-2200','1100-2200',
    '1100-2200','1200-2200','1200-2200','1145-1830','1145-2200','1200-2200',
    '1145-2200','1145-2200','1230-1400','1130-1300','1300-1430','1230-1630',
    '1600-1830','1600-1830','1400-1830','1400-1830','1000-1200','2000-2200',
    '1800-2030','1700-2200'
]

input_avail = [
    '1000-1200','2000-2200','1300-1430,2030-2200','1300-1430,2030-2200',
    '1300-1430,2030-2200,1000-1130','1000-1600','1000-1600','1030-1900',
    '1030-1900','1030-1900','1030-1900','1030-1900','1100-2200','1100-2200',
    '1100-2200','1145-1830','1230-1400','1130-1300','1300-1430','1230-1630',
    '1600-1830','1600-1830','1400-1830','1400-1830','1000-1200','2000-2200',
    '1800-2030','1700-2200'
]

main_officers_template = init_main_officers_template()
main_officers_schedule = generate_main_officers_schedule(main_officers_template, "1-18", "2RX1000, 2RA0950, 3RO2215, 5RA1037, 99RA1000")
counter_matrix, counter_no = officer_to_counter_matrix(main_officers_schedule)
counter_w_partial_availability = find_partial_availability(counter_matrix)
officer_names, base_schedules = build_officer_schedules(input_avail)
all_break_schedules = generate_break_schedules(base_schedules, officer_names)


start_time = time.perf_counter()

chosen_schedule_indices, best_work_count, min_penalty = greedy_smooth_schedule_beam(
    base_schedules,None,all_break_schedules,beam_width=20)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"--- {elapsed_time:.4f} seconds ---")

sos_schedule_matrix = generate_sos_schedule_matrix(chosen_schedule_indices, all_break_schedules, officer_names)
schedule_intervals_to_officers, schedule_intervals = get_intervals_from_schedule(sos_schedule_matrix)
chains = greedy_longest_partition_inclusive(schedule_intervals)
paths = max_coverage_paths_inclusive(chains)
full_paths, partial_paths = split_full_partial_paths(paths)

print("=== best work count ===")
print(best_work_count)
for i, path in enumerate(full_paths, 1):
    print(f"Path {i}: {path}")

print('===full paths===')

for i, path in enumerate(full_paths, 1):
    print(f"Path {i}: {path}")

print('===partial paths===')

for i, path in enumerate(partial_paths, 1):
    print(f"Path {i}: {path}")

sos_counter_manning = fill_sos_counter_manning(counter_matrix, counter_priority_list, paths, schedule_intervals_to_officers)

prefixed_counter_matrix = prefix_non_zero(counter_matrix, "M")
print("===prefixed_counter_matrix===")
print(prefixed_counter_matrix)

prefixed_sos_counter_manning = prefix_non_zero(sos_counter_manning, "S")
print("===prefixed_sos_counter_manning===")
print(prefixed_sos_counter_manning)

merged = merge_prefixed_matrices(prefixed_counter_matrix, prefixed_sos_counter_manning)
print(merged)
print("Merged Counter Matrix (counter # : sos officers):")
for i, row in enumerate(merged):
    print(f"AC {counter_no[i]:2}: {format_slots_with_sep(row)}")

