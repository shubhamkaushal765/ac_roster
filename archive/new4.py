from copy import deepcopy
from itertools import product

# -------------------------
# Constants
# -------------------------
COUNTER_COUNT = 41
NUM_SLOTS = 48  # 12h shift, 15min intervals

# -------------------------
# Helper Functions
# -------------------------
def hhmm_to_slot(hhmm: str) -> int:
    t = int(hhmm)
    h = t // 100
    m = t % 100
    slot = (h - 10) * 4 + (m // 15)
    return max(0, min(NUM_SLOTS - 1, slot))

def parse_availability(avail_str: str) -> list[int]:
    slots = [-1] * NUM_SLOTS
    intervals = [s.strip() for s in avail_str.split(",")]
    for interval in intervals:
        if not interval:
            continue
        start, end = interval.split("-")
        s, e = hhmm_to_slot(start), hhmm_to_slot(end)
        for i in range(s, min(e, NUM_SLOTS)):
            slots[i] = 1
    return slots

def segments_from_breaks(start, end, breaks):
    """Return work segments between breaks"""
    segs = []
    current = start
    for bstart, bend in sorted(breaks):
        if current < bstart:
            segs.append((current, bstart-1))
        current = bend + 1
    if current <= end:
        segs.append((current, end))
    return segs

# -------------------------
# Break Generation
# -------------------------
def generate_break_time(sos_period, availability):
    start, end = sos_period
    L = end - start + 1
    if L < 4:
        return [[]]  # no break required

    break_earliest = start + 4
    break_latest = end - 4
    available_slots = [i for i in range(break_earliest, break_latest + 1) if availability[i] == 1]

    results = []

    def place_breaks(placed, remaining_lengths):
        if not remaining_lengths:
            results.append(sorted(placed[:]))
            return
        size = remaining_lengths[0]
        for s in available_slots:
            bend = s + size - 1
            if bend > end or any(availability[i] != 1 for i in range(s, bend + 1)):
                continue
            if placed and s <= placed[-1][1] + 1:
                continue
            placed.append((s, bend))
            place_breaks(placed, remaining_lengths[1:])
            placed.pop()

    if L > 36:
        required_sizes = [2, 3, 3]
    elif L >= 20:
        required_sizes = [3]
    elif L >= 10:
        required_sizes = [2]
    else:
        required_sizes = []

    place_breaks([], required_sizes)
    return results if results else [[]]  # always at least one option

# -------------------------
# Allocate Counter
# -------------------------
def allocate_counter(counter_manning, sos_period, break_schedule, officer_id, availability):
    start, end = sos_period
    slots_to_assign = [i for i in range(start, end + 1) if availability[i] == 1]

    # Find free counters (prefer larger counter no.)
    free_counters = []
    for cno in sorted(counter_manning.keys(), reverse=True):
        if all(counter_manning[cno][i] in [-2, -1, 0] for i in slots_to_assign):
            free_counters.append(cno)

    chosen = free_counters[0] if free_counters else min(counter_manning.keys())

    # Mark slots
    for i in range(NUM_SLOTS):
        if availability[i] != 1:
            counter_manning[chosen][i] = -2
        else:
            counter_manning[chosen][i] = -1

    # Insert breaks
    if break_schedule:
        for bstart, bend in break_schedule:
            for i in range(bstart, bend + 1):
                counter_manning[chosen][i] = 0

    # Assign officer number to remaining unallocated SOS slots
    for i in slots_to_assign:
        if counter_manning[chosen][i] == -1:
            counter_manning[chosen][i] = chosen

    allocation_meta = {"counter": chosen, "span": (slots_to_assign[0], slots_to_assign[-1])}
    return counter_manning, allocation_meta

# -------------------------
# Open/Close Penalty (Counter-centric)
# -------------------------
def calculate_open_close_penalty(counter_manning):
    penalty = 0
    for cno, slots in counter_manning.items():
        prev_open = slots[0] > 0
        for i in range(1, len(slots)):
            curr_open = slots[i] > 0
            if curr_open != prev_open:
                penalty += 1
            prev_open = curr_open
    return penalty

# -------------------------
# Calculate Penalty
# -------------------------
def calculate_penalty(counter_manning, officer_options):
    multi_officer_penalty = 0
    break_penalty = 0
    open_close_penalty = calculate_open_close_penalty(counter_manning)

    # Multi-officer overlap
    for cno, slots in counter_manning.items():
        for i in range(NUM_SLOTS):
            if slots[i] > 0:
                overlap = sum(1 for c2, s2 in counter_manning.items() if c2 != cno and s2[i] > 0)
                if overlap >= 1:
                    multi_officer_penalty += 1

    # Break penalties (officer-centric)
    for officer in officer_options:
        assigned = officer.get("assigned_slots", [])
        for sos_idx, (start, end) in enumerate(officer["sos_list"]):
            segs = []
            current_seg = []
            for i in range(start, end + 1):
                if assigned[i] > 0:
                    current_seg.append(i)
                else:
                    if current_seg:
                        segs.append(current_seg)
                        current_seg = []
            if current_seg:
                segs.append(current_seg)

            # Streak >10
            for seg in segs:
                if len(seg) > 10:
                    break_penalty += (len(seg) - 10)

            # Very long schedule >36 slots
            total_work = sum(1 for i in range(NUM_SLOTS) if assigned[i] > 0)
            if total_work > 36:
                num_breaks = sum(1 for i in range(NUM_SLOTS) if assigned[i] == 0)
                if num_breaks < 8:
                    break_penalty += (8 - num_breaks)

    total_penalty = multi_officer_penalty*5 + break_penalty*2 + open_close_penalty*3
    return total_penalty

# -------------------------
# Master Solver
# -------------------------
def master_solver(input_avail, n_counters=COUNTER_COUNT):
    officer_avail = [parse_availability(av) for av in input_avail]
    n_officers = len(officer_avail)

    # Initialize counters
    counter_manning = {i: [-2]*NUM_SLOTS for i in range(1, n_counters+1)}

    # Find SOS periods per officer
    officer_sos_periods = []
    for idx, slots in enumerate(officer_avail):
        sos_list = []
        i = 0
        while i < NUM_SLOTS:
            if slots[i] == 1:
                start = i
                while i + 1 < NUM_SLOTS and slots[i+1] == 1:
                    i += 1
                end = i
                sos_list.append((start,end))
            i += 1
        officer_sos_periods.append({"id": idx+1, "slots": slots, "sos_list": sos_list})

    # Prepare break options
    officer_options = []
    for officer in officer_sos_periods:
        all_breaks = []
        for sos in officer["sos_list"]:
            breaks = generate_break_time(sos, officer["slots"])
            all_breaks.append(breaks)
        officer_options.append({
            "id": officer["id"],
            "sos_list": officer["sos_list"],
            "break_choices": all_breaks,
            "availability": officer["slots"],
            "assigned_slots": [-1]*NUM_SLOTS
        })

    # Recursive search
    best = {"penalty": float('inf'), "counter_manning": None, "allocations": None}

    def recurse(off_idx, cm, assignments):
        nonlocal best
        if off_idx >= n_officers:
            # Assign officer-centric slots for penalty calculation
            for idx, officer in enumerate(officer_options):
                officer["assigned_slots"] = [-1]*NUM_SLOTS
                for assignment in assignments[idx]["assignments"]:
                    start, end = assignment["sos"]
                    chosen_counter = assignment["meta"]["counter"]
                    for i in range(start, end+1):
                        if chosen_counter is not None and cm[chosen_counter][i] > 0:
                            officer["assigned_slots"][i] = cm[chosen_counter][i]

            pen = calculate_penalty(cm, officer_options)
            if pen < best["penalty"]:
                best["penalty"] = pen
                best["counter_manning"] = deepcopy(cm)
                best["allocations"] = deepcopy(assignments)
            return

        officer = officer_options[off_idx]
        sos_list = officer["sos_list"]
        break_choices_per_sos = officer["break_choices"]

        for break_combination in product(*break_choices_per_sos):
            cm_copy = deepcopy(cm)
            local_assignments = []
            for sos_idx, sos in enumerate(sos_list):
                breaks = break_combination[sos_idx]
                cm_copy, meta = allocate_counter(cm_copy, sos, breaks, officer["id"], officer["availability"])
                local_assignments.append({"sos": sos, "breaks": breaks, "meta": meta})
            assignments.append({"officer_id": officer["id"], "assignments": local_assignments})
            recurse(off_idx+1, cm_copy, assignments)
            assignments.pop()

    recurse(0, counter_manning, [])
    return best

# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    input_avail = [
        '1000-1200','2000-2200','1300-1430,2030-2200','1300-1430,2030-2200',
        '1300-1430,2030-2200,1000-1130','1000-1600','1000-1600','1030-1900',
        '1030-1900','1030-1900','1030-1900','1030-1900','1100-2200','1100-2200',
        '1100-2200','1200-2200','1200-2200','1145-1830','1145-2200','1200-2200',
        '1145-2200','1145-2200','1230-1400','1130-1300','1300-1430','1230-1630',
        '1600-1830','1600-1830','1400-1830','1400-1830','1000-1200','2000-2200',
        '1800-2030','1700-2200'
    ]

    best_schedule = master_solver(input_avail, n_counters=COUNTER_COUNT)
    print("Best penalty:", best_schedule["penalty"])
    for cno, slots in best_schedule["counter_manning"].items():
        print(f"Counter {cno}: {slots}")
