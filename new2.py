NUM_SLOTS = 48
NUM_COUNTERS = 41
OPEN_CLOSE_PENALTY = 1
MULTI_OFFICER_PENALTY = 3
BREAK_EDGE_PENALTY = 1000  # break in first 2 slots or last 4 slots

ZONE_COUNTERS = {
    1: list(range(1, 11)),
    2: list(range(11, 21)),
    3: list(range(21, 31)),
    4: list(range(31, 41))
}

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

def enforce_mandatory_breaks(schedule: list[int]) -> list[int]:
    work_slots = [i for i, v in enumerate(schedule) if v > 0]
    if not work_slots:
        return schedule
    total_work = len(work_slots)
    
    # >36 slots: 2+3+3 consecutive breaks
    if total_work > 36:
        positions = [len(work_slots)//4, len(work_slots)//2, 3*len(work_slots)//4]
        for pos, length in zip(positions, [2,3,3]):
            start_slot = work_slots[pos]
            for j in range(length):
                if start_slot + j < NUM_SLOTS and schedule[start_slot + j] > 0:
                    schedule[start_slot + j] = 0
    
    # enforce 2 or 3 slot breaks if >10/20 consecutive
    streak = 0
    for i in range(NUM_SLOTS):
        if schedule[i] > 0:
            streak += 1
            if streak > 20:
                for j in range(i, min(i+3, NUM_SLOTS)):
                    if schedule[j] > 0:
                        schedule[j] = 0
                streak = 0
            elif streak > 10:
                for j in range(i, min(i+2, NUM_SLOTS)):
                    if schedule[j] > 0:
                        schedule[j] = 0
                streak = 0
        else:
            streak = 0
    return schedule

def calculate_penalty(schedule, counter_manning):
    penalty = 0

    # Identify officer intervals (SOS intervals)
    intervals = []
    in_interval = False
    for i, val in enumerate(schedule):
        if val != -1:
            if not in_interval:
                start = i
                in_interval = True
        else:
            if in_interval:
                end = i - 1
                intervals.append((start, end))
                in_interval = False
    if in_interval:
        intervals.append((start, len(schedule)-1))

    # Counter-centric penalties
    for c in range(1, NUM_COUNTERS+1):
        prev_occupied = False
        for i in range(NUM_SLOTS):
            occupied = counter_manning[c][i] > 0
            # open/close penalty: empty → occupied or occupied → empty
            if occupied and not prev_occupied:
                penalty += OPEN_CLOSE_PENALTY
            elif not occupied and prev_occupied:
                penalty += OPEN_CLOSE_PENALTY
            # multi-officer penalty
            if counter_manning[c][i] > 1:
                penalty += MULTI_OFFICER_PENALTY
            prev_occupied = occupied

    # Officer-centric penalties (break-edge)
    for start, end in intervals:
        for i in range(start, end+1):
            if schedule[i] == 0:  # break slot
                if i - start < 2 or end - i < 4:
                    penalty += BREAK_EDGE_PENALTY

    return penalty



def insert_bonus_breaks(schedule, counter_manning):
    i = 0
    while i < NUM_SLOTS:
        if schedule[i] > 0:
            start = i
            while i < NUM_SLOTS and schedule[i] > 0:
                i += 1
            end = i
            if end - start >= 3:
                best_penalty = calculate_penalty(schedule, counter_manning)
                best_pos = None
                for pos in range(start+1, end-1):
                    temp = schedule.copy()
                    temp[pos] = 0
                    pen = calculate_penalty(temp, counter_manning)
                    if pen < best_penalty:
                        best_penalty = pen
                        best_pos = pos
                if best_pos is not None:
                    schedule[best_pos] = 0
        else:
            i += 1
    return schedule

def assign_counters(input_avail):
    officer_schedules = {}  # store final schedules
    counter_manning = {c: [0]*NUM_SLOTS for c in range(1, NUM_COUNTERS+1)}
    iter_num = 0
    for officer_id, avail_str in enumerate(input_avail, start=1):
        iter_num += 1
        schedule = parse_availability(avail_str)
        schedule = enforce_mandatory_breaks(schedule)
        schedule = insert_bonus_breaks(schedule, counter_manning)
        
        current_counter = None
        for i in range(NUM_SLOTS):
            if schedule[i] > 0:
                if current_counter is None or (i>0 and schedule[i-1]==0):
                    # zone selection logic
                    zone_counts = {}
                    for z in range(1,5):
                        opened = sum(1 for c in ZONE_COUNTERS[z] if counter_manning[c][i]>0)
                        zone_counts[z] = opened
                    chosen_zone = min(zone_counts, key=lambda z: (zone_counts[z], -max(ZONE_COUNTERS[z])))
                    available_counters = [c for c in ZONE_COUNTERS[chosen_zone]]
                    used = [c for c in available_counters if counter_manning[c][i]>0]
                    unused = [c for c in available_counters if c not in used]
                    current_counter = max(unused) if unused else max(available_counters)
                schedule[i] = current_counter
                counter_manning[current_counter][i] += 1
            else:
                current_counter = None
        
        # store schedule
        officer_schedules[officer_id] = schedule
    
    # compute total penalty
    total_penalty = sum(calculate_penalty(sched, counter_manning) for sched in officer_schedules.values())
    print('iter',iter_num,total_penalty)
    return officer_schedules, counter_manning, total_penalty

# Example usage
input_avail = [
    '1000-1200','2000-2200','1300-1430,2030-2200','1300-1430,2030-2200',
    '1300-1430,2030-2200,1000-1130','1000-1600','1000-1600','1030-1900',
    '1030-1900','1030-1900','1030-1900','1030-1900','1100-2200','1100-2200',
    '1100-2200','1200-2200','1200-2200','1145-1830','1145-2200','1200-2200',
    '1145-2200','1145-2200','1230-1400','1130-1300','1300-1430','1230-1630',
    '1600-1830','1600-1830','1400-1830','1400-1830','1000-1200','2000-2200',
    '1800-2030','1700-2200'
]

officer_schedules, counter_manning, total_penalty = assign_counters(input_avail)

# Pretty print
for officer, sched in officer_schedules.items():
    print(f"Officer {officer}: {sched}")

print("\nCounter Manning:")
for c in range(1, NUM_COUNTERS+1):
    print(f"Counter {c}: {counter_manning[c]}")
print("\nTotal penalty:", total_penalty)
