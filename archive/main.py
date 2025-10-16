import numpy as np

def time_to_interval(time_int: int, shift: str = "M") -> int:
    """
    Convert a time in HHMM (24-hour integer format) to its 15-min interval position.

    - Morning shift ("M"): 10:00 = 0, 22:00 = 48
    - Night shift   ("N"): 22:00 = 0, 10:00 = 48

    Args:
        time_int (int): Time in HHMM format (e.g. 1015, 2200).
        shift (str): "M" for morning, "N" for night.

    Returns:
        int: Interval position (0–48).

    Raises:
        ValueError: If time is outside the shift range or shift is invalid.
    """
    hours = time_int // 100
    minutes = time_int % 100
    total_minutes = hours * 60 + minutes

    if shift == "M":  # Morning shift
        start_minutes = 10 * 60  # 10:00
        end_minutes = 22 * 60    # 22:00
        if not (start_minutes <= total_minutes <= end_minutes):
            raise ValueError("Time outside morning shift (10:00–22:00).")
        interval_position = (total_minutes - start_minutes) // 15

    elif shift == "N":  # Night shift
        start_minutes = 22 * 60  # 22:00
        end_minutes = 10 * 60    # 10:00 (next day)
        if total_minutes < start_minutes:
            total_minutes += 24 * 60
        if not (start_minutes <= total_minutes <= 24*60 + end_minutes):
            raise ValueError("Time outside night shift (22:00–10:00).")
        interval_position = (total_minutes - start_minutes) // 15

    else:
        raise ValueError("Shift must be 'M' (morning) or 'N' (night).")

    return interval_position

#time_list = ["2300", "2345", "0030", "0900"]
#for t in time_list:
#    print(time_to_interval(int(t), "N"))


# Number of intervals per counter (12 hours × 4 = 48)
intervals_per_shift = 48

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
    print('a:', a)
    print('b:', b)
    print('c:', c)
    print('d:', d)
    return (a,b,c,d)

def add_main_officers(main_total = 24, exclude_main:list = None):
    main_officers = {i: [0] * intervals_per_shift for i in range(1, 33)}
    main_officers[1] = [41]*6 + [0]*2 + [30]*7 + [0]*3 + [20]*9 + [0]*3 + [40]*9 + [0] + [30]*8
    main_officers[2] = [30]*8 + [0]*2 + [20]*8 + [0]*3 + [41]*9 + [0]*3 + [30]*7 + [0] + [20]*7
    main_officers[3] = [20]*10+ [0]*2 + [41]*9 + [0]*3 + [30]*9 + [0]*3 + [20]*5 + [0] + [0 ]*6
    main_officers[4] = [0] *5 + [0]*1 + [40]*6 + [0]*2 + [30]*10+ [0]*3 + [20]*9 + [0]*3+[41]*9
    main_officers[5] = [40]*6 + [0]*2 + [9 ]*7 + [0]*3 + [29]*9 + [0]*3 + [41]*9 + [0] + [9 ]*8
    main_officers[6] = [9 ]*8 + [0]*2 + [29]*8 + [0]*3 + [40]*9 + [0]*3 + [9 ]*7 + [0] + [29]*7
    main_officers[7] = [29]*10+ [0]*2 + [40]*9 + [0]*3 + [9 ]*9 + [0]*3 + [29]*5 + [0] + [0 ]*6
    main_officers[8] = [0] *5 + [0]*1 + [41]*6 + [0]*2 + [9 ]*10+ [0]*3 + [29]*9 + [0]*3+[40]*9

    m_no = [9,10,11,12]
    main_officers[m_no[0]],main_officers[m_no[1]],main_officers[m_no[2]],main_officers[m_no[3]] = add_4main_roster([19,38,10])
    m_no = [13,14,15,16]
    main_officers[m_no[0]],main_officers[m_no[1]],main_officers[m_no[2]],main_officers[m_no[3]] = add_4main_roster([28,17,39])
    m_no = [17,18,19,20]
    main_officers[m_no[0]],main_officers[m_no[1]],main_officers[m_no[2]],main_officers[m_no[3]] = add_4main_roster([7,27,18])
    m_no = [21,22,23,24]
    main_officers[m_no[0]],main_officers[m_no[1]],main_officers[m_no[2]],main_officers[m_no[3]] = add_4main_roster([37,8,26])
    if main_total == 23:
        main_officers[24] = [0] * intervals_per_shift # no officers, respective counters empty
    if main_total > 24:
        m_no = [25,26,27,28]
        main_officers[m_no[0]],main_officers[m_no[1]],main_officers[m_no[2]],main_officers[m_no[3]] = add_4main_roster([15,35,5])
    if main_total > 28:
        m_no = [29,30,31,32]
        main_officers[m_no[0]],main_officers[m_no[1]],main_officers[m_no[2]],main_officers[m_no[3]] = add_4main_roster([24,16,36])
    
    if exclude_main != None:
        for i in exclude_main:
            main_officers[i] = [0] * intervals_per_shift # no officers, respective counters empty
        
    
    #for i in range (1,33):
        #print('m', i, '=', main_officers[i])
    return(main_officers)

main_officers = add_main_officers(exclude_main=[11])


def generate_ctr_stats(main_officers):
    counter_manning = {i: [0] * len(next(iter(main_officers.values()))) for i in range(1, 42)}

    # Fill in the counts
    for officer_schedule in main_officers.values():
        for interval_index, counter_no in enumerate(officer_schedule):
            if counter_no != 0:
                counter_manning[counter_no][interval_index] += 1
    return counter_manning

counter_manning = generate_ctr_stats(main_officers)
print(counter_manning)
# for i in range (1,42):
#     print(i,':', counter_manning[i])


def add_sos_officers(sos_time:list):
    schedules = np.full((len(sos_time), intervals_per_shift), -2, dtype=int) # initiate each sos officer availability. -2 = out of sos period, -1 = unallocated sos counter, 0 = break, 1 to 40 = counter no.
    for officer_idx, periods_str in enumerate(sos_time):
        periods = periods_str.split(",")
        for p in periods:
            start_time, end_time = p.split("-")
            start_idx = time_to_interval(int(start_time))
            end_idx = time_to_interval(int(end_time))
            mask = np.arange(intervals_per_shift)
            schedules[officer_idx, (mask >= start_idx) & (mask <= end_idx)] += 1

    # Convert schedules to dictionary with officer serial numbers as keys
    sos_officer_schedule_dict = {i+1: list(schedules[i].tolist()) for i in range(len(sos_time))}
    print(sos_officer_schedule_dict)
    return (sos_officer_schedule_dict)

#sos_time = ['1000-1200','2000-2200','1300-1430, 2030-2200','1300-1430, 2030-2200','1300-1430, 2030-2200,1000-1130','1400-1830','1800-2030']
#add_sos_officers(sos_time)

# Utility to compute contiguous SOS segments for an officer (list of (start,end) inclusive)
# -------------------------
def compute_sos_segments(schedule: list[int]) -> list[tuple[int,int]]:
    segs = []
    i = 0
    L = len(schedule)
    while i < L:
        if schedule[i] == -1:
            j = i
            while j + 1 < L and schedule[j + 1] == -1:
                j += 1
            segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs

# -------------------------
# Decide break length given SOS segment length (in intervals)
# Uses your rules (approximate):
# - >9h (>=36 intervals): give standard pattern (2,3,3) handled as multiple breaks when needed
# - >5h (>=20 intervals): total break time 3-4 intervals
# - >3h (>=12 intervals): total break 2-4 intervals
# else default 2 intervals
# This function returns recommended total break intervals for that segment.
# -------------------------
def recommended_total_breaks(segment_len:int) -> int:
    hours = segment_len / 4.0
    if segment_len >= 36:   # >=9 hours
        return 8  # we'll approximate by several small breaks totalling 8 (2+3+3)
    if segment_len >= 20:   # >5h
        return 3
    if segment_len >= 12:   # >3h
        return 2
    return 2

# -------------------------
# Zone helper for rule 4:
# zone 1: counters 1-10, zone 2:11-20, zone3:21-30, zone4:31-40
# choose zone with least opened counters, then highest empty counter in that zone.
# counters "opened" are those that have any positive value anywhere in counter_manning OR currently used >0
# -------------------------
def choose_counter_to_open(counter_manning: dict[int, list[int]]) -> int:
    zone_ranges = {1: range(1,11), 2: range(11,21), 3: range(21,31), 4: range(31,41)}
    opened_per_zone = {}
    for zone, rng in zone_ranges.items():
        opened = 0
        for c in rng:
            if any(v > 0 for v in counter_manning.get(c, [0]*48)):
                opened += 1
        opened_per_zone[zone] = opened
    # choose zone with least opened; tie-breaker: smallest zone id
    min_zone = min(opened_per_zone.items(), key=lambda kv: (kv[1], kv[0]))[0]
    # find highest empty counter in that zone (empty across whole shift)
    candidates = [c for c in zone_ranges[min_zone] if all(v == 0 for v in counter_manning.get(c, [0]*48))]
    if candidates:
        return max(candidates)
    # fallback: pick any highest-numbered counter that is 0 in this zone
    for c in reversed(zone_ranges[min_zone]):
        if counter_manning.get(c, [0]*48).count(0) == 48:
            return c
    # final fallback: return first counter with any zero at all
    for c, arr in counter_manning.items():
        if arr.count(0) == 48:
            return c
    # if no empty, just pick highest counter number
    return max(counter_manning.keys())

# -------------------------
# Main rostering function (heuristic greedy)
# Input:
#  - counter_manning: dict counter_no -> list[int] length 48 (current counts)
#  - sos_time: list[str] length N (each string periods separated by comma)
# Output:
#  - sos_officer_schedule_dict: dict officer_no -> list[int] (values -2,-1,0, or counter_no)
#  - updated counter_manning
# -------------------------
def roster_sos_officers(counter_manning: dict[int, list[int]], sos_time: list[str]):
    officers = add_sos_officers(sos_time)  # officer -> list
    N = len(sos_time)
    num_intervals = 48

    # Ensure counters 1..40 exist in counter_manning
    for c in range(1, 41):
        if c not in counter_manning:
            counter_manning[c] = [0] * num_intervals

    # tracking
    consecutive_assigned = {i: 0 for i in range(1, N+1)}   # current consecutive on-counter
    last_assigned_counter = {i: None for i in range(1, N+1)}  # last counter assigned (to prefer continuation)
    break_remaining = {i: 0 for i in range(1, N+1)}  # if >0, officer is forced break (0 means free)
    total_sos_segments = {i: compute_sos_segments(officers[i]) for i in officers}

    # To simplify: precompute each officer's total SOS coverage length (number of -1 slots)
    total_sos_len = {i: officers[i].count(-1) for i in officers}

    # iterate intervals sequentially
    for t in range(num_intervals):
        # 1) First, attempt to continue previous assignment where possible
        # This preserves continuity and reduces counter churn
        # Build list of officers that were assigned at t-1 and still in SOS at t
        if t > 0:
            for officer in range(1, N+1):
                if officers[officer][t] == -2:    # outside sos
                    consecutive_assigned[officer] = 0
                    last_assigned_counter[officer] = None
                    break
                # if officer was assigned at previous interval and still in SOS now and not on enforced break
                prev = officers[officer][t-1]
                if prev and prev > 0 and officers[officer][t] == -1 and break_remaining[officer] == 0:
                    counter_no = prev
                    # compute remaining_in_segment (how many intervals from t to end of this contiguous -1 block)
                    # find segment containing t
                    segs = total_sos_segments[officer]
                    rem_in_seg = 0
                    for (s,e) in segs:
                        if s <= t <= e:
                            rem_in_seg = e - t + 1
                            seg_len = e - s + 1
                            break
                    else:
                        rem_in_seg = 0
                        seg_len = 0

                    # decide max_allowed consecutive for this officer now:
                    if rem_in_seg <= 16:  # near end, allow up to 16
                        max_allowed = 16
                    else:
                        max_allowed = 10

                    # if consecutive already below max and counter free at t => continue
                    if consecutive_assigned[officer] < max_allowed and counter_manning[counter_no][t] == 0:
                        # assign same counter
                        officers[officer][t] = counter_no
                        counter_manning[counter_no][t] += 1
                        consecutive_assigned[officer] += 1
                        last_assigned_counter[officer] = counter_no
                        continue
                    else:
                        # must take break if we've hit the limit and there is still SOS left
                        if consecutive_assigned[officer] >= max_allowed and rem_in_seg > 0:
                            # compute break length heuristically
                            total_breaks = recommended_total_breaks(seg_len)
                            # choose minimal break 2 or 3, if total_breaks >=3 choose 3 else 2
                            br_len = 3 if total_breaks >= 3 else 2
                            # If not enough space left for break, allow extended run (we're near end)
                            # set break_remaining
                            break_remaining[officer] = br_len
                            # mark current t as break (if within SOS)
                            if officers[officer][t] == -1:
                                officers[officer][t] = 0
                                consecutive_assigned[officer] = 0
                            last_assigned_counter[officer] = None
                            continue
                        # else can't continue because counter occupied; leave unallocated for now (will attempt assign below)

        # decrement break_remaining where applicable for everyone before allocations
        for off in range(1, N+1):
            if break_remaining[off] > 0:
                # if this interval is in SOS and currently -1, mark as break (0)
                if officers[off][t] == -1:
                    officers[off][t] = 0
                break_remaining[off] = max(0, break_remaining[off] - 1)
                if break_remaining[off] == 0:
                    consecutive_assigned[off] = 0
                # do not attempt to assign during break
        # 2) Build list of still-unassigned officers at this t (value == -1)
        unassigned = [o for o in range(1, N+1) if officers[o][t] == -1 and break_remaining[o]==0]

        # To maximize reuse of open counters and continuity, sort unassigned by:
        #  - descending total remaining SOS (so long-duty officers prioritized)
        #  - then by officer id
        def remaining_sos(o):
            # count remaining -1 intervals from t to end
            return sum(1 for x in officers[o][t:] if x == -1)
        unassigned.sort(key=lambda o: (-remaining_sos(o), o))

        # 3) For each unassigned officer try to find an open counter (counter_manning[c][t] == 0)
        # prefer counters that were used previously by that officer (last_assigned_counter), else counters already open
        for off in unassigned:
            # re-check still -1 (may have been switched to break)
            if officers[off][t] != -1:
                continue
            # prefer previous counter if available and free now
            pref = last_assigned_counter.get(off)
            assigned = False
            if pref and counter_manning[pref][t] == 0:
                officers[off][t] = pref
                counter_manning[pref][t] += 1
                consecutive_assigned[off] += 1
                assigned = True
                last_assigned_counter[off] = pref
            if assigned:
                continue

            # search for any currently opened counter (a counter where at least one other interval has occupancy or this interval has 0)
            # we prefer counters that are already open (sum>0 over whole shift) to avoid opening new
            open_counters = [c for c, arr in counter_manning.items() if any(v > 0 for v in arr)]
            # of those, pick ones free at this interval
            candidates = [c for c in open_counters if counter_manning[c][t] == 0]
            if not candidates:
                # try any counter free at this interval
                candidates = [c for c, arr in counter_manning.items() if arr[t] == 0]
            if candidates:
                # pick candidate with lowest total future load (so we don't block heavily used counters)
                # compute future load = sum(counter_manning[c][t:]) prefer lower
                cand = min(candidates, key=lambda c: sum(counter_manning[c][t:]))
                officers[off][t] = cand
                counter_manning[cand][t] += 1
                consecutive_assigned[off] += 1
                last_assigned_counter[off] = cand
                assigned = True
            else:
                # need to open a new counter following rule 4
                new_counter = choose_counter_to_open(counter_manning)
                # ensure it's created
                if new_counter not in counter_manning:
                    counter_manning[new_counter] = [0]*num_intervals
                # assign
                officers[off][t] = new_counter
                counter_manning[new_counter][t] += 1
                consecutive_assigned[off] += 1
                last_assigned_counter[off] = new_counter
                assigned = True

            # if assigned, check consecutive limit; if after assignment we exceed allowed, schedule break next iteration
            if assigned:
                # compute segment remaining
                segs = total_sos_segments[off]
                rem_in_seg = 0
                seg_len = 0
                for (s,e) in segs:
                    if s <= t <= e:
                        rem_in_seg = e - t
                        seg_len = e - s + 1
                        break
                max_allowed = 16 if rem_in_seg <= 16 else 10
                if consecutive_assigned[off] >= max_allowed:
                    # schedule break of 2 or 3 intervals depending on segment_len
                    tot_breaks = recommended_total_breaks(seg_len)
                    br_len = 3 if tot_breaks >= 3 else 2
                    # set break_remaining to br_len for upcoming intervals
                    break_remaining[off] = br_len
                    # do not change current assignment; break will start next interval

    # end of time loop

    # convert schedules into requested dict: officer serial (1..N) -> list
    sos_officer_schedule_dict = {i: officers[i] for i in range(1, N+1)}
    return sos_officer_schedule_dict, counter_manning

def run_main():
    # sample minimal counter_manning skeleton (1..40) zeros
    counter_manning = {i: [0]*48 for i in range(1,41)}

    # sample sos_time (officer A, B, ...)
    #sos_time = ['1000-1200', '1030-1900', '1600-1800']  # simple example (you'll pass full list)
    sos_time = ['1000-1200','2000-2200','1300-1430, 2030-2200','1300-1430, 2030-2200','1300-1430, 2030-2200,1000-1130','1400-1830','1800-2030']

    schedules, updated_cm = roster_sos_officers(counter_manning, sos_time)

    # pretty print (convert numpy scalars to python ints not needed here since we used lists)
    for off_no, sched in schedules.items():
        print(f"Officer {off_no}: {sched}")
    return (schedules, updated_cm)


schedules, updated_cm = run_main()
