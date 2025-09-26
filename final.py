from ortools.sat.python import cp_model

slot_length = 15
def hhmm_to_slot(hhmm: str, slot_length=slot_length, NUM_SLOTS = 48):
    """Convert 'HHMM' string to slot index. Default slot = 30 min."""
    t = int(hhmm)
    h = t // 100
    m = t % 100
    slot = (h - 10) * 4 + (m // slot_length)
    return max(0, min(NUM_SLOTS - 1, slot))

def calculate_total_break2(officer_intervals):
    officers_break_quota = {}
    for officer in officer_intervals:
        total_break = 0
        for each_interval in officer_intervals[officer]:
            s, e = each_interval
            if e - s >= 36:   # >= 9 hours
                total_break = 8
                break
            elif e - s >= 20: # >= 5 hours
                total_break = max(total_break, 3)
            elif e - s >= 10: # >= 2.5 hours
                total_break = max(total_break, 2)
        officers_break_quota[officer] = total_break
    return officers_break_quota

def build_model_inputs(input_avail: list[str], slot_length=slot_length):
    officers = [f"O{i+1}" for i in range(len(input_avail))]
    counters = [f"C{i+1}" for i in range(40)]  # 40 counters
    
    sos_availability = {}
    availability = {}
    for i, avail_str in enumerate(input_avail):
        intervals = [s.strip() for s in avail_str.split(",") if s.strip()]
        officer_intervals = []
        for interval in intervals:
            start_str, end_str = interval.split("-")
            start_slot = hhmm_to_slot(start_str, slot_length)
            end_slot = hhmm_to_slot(end_str, slot_length)
            officer_intervals.append((start_slot, end_slot))
        sos_availability[officers[i]] = officer_intervals
        availability[officers[i]] = officer_intervals
    
    break_requirements = calculate_total_break2(sos_availability)
    return officers, counters, availability, break_requirements

def run_cp_model(officers, counters, slots, sos_availability, min_total_breaks, 
                 break_lengths, max_working_slots,
                 COUNTER_CHANGE_WEIGHT, BREAK_TIME_WEIGHT, MIN_COUNTER_WEIGHT):

    model = cp_model.CpModel()
    officer_slot = {}
    is_break = {}

    # === Create variables ===
    for o in officers:
        is_break[o] = []
        for s in slots:
            # Check if slot is inside SOS interval
            in_sos = any(start <= s < end for start, end in sos_availability[o])
            if not in_sos:
                # Outside SOS: fixed value -2
                officer_slot[o, s] = -2
                is_break[o].append(None)
            else:
                # Inside SOS: decision variable
                officer_slot[o, s] = model.NewIntVar(-1, len(counters)-1, f"officer_{o}_slot_{s}")
                b = model.NewBoolVar(f"break_{o}_{s}")
                is_break[o].append(b)
                model.Add(officer_slot[o, s] == -1).OnlyEnforceIf(b)
                model.Add(officer_slot[o, s] != -1).OnlyEnforceIf(b.Not())

    # === Working slots & penalties ===
    working_slots = []
    break_timing_penalties = []
    counter_change_penalties = []
    counter_coverage_penalties = []

    for o in officers:
        # Minimum total breaks
        model.Add(sum([b for b in is_break[o] if b is not None]) >= min_total_breaks[o])

        # Maximum consecutive working slots per SOS interval
        for start_s, end_s in sos_availability[o]:
            for s_start in range(start_s, end_s - max_working_slots):
                window_breaks = [is_break[o][s_start + i] for i in range(max_working_slots + 1)]
                model.Add(sum(window_breaks) >= 1)

        # Working slots & break timing penalties
        for start_s, end_s in sos_availability[o]:
            for s in range(start_s, end_s):
                is_working = model.NewBoolVar(f"working_{o}_{s}")
                model.Add(officer_slot[o, s] != -1).OnlyEnforceIf(is_working)
                model.Add(officer_slot[o, s] == -1).OnlyEnforceIf(is_working.Not())
                working_slots.append(is_working)

                # Optional break timing near interval edges
                if s - start_s < 4 or end_s - s <= 4:
                    break_penalty = model.NewBoolVar(f"break_penalty_{o}_{s}")
                    model.Add(is_break[o][s] == 1).OnlyEnforceIf(break_penalty)
                    model.Add(is_break[o][s] == 0).OnlyEnforceIf(break_penalty.Not())
                    break_timing_penalties.append(break_penalty)

        # Counter change penalties
        for s in range(1, len(slots)):
            # Skip if any slot is outside SOS
            prev_slot = officer_slot[o, s-1]
            curr_slot = officer_slot[o, s]
            if (isinstance(prev_slot, int) and prev_slot == -2) or (isinstance(curr_slot, int) and curr_slot == -2):
                continue

            changed = model.NewBoolVar(f"{o}_changed_{s}")
            model.Add(curr_slot != prev_slot).OnlyEnforceIf(changed)
            model.Add(curr_slot == prev_slot).OnlyEnforceIf(changed.Not())

            prev_working = model.NewBoolVar(f"{o}_prev_working_{s}")
            curr_working = model.NewBoolVar(f"{o}_curr_working_{s}")
            model.Add(prev_slot != -1).OnlyEnforceIf(prev_working)
            model.Add(prev_slot == -1).OnlyEnforceIf(prev_working.Not())
            model.Add(curr_slot != -1).OnlyEnforceIf(curr_working)
            model.Add(curr_slot == -1).OnlyEnforceIf(curr_working.Not())

            both_working = model.NewBoolVar(f"{o}_both_working_{s}")
            model.AddBoolAnd([prev_working, curr_working]).OnlyEnforceIf(both_working)
            model.AddBoolOr([prev_working.Not(), curr_working.Not()]).OnlyEnforceIf(both_working.Not())

            penalty_if_working = model.NewBoolVar(f"penalty_if_working_{o}_{s}")
            model.AddBoolAnd([changed, both_working]).OnlyEnforceIf(penalty_if_working)
            model.AddBoolOr([changed.Not(), both_working.Not()]).OnlyEnforceIf(penalty_if_working.Not())
            counter_change_penalties.append(penalty_if_working)

    # === Max 1 officer per counter per slot ===
    for c_index, c in enumerate(counters):
        for s in slots:
            working_at_counter = []
            for o in officers:
                val = officer_slot[o, s]
                if isinstance(val, int) and val == -2:
                    continue
                is_at_c = model.NewBoolVar(f"{o}_at_{c}_{s}")
                model.Add(officer_slot[o, s] == c_index).OnlyEnforceIf(is_at_c)
                model.Add(officer_slot[o, s] != c_index).OnlyEnforceIf(is_at_c.Not())
                working_at_counter.append(is_at_c)
            if working_at_counter:
                model.Add(sum(working_at_counter) <= 1)

    # === Counter coverage penalties ===
    for s in slots:
        counter_open = []
        for c_index, c in enumerate(counters):
            officer_at_c = []
            for o in officers:
                val = officer_slot[o, s]
                if isinstance(val, int) and val == -2:
                    continue
                at_c = model.NewBoolVar(f"{o}_at_{c}_{s}")
                model.Add(val == c_index).OnlyEnforceIf(at_c)
                model.Add(val != c_index).OnlyEnforceIf(at_c.Not())
                officer_at_c.append(at_c)
            if officer_at_c:
                has_officer = model.NewBoolVar(f"{c}_open_{s}")
                model.AddBoolOr(officer_at_c).OnlyEnforceIf(has_officer)
                model.AddBoolAnd([x.Not() for x in officer_at_c]).OnlyEnforceIf(has_officer.Not())
                counter_open.append(has_officer)
        if counter_open:
            num_open = model.NewIntVar(0, len(counters), f"num_open_{s}")
            model.Add(num_open == sum(counter_open))
            penalty = model.NewBoolVar(f"coverage_penalty_{s}")
            model.Add(num_open < 2).OnlyEnforceIf(penalty)
            model.Add(num_open >= 2).OnlyEnforceIf(penalty.Not())
            counter_coverage_penalties.append(penalty)

    # === Objective ===
    model.Maximize(
        sum(working_slots)
        - COUNTER_CHANGE_WEIGHT * sum(counter_change_penalties)
        - BREAK_TIME_WEIGHT * sum(break_timing_penalties)
        - MIN_COUNTER_WEIGHT * sum(counter_coverage_penalties)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60 
    status = solver.Solve(model)
    return solver, status, officer_slot


def print_model_output(solver, status, officers, counters, slots, officer_slot):
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("=== Officer Schedules ===")
        print("Legend: # = break, X = outside SOS, Cn = counter number\n")

        for o in officers:
            schedule = []
            for s in slots:
                val = solver.Value(officer_slot[o, s])
                if val == -1:
                    schedule.append('#')   # break inside SOS
                elif val == -2:
                    schedule.append('X')   # outside SOS
                else:
                    schedule.append(counters[val])
            print(f"{o}: {schedule}")

        print("\n=== Counter Manning ===")
        counter_manning = {c: [] for c in counters}
        for s in slots:
            manning = {c: 0 for c in counters}
            for o in officers:
                val = solver.Value(officer_slot[o, s])
                if val != -1 and val != -2:
                    manning[counters[val]] += 1
            for c in counters:
                counter_manning[c].append(manning[c])
        for c in counters:
            print(f"{c}: {counter_manning[c]}")
    else:
        print("No feasible solution found.")


# === Example Usage ===

input_avail = [
    '1000-1200','2000-2200','1300-1430,2030-2200','1300-1430,2030-2200',
    '1300-1430,2030-2200,1000-1130','1000-1600','1000-1600','1030-1900',
    '1030-1900','1030-1900','1030-1900','1030-1900','1100-2200','1100-2200',
    '1100-2200','1200-2200','1200-2200','1145-1830','1145-2200','1200-2200',
    '1145-2200','1145-2200','1230-1400','1130-1300','1300-1430','1230-1630',
    '1600-1830','1600-1830','1400-1830','1400-1830','1000-1200','2000-2200',
    '1800-2030','1700-2200'
]

COUNTER_CHANGE_WEIGHT, BREAK_TIME_WEIGHT, MIN_COUNTER_WEIGHT = 0.5, 2, 2.5
slots = list(range(48))
break_lengths = [2, 3]
max_working_slots = 12

officers, counters, availability, break_requirements = build_model_inputs(input_avail, 15)
min_total_breaks = break_requirements

solver, status, officer_slot = run_cp_model(
    officers, counters, slots, availability, min_total_breaks,
    break_lengths, max_working_slots,
    COUNTER_CHANGE_WEIGHT, BREAK_TIME_WEIGHT, MIN_COUNTER_WEIGHT
)

print_model_output(solver, status, officers, counters, slots, officer_slot)
