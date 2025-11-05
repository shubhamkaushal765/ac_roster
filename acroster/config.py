NUM_SLOTS = 48
NUM_COUNTERS = 41
START_HOUR = 10
counter_priority_list = [41] + [
    n for offset in range(0, 10) for n in range(40 - offset, 0, -10)
]
