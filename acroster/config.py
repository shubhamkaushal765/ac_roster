NUM_SLOTS = 48
START_HOUR = 10

from enum import Enum

class OperationMode(Enum):
    """Operation modes for the roster system"""
    ARRIVAL = "arrival"
    DEPARTURE = "departure"

# Mode-specific configurations
MODE_CONFIG = {
    OperationMode.ARRIVAL: {
        'num_counters': 41,
        'counter_range': (1, 41),
        'description': 'Arrival Hall Configuration'
    },
    OperationMode.DEPARTURE: {
        'num_counters': 38,
        'counter_range': (1, 38),
        'description': 'Departure Hall Configuration'
    }
}

# Default mode
DEFAULT_MODE = OperationMode.ARRIVAL
NUM_COUNTERS = MODE_CONFIG[DEFAULT_MODE]['num_counters']
counter_priority_list = [NUM_COUNTERS] + [
    n for offset in range(0, 10) for n in range(NUM_COUNTERS-1 - offset, 0, -10)
]

'''

if departure
zone1 = range(0, 8)
zone2 = range(8, 18)
zone3 = range(18, 28)
zone4 = range(28, 36)


zones = [list(zone2), list(zone3), list(zone1), list(zone4)]

counter_priority_list = [37] + [
    n+1
    for i in range(7, -1, -1)  # backwards
    for zone in zones
    for n in [zone[i]]
] + [19, 20, 9, 10]
print(counter_priority_list)


'''