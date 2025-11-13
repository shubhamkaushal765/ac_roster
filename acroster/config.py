NUM_SLOTS = 48
START_HOUR = 10

from enum import Enum
from typing import List

class OperationMode(Enum):
    """Operation modes for the roster system"""
    ARRIVAL = "arrival"
    DEPARTURE = "departure"

# Mode-specific configurations
MODE_CONFIG = {
    OperationMode.ARRIVAL: {
        'num_counters': 41,
        'zone1' : (0,10),
        'zone2' : (10,20),
        'zone3' : (20,30),
        'zone4' : (30,40),
        'description': 'Arrival Car (M)'
    },
    OperationMode.DEPARTURE: {
        'num_counters': 38,
        'zone1' : (0,8),
        'zone2' : (8,18),
        'zone3' : (18,28),
        'zone4' : (28,36),
        'description': 'Departure Car (N)'
    }
}

def _compute_priority_list(mode: OperationMode) -> List[int]:
    """Compute counter priority list based on operation mode."""
    
    cfg = MODE_CONFIG[mode]
    num_counters = cfg['num_counters']
    
    if mode == OperationMode.ARRIVAL:
        return [num_counters] + [
            n for offset in range(10) 
            for n in range(num_counters - 1 - offset, 0, -10)
        ]
    
    # DEPARTURE mode
    zone1 = list(range(cfg['zone1'][0], cfg['zone1'][1]))
    zone2 = list(range(cfg['zone2'][0], cfg['zone2'][1]))
    zone3 = list(range(cfg['zone3'][0], cfg['zone3'][1]))
    zone4 = list(range(cfg['zone4'][0], cfg['zone4'][1]))
    zones = [zone2, zone3, zone1, zone4]
    
    return [37] + [
        zone[i]+1 for i in range(7, -1, -1) 
        for zone in zones if i < len(zone)
    ] + [19, 20, 9, 10]


# Add computed priority lists to MODE_CONFIG
for mode in OperationMode:
    MODE_CONFIG[mode]['counter_priority_list'] = _compute_priority_list(mode)


# Default mode
# DEFAULT_MODE = OperationMode.ARRIVAL
# NUM_COUNTERS = MODE_CONFIG[DEFAULT_MODE]['num_counters']
# counter_priority_list = MODE_CONFIG[DEFAULT_MODE]['counter_priority_list']