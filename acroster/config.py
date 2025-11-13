NUM_SLOTS = 48
START_HOUR = 10

from enum import Enum
from typing import List
import numpy as np

def get_morning_main_roster(arr_dep):
    def add_4main_roster(full_counters):
        """Generate 4 roster patterns (a, b, c, d) from 3 counter assignments."""
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

    main_officers = {}
    # First 8 officers with predefined patterns
    main_officers[1] = (
            [41] * 6 + [0] * 2 + [30] * 7 + [0] * 3 + [20] * 9 + [0] * 3 + [
        40] * 9 + [0] + [30] * 8
    )
    main_officers[2] = (
            [30] * 8 + [0] * 2 + [20] * 8 + [0] * 3 + [41] * 9 + [0] * 3 + [
        30] * 7 + [0] + [20] * 7
    )
    main_officers[3] = (
            [20] * 10 + [0] * 2 + [41] * 9 + [0] * 3 + [30] * 9 + [0] * 3 + [
        20] * 5 + [0] + [0] * 6
    )
    main_officers[4] = (
            [0] * 5 + [0] * 1 + [40] * 6 + [0] * 2 + [30] * 10 + [0] * 3 + [
        20] * 9 + [0] * 3 + [41] * 9
    )
    main_officers[5] = (
            [40] * 6 + [0] * 2 + [9] * 7 + [0] * 3 + [29] * 9 + [0] * 3 + [
        41] * 9 + [0] + [9] * 8
    )
    main_officers[6] = (
            [9] * 8 + [0] * 2 + [29] * 8 + [0] * 3 + [40] * 9 + [0] * 3 + [
        9] * 7 + [0] + [29] * 7
    )
    main_officers[7] = (
            [29] * 10 + [0] * 2 + [40] * 9 + [0] * 3 + [9] * 9 + [0] * 3 + [
        29] * 5 + [0] + [0] * 6
    )
    main_officers[8] = (
            [0] * 5 + [0] * 1 + [41] * 6 + [0] * 2 + [9] * 10 + [0] * 3 + [
        29] * 9 + [0] * 3 + [40] * 9
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

    # Generate rosters for grouped officers
    for m_no, roster in groups:
        results = add_4main_roster(roster)
        for i, officer in enumerate(m_no):
            main_officers[officer] = results[i]

    # Convert to numpy arrays
    main_officers = {i: np.array(v) for i, v in main_officers.items()}
    if arr_dep == 'arr':
        return main_officers
    else:
        main_officers_dep = {}

        for officer_id, arr_roster in main_officers.items():
            arr = np.array(arr_roster)
            dep = np.where((arr >= 1) & (arr <= 30), arr - 2,
                np.where((arr >= 31) & (arr <= 41), arr - 4, arr))
            main_officers_dep[officer_id] = dep
        return main_officers_dep


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
        "roster_templates": get_morning_main_roster(arr_dep = 'arr'),
        'stats_label': 'ACar',
        'description': 'Arrival Car (M)'
    },
    OperationMode.DEPARTURE: {
        'num_counters': 37,
        'zone1' : (0,8),
        'zone2' : (8,18),
        'zone3' : (18,28),
        'zone4' : (28,36),
        "roster_templates": get_morning_main_roster(arr_dep = 'dep'),
        'stats_label': 'DCar',
        'description': 'Departure Car (M)'
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