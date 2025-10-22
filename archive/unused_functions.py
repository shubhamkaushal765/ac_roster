import numpy as np
import copy
from collections import defaultdict

def format_slots_with_sep(row, sep_every=4):
    formatted = []
    for i, x in enumerate(row):
        formatted.append(f"{x:4}" if str(x) != '0' else " .  ")
        if (i + 1) % sep_every == 0 and (i + 1) != len(row):
            formatted.append("|")  # add separator
    return ' '.join(formatted)

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

#paths = max_coverage_paths_inclusive(chains)

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

#empty_rows, partial_empty_rows, partial_empty_rows_index = find_empty_rows(main_counter_matrix)
        