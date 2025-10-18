from backend_algo import init_main_officers_template, generate_main_officers_schedule, officer_to_counter_matrix, find_partial_availability, build_officer_schedules, generate_break_schedules, generate_sos_schedule_matrix, greedy_smooth_schedule_beam, greedy_longest_partition_inclusive, get_intervals_from_schedule, max_coverage_paths_inclusive, fill_sos_counter_manning, prefix_non_zero
from backend_algo import find_empty_rows, slot_officers_matrix_gap_aware, merge_prefixed_matrices, plot_officer_timetable_with_labels
# === user inputs ===
main_officers_no = '1-18'
sos_timings = '1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, 1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, 1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945'
ro_ra_officers = "6RO2000"


main_officers_template = init_main_officers_template()
main_officers_schedule = generate_main_officers_schedule(main_officers_template, main_officers_no, ro_ra_officers )
counter_matrix, counter_no = officer_to_counter_matrix(main_officers_schedule)
print(main_officers_schedule)
counter_w_partial_availability = find_partial_availability(counter_matrix)
officer_names, base_schedules = build_officer_schedules(sos_timings)
all_break_schedules = generate_break_schedules(base_schedules, officer_names)
chosen_schedule_indices, best_work_count, min_penalty = greedy_smooth_schedule_beam(
    base_schedules,None,all_break_schedules,beam_width=20)
sos_schedule_matrix = generate_sos_schedule_matrix(chosen_schedule_indices, all_break_schedules, officer_names)
schedule_intervals_to_officers, schedule_intervals = get_intervals_from_schedule(sos_schedule_matrix)
chains = greedy_longest_partition_inclusive(schedule_intervals)
paths = max_coverage_paths_inclusive(chains)
print("=== best work count ===")
print(best_work_count)
# for i, path in enumerate(full_paths, 1):
#     print(f"Path {i}: {path}")

# print('===full paths===')

# for i, path in enumerate(full_paths, 1):
#     print(f"Path {i}: {path}")

# print('===partial paths===')

# for i, path in enumerate(partial_paths, 1):
#     print(f"Path {i}: {path}")

sos_counter_manning = fill_sos_counter_manning(counter_matrix, paths, schedule_intervals_to_officers)
prefixed_counter_matrix = prefix_non_zero(counter_matrix, "M")
empty_rows, partial_empty_rows, partial_empty_rows_index = find_empty_rows(counter_matrix)
prefixed_sos_counter_manning = slot_officers_matrix_gap_aware(schedule_intervals_to_officers, partial_empty_rows)
merged = merge_prefixed_matrices(prefixed_counter_matrix, prefixed_sos_counter_manning)
plot_officer_timetable_with_labels(merged)