"""
Streamlit web application for AC Roster Generation (Morning Shift)
Updated to use ScheduleManager class from refactored backend_algo
"""

import streamlit as st

from acroster import Plotter
from acroster.config import NUM_SLOTS, NUM_COUNTERS, START_HOUR
from acroster.schedule_manager import ScheduleManager

# === Streamlit setup ===
st.set_page_config(page_title="AC roster Morning", layout="wide")
st.title("Generate AC roster (Morning)")
st.markdown(
    "<p style='font-size:14px; color:gray; margin-top:-10px;'>💡 For better display on mobile, please enable <b>Desktop site</b> in your browser settings.</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="
        border: 1px solid white;
        border-radius: 8px;
        padding: 10px;
    ">
<h3>How to input SOS timings</h3>

1. Different officers are separated by commas `,`.
2. If an officer has multiple SOS timings, separate them with semicolons `;`.
3. Optional pre-assigned counters must be enclosed in parentheses `()` before the time. 
   Only valid if the first SOS timing starts at `1000`.
4. Times are in 24-hour `HHMM-HHMM` format.

<h4>Example</h4>

`(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200`

- Officer 1: pre-assigned counter 22 at `1000`, also has sos `2000-2200`.
- Officer 2: multiple SOS timings: `1315-1430` and `2030-2200`.
- Officer 3: pre-assigned counter 23 at `1000`, also has sos from `1315-1430` and `2030-2200`.
- Remaining officers: single shifts without pre-assigned counters.

<h3>Additional Notes</h3>

- `GL counters` is only applicable for officers in main roster with S/N 2,8,12 ...
- `Handwritten counters` for `S/N 3` in main roster to `AC12` is written as `3AC12`
- `RO/RA Officers` for `S/N 11` to `RO` at `1700` is written as `11RO1700`
    </div>
    """,
    unsafe_allow_html=True,
)

# === User Inputs ===
st.markdown("<br>", unsafe_allow_html=True)


# Input validation helper
def validate_input(value, field_name, required=False):
    """Validate input and return cleaned value"""
    if required and not value.strip():
        return None, f"⚠️ '{field_name}' is a required field."
    return value.strip(), None


# Main input fields
main_officers_reported = st.text_input(
    "Main Officers Reported",
    value="1-18",
    help="Enter officer numbers (e.g., 1-18 or 1,3,5-10)"
)

report_gl_counters = st.text_input(
    "GL Counters",
    value="4AC1, 8AC11, 12AC21, 16AC31",
    help="Ground level counter assignments for officers divisible by 4"
)

handwritten_counters = st.text_input(
    "Handwritten Counters (30mins only)",
    value="3AC12,5AC13",
    help="Manual counter assignments for first 2 slots (e.g., 3AC12)"
)

OT_counters = st.text_input(
    "OT Counters (30mins only)",
    value="2,20,40",
    help="Overtime counter numbers (comma-separated)"
)

ro_ra_officers = st.text_input(
    "RO/RA Officers",
    value="3RO2100, 11RO1700,15RO2130",
    help="Officers reporting late (RA) or leaving early (RO)"
)

sos_timings = st.text_area(
    "SOS Timings",
    value="(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, 1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945",
    height=100,
    help="SOS officer timings (see instructions above)"
)

# Advanced options (collapsible)
with st.expander("⚙️ Advanced Options"):
    beam_width = st.slider(
        "Beam Search Width",
        min_value=10,
        max_value=100,
        value=20,
        help="Higher values may produce better schedules but take longer"
    )

    show_debug = st.checkbox(
        "Show Debug Information",
        value=False,
        help="Display additional debugging information"
    )

# === Generate button ===
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button(
        "🚀 Generate Schedule", use_container_width=True, type="primary"
    )

if generate_button:
    # Validate required fields
    main_officers_validated, error = validate_input(
        main_officers_reported,
        "Main Officers Reported",
        required=True
    )

    if error:
        st.error(error)
    else:
        try:
            # Show loading spinner
            with st.spinner("Generating schedule... Please wait."):
                # ========== NEW: Create ScheduleManager ==========
                manager = ScheduleManager()

                # Run the roster algorithm using ScheduleManager
                results = manager.run_algorithm(
                    main_officers_reported=main_officers_validated,
                    report_gl_counters=report_gl_counters.strip(),
                    sos_timings=sos_timings.strip(),
                    ro_ra_officers=ro_ra_officers.strip(),
                    handwritten_counters=handwritten_counters.strip(),
                    ot_counters=OT_counters.strip(),
                )

                counter_matrix, final_counter_matrix, officer_schedule, output_text = results

            # Success message with additional info
            st.success("✅ Schedule generated successfully!")

            # ========== NEW: Display officer counts ==========
            counts = manager.get_all_officers_count()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👮 Main Officers", counts['main'])
            with col2:
                st.metric("🆘 SOS Officers", counts['sos'])
            with col3:
                st.metric("⏰ OT Officers", counts['ot'])
            with col4:
                st.metric("📊 Total Officers", counts['total'])

            # ========== NEW: Show optimization penalty if available ==========
            penalty = manager.get_optimization_penalty()
            if penalty is not None:
                st.info(f"🎯 Optimization Penalty: {penalty:.2f}")

            # Initialize plotter
            plotter = Plotter(
                num_slots=NUM_SLOTS,
                num_counters=NUM_COUNTERS,
                start_hour=START_HOUR
            )

            # === Display Main Counter Timetable (without SOS) ===
            st.markdown("---")
            st.subheader("📊 Counter Timetable (Main Officers Only)")

            fig1 = plotter.plot_officer_timetable_with_labels(counter_matrix)
            st.plotly_chart(
                fig1, use_container_width=True, key="fig_counter_matrix"
            )

            # Display statistics in two columns
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text_area(
                    "Counter Manning Statistics",
                    value=output_text[0],
                    height=400,
                    key="stats_text1"
                )
            with col2:
                st.download_button(
                    label="📥 Download Statistics",
                    data=output_text[0],
                    file_name="counter_manning_main.txt",
                    mime="text/plain",
                    key="download_stats1",
                    use_container_width=True
                )

            # === Display Final Counter Timetable (with SOS) ===
            st.markdown("---")
            st.subheader("📊 Counter Timetable (Including SOS Officers)")

            fig2 = plotter.plot_officer_timetable_with_labels(
                final_counter_matrix
            )
            st.plotly_chart(
                fig2, use_container_width=True, key="fig_counter_matrix_w_SOS"
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                st.text_area(
                    "Counter Manning Statistics (with SOS)",
                    value=output_text[1],
                    height=400,
                    key="stats_text2"
                )
            with col2:
                st.download_button(
                    label="📥 Download Statistics",
                    data=output_text[1],
                    file_name="counter_manning_with_sos.txt",
                    mime="text/plain",
                    key="download_stats2",
                    use_container_width=True
                )

            # === Display Officer Schedule ===
            st.markdown("---")
            st.subheader("👮 Individual Officer Schedules")

            fig3 = plotter.plot_officer_schedule_with_labels(officer_schedule)
            st.plotly_chart(
                fig3, use_container_width=True, key="fig_officer_matrix"
            )

            # Officer schedule download button
            # Convert officer_schedule dict to readable text format
            officer_schedule_text = "Officer Schedules\n" + "=" * 50 + "\n\n"
            for officer_key, schedule in officer_schedule.items():
                officer_schedule_text += f"{officer_key}:\n"
                schedule_str = ", ".join(
                    [str(c) if c != 0 else "-" for c in schedule]
                )
                officer_schedule_text += f"  {schedule_str}\n\n"

            st.download_button(
                label="📥 Download Officer Schedules",
                data=officer_schedule_text,
                file_name="officer_schedules.txt",
                mime="text/plain",
                key="download_officer_schedules",
                use_container_width=False
            )

            # === Debug Information (if enabled) ===
            if show_debug:
                st.markdown("---")
                st.subheader("🔍 Debug Information")

                with st.expander("View Raw Data"):
                    st.write("**Counter Matrix Shape:**", counter_matrix.shape)
                    st.write(
                        "**Final Counter Matrix Shape:**",
                        final_counter_matrix.shape
                    )
                    st.write("**Number of Officers:**", len(officer_schedule))
                    st.write(
                        "**Officer Keys:**", list(officer_schedule.keys())
                    )

                    # Show first few rows of matrices
                    st.write("**Counter Matrix (first 5 rows):**")
                    st.dataframe(counter_matrix[:5, :10])

                    st.write("**Final Counter Matrix (first 5 rows):**")
                    st.dataframe(final_counter_matrix[:5, :10])

                # ========== NEW: Enhanced debug info using ScheduleManager ==========
                with st.expander("📊 ScheduleManager State & Officer Details"):
                    st.write("**Manager State:**")
                    st.code(str(manager))

                    st.write("**Officer Breakdown:**")

                    # Main officers
                    st.write("**Main Officers:**")
                    main_officers = manager.get_main_officers()
                    if main_officers:
                        main_officers_info = []
                        for key, officer in list(main_officers.items())[
                                            :5]:  # Show first 5
                            main_officers_info.append(
                                {
                                    'Key':            key,
                                    'ID':             officer.officer_id,
                                    'Type':           officer.__class__.__name__,
                                    'Non-zero slots': (
                                                              officer.schedule != 0).sum()
                                }
                            )
                        st.dataframe(main_officers_info)
                        if len(main_officers) > 5:
                            st.caption(
                                f"Showing 5 of {len(main_officers)} main officers"
                            )

                    # SOS officers
                    st.write("**SOS Officers:**")
                    sos_officers = manager.get_sos_officers()
                    if sos_officers:
                        sos_officers_info = []
                        for officer in sos_officers[:5]:  # Show first 5
                            sos_officers_info.append(
                                {
                                    'Key':                  officer.officer_key,
                                    'ID':                   officer.officer_id,
                                    'Pre-assigned Counter': officer.pre_assigned_counter,
                                    'Break Schedules':      len(
                                        officer.break_schedules
                                    ),
                                    'Selected Index':       officer.selected_schedule_index
                                }
                            )
                        st.dataframe(sos_officers_info)
                        if len(sos_officers) > 5:
                            st.caption(
                                f"Showing 5 of {len(sos_officers)} SOS officers"
                            )
                    else:
                        st.info("No SOS officers in this schedule")

                    # OT officers
                    st.write("**OT Officers:**")
                    ot_officers = manager.get_ot_officers()
                    if ot_officers:
                        ot_officers_info = []
                        for officer in ot_officers:
                            ot_officers_info.append(
                                {
                                    'Key':     officer.officer_key,
                                    'ID':      officer.officer_id,
                                    'Counter': officer.counter_no
                                }
                            )
                        st.dataframe(ot_officers_info)
                    else:
                        st.info("No OT officers in this schedule")

                    # Export data structure
                    st.write("**Export Data Structure:**")
                    export_data = manager.export_schedules_to_dict()
                    st.json(
                        {
                            'keys':                 list(export_data.keys()),
                            'officer_counts':       export_data[
                                                        'officer_counts'],
                            'config':               export_data['config'],
                            'optimization_penalty': export_data[
                                                        'optimization_penalty']
                        }
                    )

                # Additional OOP info
                st.info(
                    "✨ Using ScheduleManager class with OOP architecture | "
                    "Counter and CounterMatrix classes | "
                    "Enhanced state management and debugging capabilities"
                )

        except Exception as e:
            st.error(
                f"❌ An error occurred while generating the schedule: {str(e)}"
            )

            if show_debug:
                st.exception(e)
            else:
                st.info(
                    "💡 Enable 'Show Debug Information' in Advanced Options for more details."
                )

# === Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        <p>AC Roster Generator | Morning Shift Planning Tool</p>
        <p>Powered by ScheduleManager OOP Architecture v3.0 🚀</p>
        <p>For issues or suggestions, please contact your system administrator.</p>
    </div>
    """,
    unsafe_allow_html=True
)
