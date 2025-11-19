"""
Streamlit web application for AC Roster Generation (Morning Shift)
Updated with raw text parser for SOS timings
"""
#app.py
import streamlit as st
import pandas as pd
import re

from acroster import Plotter
from acroster.config import NUM_SLOTS, START_HOUR, MODE_CONFIG, OperationMode
from acroster.schedule_manager import ScheduleManager
from acroster.database import get_db_instance
from acroster.db_handlers import save_last_inputs, get_last_inputs, save_roster_history

# === Raw Text Extraction Functions ===
def clean_time(t):
    """Cleans timing text by removing 'ish' and spaces."""
    t = t.lower().replace("ish", "")
    t = t.replace(" ", "")
    return t if re.match(r'\d{4}-\d{4}', t) else None

def extract_officer_timings(full_text):
    """Extract officer timings from raw text format."""
    blocks = re.split(r'\n(?=\d{2}\s*x\s*)', full_text.strip(), flags=re.IGNORECASE)
    final_records = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        base_parentheses = re.search(r'\(([^)]*?\d{4}.*?\d{4}[^)]*?)\)', block)
        if not base_parentheses:
            continue
            
        base_text = base_parentheses.group(1)
        raw_base_times = re.split(r'[/,&]', base_text)
        base_times = []
        for t in raw_base_times:
            cleaned = clean_time(t)
            if cleaned:
                base_times.append(cleaned)
        
        officer_lines = re.findall(r'(?:[-*]\s*)?([A-Za-z0-9@_ ]+(?:\([^)]*\))?)', block)
        officer_lines = [l.strip() for l in officer_lines if l.strip() and not re.match(r'\d{2}\s*x', l)]
        
        for line in officer_lines:
            name = re.sub(r'\(.*?\)', '', line).strip()
            
            extra_match = re.search(r'\(([^)]*?)\)', line)
            if extra_match:
                extra_raw = extra_match.group(1)
                extra_clean = clean_time(extra_raw)
            else:
                extra_clean = None
            
            combined_times = base_times.copy()
            if extra_clean:
                combined_times.append(extra_clean)
                
            timing_str = ";".join(combined_times)
            final_records.append({
                "name": name,
                "timing": timing_str
            })
    
    return final_records

# === Streamlit setup ===
get_db_instance()
st.set_page_config(page_title="Generate ACar/DCar Roster Morning", layout="wide")
st.title("Generate AC/DC roster (Morning)")
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
<h3>How to use</h3>

- `GL counters` is only applicable for officers in main roster with S/N `4,8,12 ...` and are not assigned a counter from `1000-1115`.
- `Handwritten counters` are counters assigned by chops room at the start of the shift, when an officer's original counter on main roster is already occupied (e.g. has OT officer). If `S/N 3` in main roster should report to `AC12` at the start of shift,  key in as `3AC12`.
- `RO/RA Officers` for `S/N 11` to `RO` at `1700` is written as `11RO1700`.
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

saved_inputs = get_last_inputs() or {}

# Main input fields
operation_mode = st.selectbox(
    "Operation Mode",
    options=[OperationMode.ARRIVAL.value, OperationMode.DEPARTURE.value],
    index=0,
    help="Select Arrival (41 counters) or Departure (38 counters)"
)

main_officers_reported = st.text_input(
    "Main Officers Reported",
    value=saved_inputs.get('main_officers', '1-18'),
    help="Enter officer numbers (e.g., 1-18 or 1,3,5-10)"
)

report_gl_counters = st.text_input(
    "GL Counters",
    value=saved_inputs.get('gl_counters', '4AC1, 8AC11, 12AC21, 16AC31'),
    help="Ground level counter assignments for officers divisible by 4"
)

handwritten_counters = st.text_input(
    "Handwritten Counters (1000-1030, 30 minutes only)",
    value=saved_inputs.get('handwritten_counters', '3AC12,5AC13'),
    help="Manual counter allocation by chops room at the start of shift"
)

OT_counters = st.text_input(
    "OT Counters (1000-1030)",
    value=saved_inputs.get('ot_counters', '2,20,40'),
    help="insert counter numbers indicated as _OT_ in HOTO list"
)

ro_ra_officers = st.text_input(
    "RO/RA Officers",
    value=saved_inputs.get('ro_ra_officers', '3RO2100, 11RO1700,15RO2130'),
    help="Officers reporting late (RA) or leaving early (RO)"
)

# === SOS Timings Input Section ===
st.markdown("---")
st.subheader("📝 SOS Timings")

# Initialize session state
if 'extracted_sos_data' not in st.session_state:
    st.session_state.extracted_sos_data = []
if 'sos_input_confirmed' not in st.session_state:
    st.session_state.sos_input_confirmed = False


st.markdown(
    f"""
    <div style="
        border: 1px solid white;
        border-radius: 8px;
        padding: 10px;
    ">
<h4>📋 Raw Text Format (Recommended) </h4>
Paste the raw text given by Ops Room. The system will automatically extract officer names and timings.

<h4> ✏️ Manual Fill </h4> \n
Example : `(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200`
1. Different officers are separated by commas `,`. \n
2. If an officer has multiple SOS timings, separate them with semicolons `;`. \n
3. Pre-assigned counters for SOS officers must be enclosed in parentheses `()`.
   Only valid if the first SOS starts at `1000`. For example, `(AC22)1000-1200`. \n
4. Time range is in 24-hour `HHMM-HHMM` format.
</div>
    """,
    unsafe_allow_html=True,
)

example_raw_text = '''ACAR SOS AM
02 x GC

officer A (1000-1200)

officer B (2000-2200)

03 x Bikes 2 (1300-1430 / 2030-2200)

officer C

officer D

officer E (1000-1130)'''

input_method = st.radio(
    "Choose input method:",
    ["📋 Raw Text (Auto-extract)", "✏️ Manual Format (Legacy)"],
    index=0,
    horizontal=True
)
if input_method == "📋 Raw Text (Auto-extract)":
    raw_sos_text = st.text_area(
        "Paste the SOS timings given by Ops Room here",
        value=saved_inputs.get('raw_sos_text', example_raw_text),
        height=200,
        help="Paste the SOS timings message from Ops Room")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        extract_button = st.button("🔍 Extract", use_container_width=True)
    with col2:
        if st.session_state.extracted_sos_data:
            reset_button = st.button("🔄 Reset", use_container_width=True)
        else:
            reset_button = False
    
    if extract_button:
        if raw_sos_text.strip():
            try:
                extracted_data = extract_officer_timings(raw_sos_text)
                if extracted_data:
                    st.session_state.extracted_sos_data = extracted_data
                    st.session_state.sos_input_confirmed = False
                    st.success(f"✅ Extracted {len(extracted_data)} officer records")
                else:
                    st.warning("⚠️ No officer data could be extracted. Please check the format.")
            except Exception as e:
                st.error(f"❌ Extraction failed: {str(e)}")
        else:
            st.warning("⚠️ Please enter raw text first")
    
    if reset_button:
        st.session_state.extracted_sos_data = []
        st.session_state.sos_input_confirmed = False
        st.rerun()
    
    # Display and edit extracted data
    if st.session_state.extracted_sos_data:
        st.markdown("### 📊 Extracted Officer Timings (Editable)")
        st.info("💡 Review and edit the extracted data below. Check the boxes to include officers in the schedule.")
        
        # Convert to DataFrame for editing
        df = pd.DataFrame(st.session_state.extracted_sos_data).reset_index(drop=True)
        
        # Add a 'selected' column if it doesn't exist
        if 'selected' not in df.columns:
            df.insert(0, 'selected', True)  # Default all to selected
        
        # Editable data editor with checkbox column
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "selected": st.column_config.CheckboxColumn(
                    "✓ Include",
                    width="small",
                    help="Check to include this officer in the schedule",
                    default=True
                ),
                "name": st.column_config.TextColumn(
                    "Officer Name", 
                    width="medium",
                    help="Name or identifier of the officer",
                    required=True
                ),
                "timing": st.column_config.TextColumn(
                    "Timing", 
                    width="large", 
                    help="Format: 1300-1430;2030-2200 (use ; for multiple ranges)",
                    required=True
                )
            },
            key="sos_data_editor"
        )
        
        # Update session state with edited data
        st.session_state.extracted_sos_data = edited_df.to_dict('records')
        
        # Show summary of selected rows
        selected_count = edited_df['selected'].sum() if 'selected' in edited_df.columns else len(edited_df)
        st.caption(f"📊 {selected_count} of {len(edited_df)} officers selected")
        
        # Confirm button
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ Confirm Data", use_container_width=True, type="primary"):
                # Filter for selected rows with valid data
                selected_df = edited_df[edited_df['selected'] == True].copy()
                
                # Validate that selected rows have both name and timing
                invalid_rows = selected_df[
                    (selected_df['name'].isna() | (selected_df['name'] == '')) |
                    (selected_df['timing'].isna() | (selected_df['timing'] == ''))
                ]
                
                if not invalid_rows.empty:
                    st.error(f"⚠️ {len(invalid_rows)} selected row(s) have missing name or timing. Please fill in all fields or uncheck these rows.")
                elif len(selected_df) == 0:
                    st.warning("⚠️ No officers selected. Please check at least one row.")
                else:
                    st.session_state.sos_input_confirmed = True
                    st.success(f"✅ {len(selected_df)} officers confirmed and ready for scheduling")
        
        # Convert selected rows to format expected by backend
        if st.session_state.sos_input_confirmed:
            selected_df = edited_df[edited_df['selected'] == True].copy()
            
            # Filter out rows with missing data
            valid_df = selected_df[
                (selected_df['name'].notna()) & (selected_df['name'] != '') &
                (selected_df['timing'].notna()) & (selected_df['timing'] != '')
            ]
            
            if len(valid_df) > 0:
                sos_timings_list = valid_df['timing'].tolist()
                sos_timings_str = ", ".join(sos_timings_list)
                st.success(f"✅ {len(sos_timings_list)} officers ready for scheduling")
            else:
                sos_timings_str = ""
                st.warning("⚠️ No valid officer data available")
        else:
            sos_timings_str = ""
    else:
        sos_timings_str = ""
else:  # Manual Format
    sos_timings = st.text_area(
        "SOS Timings (Manual Format)",
        value=saved_inputs.get('sos_timings', '(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200,1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200, 1200-2200, 1400-1830, 1400-1830, 1630-1830,1330-2200,1800-2030, 1800-2030, 1730-2200, 1730-1900, 1700-1945'),
        height=100,
        help="SOS officer timings in manual format"
    )
    sos_timings_str = sos_timings.strip()
    st.session_state.sos_input_confirmed = True  # Manual input is always confirmed

# Advanced options (collapsible)
with st.expander("⚙️ Advanced Options"):
    beam_width = st.slider(
        "Beam Search Width",
        min_value=10,
        max_value=100,
        value=saved_inputs.get('beam_width', 20),
        help="Higher values may produce better schedules but take longer"
    )

    show_debug = st.checkbox(
        "Show Debug Information",
        value=True,
        help="Display additional debugging information"
    )

# === update variables based on config.py ===
config = MODE_CONFIG[OperationMode(operation_mode)]
num_counters = config['num_counters']
counter_priority_list = config['counter_priority_list']
description = config['description']

# === Generate button ===
st.markdown("---")
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
    elif not sos_timings_str:
        st.error("⚠️ Please provide SOS timings using either method")
    elif input_method == "📋 Raw Text (Auto-extract)" and not st.session_state.sos_input_confirmed:
        st.warning("⚠️ Please confirm the extracted SOS data before generating schedule")
    else:
        try:
            # Show loading spinner
            with st.spinner("Generating schedule... Please wait."):
                # Create ScheduleManager
                manager = ScheduleManager(mode=OperationMode(operation_mode))

                # Run the roster algorithm using ScheduleManager
                results = manager.run_algorithm(
                    main_officers_reported=main_officers_validated,
                    report_gl_counters=report_gl_counters.strip(),
                    sos_timings=sos_timings_str,
                    ro_ra_officers=ro_ra_officers.strip(),
                    handwritten_counters=handwritten_counters.strip(),
                    ot_counters=OT_counters.strip(),
                )

                counter_matrix, final_counter_matrix, officer_schedule, output_text = results

            # Success message with additional info
            st.success("✅ Schedule generated successfully!")

            # Display officer counts
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

            # Show optimization penalty if available
            penalty = manager.get_optimization_penalty()
            if penalty is not None:
                st.info(f"🎯 Optimization Penalty: {penalty:.2f}")

            # Save inputs for next time
            save_inputs_dict = {
                'main_officers': main_officers_validated,
                'gl_counters': report_gl_counters.strip(),
                'handwritten_counters': handwritten_counters.strip(),
                'ot_counters': OT_counters.strip(),
                'ro_ra_officers': ro_ra_officers.strip(),
                'beam_width': beam_width
            }
            
            if input_method == "📋 Raw Text (Auto-extract)":
                save_inputs_dict['raw_sos_text'] = raw_sos_text
            else:
                save_inputs_dict['sos_timings'] = sos_timings_str
                
            save_last_inputs(save_inputs_dict)

            # Save to history
            save_roster_history(
                inputs=save_inputs_dict,
                results={
                    'optimization_penalty': penalty,
                    'main_officer_count': counts['main'],
                    'sos_officer_count': counts['sos'],
                    'ot_officer_count': counts['ot'],
                    'total_officer_count': counts['total']
                }
            )

            # Initialize plotter
            plotter = Plotter(
                num_slots=NUM_SLOTS,
                num_counters=num_counters,
                start_hour=START_HOUR
            )

            # === Display Main Counter Timetable (without SOS) ===
            st.markdown("---")
            st.subheader("📊 Timetable (Main Officers Only)")

            fig1 = plotter.plot_officer_timetable_with_labels(counter_matrix)
            st.plotly_chart(
                fig1, use_container_width=True, key="fig_counter_matrix"
            )

            st.text_area(
                "Counter Manning Statistics",
                value=output_text[0],
                height=400,
                key="stats_text1"
            )


            # === Display Final Counter Timetable (with SOS) ===
            st.markdown("---")
            st.subheader("📊 Timetable (Including SOS Officers)")

            fig2 = plotter.plot_officer_timetable_with_labels(
                final_counter_matrix
            )
            st.plotly_chart(
                fig2, use_container_width=True, key="fig_counter_matrix_w_SOS"
            )

            st.text_area(
                "Counter Manning Statistics (with SOS)",
                value=output_text[1],
                height=400,
                key="stats_text2"
            )

            # === Display Officer Schedule ===
            st.markdown("---")
            st.subheader("👮 Individual Officer Schedules")

            fig3 = plotter.plot_officer_schedule_with_labels(officer_schedule)
            st.plotly_chart(
                fig3, use_container_width=True, key="fig_officer_matrix"
            )

            # Officer schedule download button
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

                    st.write("**Counter Matrix (first 5 rows):**")
                    st.dataframe(counter_matrix[:5, :10])

                    st.write("**Final Counter Matrix (first 5 rows):**")
                    st.dataframe(final_counter_matrix[:5, :10])

                with st.expander("📊 ScheduleManager State & Officer Details"):
                    st.write("**Manager State:**")
                    st.code(str(manager))

                    st.write("**Officer Breakdown:**")

                    # Main officers
                    st.write("**Main Officers:**")
                    main_officers = manager.get_main_officers()
                    if main_officers:
                        main_officers_info = []
                        for key, officer in list(main_officers.items())[:5]:
                            main_officers_info.append(
                                {
                                    'Key': key,
                                    'ID': officer.officer_id,
                                    'Type': officer.__class__.__name__,
                                    'Non-zero slots': (officer.schedule != 0).sum()
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
                        for officer in sos_officers[:5]:
                            sos_officers_info.append(
                                {
                                    'Key': officer.officer_key,
                                    'ID': officer.officer_id,
                                    'Pre-assigned Counter': officer.pre_assigned_counter,
                                    'Break Schedules': len(officer.break_schedules),
                                    'Selected Index': officer.selected_schedule_index
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
                                    'Key': officer.officer_key,
                                    'ID': officer.officer_id,
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
                            'keys': list(export_data.keys()),
                            'officer_counts': export_data['officer_counts'],
                            'config': export_data['config'],
                            'optimization_penalty': export_data['optimization_penalty']
                        }
                    )

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
        <p>Powered by ScheduleManager OOP Architecture v3.1 🚀</p>
        <p>Now with automated raw text parsing and editable preview!</p>
        <p>For issues or suggestions, please contact your system administrator.</p>
    </div>
    """,
    unsafe_allow_html=True
)