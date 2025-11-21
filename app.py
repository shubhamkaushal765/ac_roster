"""
Streamlit web application for AC Roster Generation (Morning Shift)
Updated with restructured UI flow: SOS officers added via roster editor
"""
#app.py
import streamlit as st
import pandas as pd
import numpy as np
import re

from acroster import Plotter
from acroster.config import NUM_SLOTS, START_HOUR, MODE_CONFIG, OperationMode
from acroster.schedule_manager import ScheduleManager
from acroster.database import get_db_instance
from acroster.db_handlers import (
    save_last_inputs, get_last_inputs, save_roster_history,
    save_roster_edit, get_roster_edits
)
from backend_algo import hhmm_to_slot, slot_to_hhmm, generate_time_slots

def get_slot_end_time(slot_idx: int) -> str:
    """Get the end time of a slot (start time + 15 minutes)"""
    return slot_to_hhmm(slot_idx + 1)
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

# === Roster Editor Helper Functions ===
# def time_to_slot(time_str: str) -> int:
#     """Convert time string (HH:MM) to slot index (0-47)"""
#     time_slots = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(48)]
#     return time_slots.index(time_str)

def get_all_officer_ids(officer_schedule: dict) -> list:
    """Extract all officer IDs from the schedule"""
    return list(officer_schedule.keys())

def schedule_to_matrix(officer_schedule: dict) -> np.ndarray:
    """Convert officer schedule dictionary to numpy matrix"""
    officer_ids = get_all_officer_ids(officer_schedule)
    matrix = np.zeros((len(officer_ids), 48), dtype=int)
    for i, officer_id in enumerate(officer_ids):
        matrix[i, :] = officer_schedule[officer_id]
    return matrix

def matrix_to_schedule(matrix: np.ndarray, officer_ids: list) -> dict:
    """Convert numpy matrix back to officer schedule dictionary"""
    return {officer_ids[i]: matrix[i, :] for i in range(len(officer_ids))}

# === Streamlit setup ===
# === Initialize Edit History at Page Load ===
if 'edit_history' not in st.session_state:
    st.session_state['edit_history'] = []
if 'schedule_initialized' not in st.session_state:
    st.session_state['schedule_initialized'] = False
if 'confirm_reset' not in st.session_state:
    st.session_state['confirm_reset'] = False
if 'confirm_delete' not in st.session_state:
    st.session_state['confirm_delete'] = False

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
- `Handwritten counters` are counters assigned by chops room at the start of the shift as their original counter on main roster is already occupied (e.g. has OT officer). If `S/N 3` in main roster should report to `AC12` at the start of shift,  key in as as `3AC12`
- `RO/RA Officers` for `S/N 11` to `RO` at `1700` is written as `11RO1700`
- **SOS officers can be added after generating the initial schedule using the Roster Editor below**
    </div>
    """,
    unsafe_allow_html=True,
)

# === Edit History Sidebar (Always Visible) ===
with st.sidebar:
    st.subheader("📝 Edit History")
    if st.session_state['edit_history']:
        for i, edit_entry in enumerate(reversed(st.session_state['edit_history'][-20:]), 1):
            # Handle legacy string format (backwards compatibility)
            if isinstance(edit_entry, str):
                description = edit_entry
                is_legacy = True
            else:
                description = edit_entry.get('description', 'Unknown edit')
                is_legacy = False
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"{len(st.session_state['edit_history']) - i + 1}. {description}")
            with col2:
                if not is_legacy:  # Only show delete button for new format
                    # Delete button
                    if st.button("🗑️", key=f"delete_btn_{len(st.session_state['edit_history']) - i}", help="Delete this entry"):
                        if edit_entry.get('type') == 'initialize':
                            # Delete initialization = clear all fields and reset
                            st.session_state['confirm_reset'] = True
                            st.session_state['reset_index'] = len(st.session_state['edit_history']) - i
                        else:
                            # Delete this specific edit and revert
                            st.session_state['confirm_delete'] = True
                            st.session_state['delete_index'] = len(st.session_state['edit_history']) - i
                            st.session_state['delete_entry'] = edit_entry
        else:
            st.info("No history yet. Generate a schedule to begin.")
        
        st.divider()
    
    # Quick stats if schedule exists
    if 'generated_schedule' in st.session_state:
        counts = st.session_state.get('schedule_manager', None)
        if counts:
            counts_dict = counts.get_all_officers_count()
            st.metric("Total Officers", counts_dict['total'])
            st.metric("SOS Officers", counts_dict['sos'])

# === Handle Delete Confirmations ===
if st.session_state.get('confirm_reset', False):
    st.warning("⚠️ Delete 'Initialized Schedule'?")
    st.write("This will clear all schedules and reset all input fields. You'll start fresh.")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Yes, Reset All", type="primary", use_container_width=True):
            # Clear all session state related to schedules
            keys_to_clear = [
                'generated_schedule', 'counter_matrix', 'final_counter_matrix', 
                'output_text', 'roster_history_id', 'schedule_manager', 
                'edited_schedule', 'edit_history', 'schedule_initialized',
                'sos_extracted_data', 'sos_confirmed', 'saved_inputs',
                'confirm_reset', 'reset_index'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Also clear saved inputs from database
            save_last_inputs({})
            
            st.success("✅ All cleared! Starting fresh.")
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state['confirm_reset'] = False
            st.rerun()

if st.session_state.get('confirm_delete', False):
    delete_entry = st.session_state.get('delete_entry')
    st.warning(f"⚠️ Delete edit: {delete_entry['description']}?")
    st.write("This will remove this edit and revert to the previous state.")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Yes, Delete", type="primary", use_container_width=True, key="confirm_delete_btn"):
            delete_idx = st.session_state.get('delete_index')
            
            # Remove the edit from history
            if delete_idx < len(st.session_state['edit_history']):
                st.session_state['edit_history'].pop(delete_idx)
            
            # Rebuild schedule from scratch based on remaining history
            # TODO: Implement replay logic if needed, or just remove from display
            
            st.session_state['confirm_delete'] = False
            st.session_state['delete_index'] = None
            st.session_state['delete_entry'] = None
            st.success("✅ Edit deleted")
            st.rerun()
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True, key="cancel_delete_btn"):
            st.session_state['confirm_delete'] = False
            st.session_state['delete_index'] = None
            st.session_state['delete_entry'] = None
            st.rerun()

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
    else:
        try:
            # Show loading spinner
            with st.spinner("Generating schedule... Please wait."):
                # Create ScheduleManager
                manager = ScheduleManager(mode=OperationMode(operation_mode))

                # Run the roster algorithm WITHOUT SOS officers initially
                results = manager.run_algorithm(
                    main_officers_reported=main_officers_validated,
                    report_gl_counters=report_gl_counters.strip(),
                    sos_timings="",  # Empty SOS timings initially
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
            
            save_last_inputs(save_inputs_dict)

            # Save to history
            roster_history_id = save_roster_history(
                inputs=save_inputs_dict,
                results={
                    'optimization_penalty': penalty,
                    'main_officer_count': counts['main'],
                    'sos_officer_count': counts['sos'],
                    'ot_officer_count': counts['ot'],
                    'total_officer_count': counts['total']
                }
            )

            # Store in session state for roster editor
            st.session_state['generated_schedule'] = officer_schedule
            st.session_state['counter_matrix'] = counter_matrix
            st.session_state['final_counter_matrix'] = final_counter_matrix
            st.session_state['output_text'] = output_text
            st.session_state['roster_history_id'] = roster_history_id
            st.session_state['schedule_manager'] = manager
            st.session_state['saved_inputs'] = save_inputs_dict

            # Initialize edited_schedule
            if 'edited_schedule' not in st.session_state:
                st.session_state['edited_schedule'] = officer_schedule.copy()
            
            # Add "Initialized Schedule" to edit history (only once)
            if not st.session_state.get('schedule_initialized', False):
                st.session_state['edit_history'] = [{
                    'type': 'initialize',
                    'description': '🚀 Initialized Schedule',
                    'timestamp': pd.Timestamp.now(),
                    'data': {
                        'inputs': save_inputs_dict,
                        'roster_history_id': roster_history_id
                    }
                }]
                st.session_state['schedule_initialized'] = True

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

            # === Display Final Counter Timetable (initially same as main) ===
            st.markdown("---")
            st.subheader("📊 Timetable (Including SOS Officers)")
            st.info("ℹ️ No SOS officers added yet. Use the Roster Editor below to add SOS officers.")

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

# === ROSTER EDITOR SECTION ===
# Only show if a schedule has been generated
if 'generated_schedule' in st.session_state and st.session_state['generated_schedule']:
    st.markdown("---")
    st.markdown("---")
    st.title("🗓️ Roster Editor")
    st.markdown("Add SOS officers or make manual adjustments to the generated roster")
    
    # Initialize editor state
    if 'edited_schedule' not in st.session_state:
        st.session_state['edited_schedule'] = st.session_state['generated_schedule'].copy()
    if 'edit_history' not in st.session_state:
        st.session_state['edit_history'] = []
    
    officer_schedule = st.session_state['edited_schedule']
    officer_ids = sorted(get_all_officer_ids(officer_schedule))  # Sort to show M, S, OT officers
    time_slots = generate_time_slots(START_HOUR, NUM_SLOTS)

    # Convert to matrix for display
    roster_matrix = schedule_to_matrix(officer_schedule)
    
    # Edit operations - go straight to tabs
    st.subheader("Edit Operations")
    
    tab1, tab2, tab3 = st.tabs(["➕ Add SOS Officers", "🔄 Swap", "🗑️ Delete"])
    
    # TAB 1: ADD SOS OFFICERS
    with tab1:
        
        # Initialize SOS session state if not exists
        if 'sos_extracted_data' not in st.session_state:
            st.session_state.sos_extracted_data = []
        if 'sos_confirmed' not in st.session_state:
            st.session_state.sos_confirmed = False
        
        st.markdown(
            """
            <div style="border: 1px solid white; border-radius: 8px; padding: 10px;">
            <h4>📋 Raw Text Format (Recommended)</h4>
            Paste the raw text given by Ops Room. The system will automatically extract officer names and timings.
            
            <h4>✏️ Manual Fill</h4>
            Example: <code>(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200</code>
            <ol>
            <li>Different officers are separated by commas <code>,</code></li>
            <li>If an officer has multiple SOS timings, separate them with semicolons <code>;</code></li>
            <li>Optional pre-assigned counters must be enclosed in parentheses <code>()</code> before the time</li>
            <li>Time ranges are in 24-hour <code>HHMM-HHMM</code> format</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        example_raw_text = '''ACAR SOS AM
02 x GC

officer A (1000-1200)

officer B (2000-2200)

03 x Bikes 2 (1300-1430 / 2030-2200)

officer C

officer D

officer E (1000-1130)'''
        
        input_method_add = st.radio(
            "Choose input method:",
            ["📋 Raw Text (Auto-extract)", "✏️ Manual Format (Legacy)"],
            index=0,
            horizontal=True,
            key="add_sos_method"
        )
        
        if input_method_add == "📋 Raw Text (Auto-extract)":
            raw_sos_text_add = st.text_area(
                "Paste the SOS timings msg from Ops Room",
                value=example_raw_text,
                height=200,
                help="Paste the SOS timings msg from Ops Room",
                key="raw_sos_add"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                extract_button_add = st.button("🔍 Extract", use_container_width=True, key="extract_add")
            with col2:
                if st.session_state.sos_extracted_data:
                    reset_button_add = st.button("🔄 Reset", use_container_width=True, key="reset_add")
                else:
                    reset_button_add = False
            
            if extract_button_add:
                if raw_sos_text_add.strip():
                    try:
                        extracted_data = extract_officer_timings(raw_sos_text_add)
                        if extracted_data:
                            st.session_state.sos_extracted_data = extracted_data
                            st.session_state.sos_confirmed = False
                            st.success(f"✅ Extracted {len(extracted_data)} officer records")
                        else:
                            st.warning("⚠️ No officer data could be extracted. Please check the format.")
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {str(e)}")
                else:
                    st.warning("⚠️ Please enter raw text first")
            
            if reset_button_add:
                st.session_state.sos_extracted_data = []
                st.session_state.sos_confirmed = False
                st.rerun()
            
            # Display and edit extracted data
            if st.session_state.sos_extracted_data:
                st.markdown("### 📊 Extracted Officer Timings (Editable)")
                st.info("💡 Review and edit the extracted data below. Check the boxes to include officers in the schedule.")
                
                # Convert to DataFrame for editing
                df_sos = pd.DataFrame(st.session_state.sos_extracted_data).reset_index(drop=True)
                
                # Add a 'selected' column if it doesn't exist
                if 'selected' not in df_sos.columns:
                    df_sos.insert(0, 'selected', True)
                
                # Editable data editor with checkbox column
                edited_df_sos = st.data_editor(
                    df_sos,
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
                    key="sos_data_editor_add"
                )
                
                # Update session state with edited data
                st.session_state.sos_extracted_data = edited_df_sos.to_dict('records')
                
                # Show summary of selected rows
                selected_count = edited_df_sos['selected'].sum() if 'selected' in edited_df_sos.columns else len(edited_df_sos)
                st.caption(f"📊 {selected_count} of {len(edited_df_sos)} officers selected")
                
                # Add to roster button
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    if st.button("✅ Add to Roster", use_container_width=True, type="primary", key="add_sos_to_roster"):
                        # Filter for selected rows with valid data
                        selected_df = edited_df_sos[edited_df_sos['selected'] == True].copy()
                        
                        # Validate that selected rows have both name and timing
                        invalid_rows = selected_df[
                            (selected_df['name'].isna() | (selected_df['name'] == '')) |
                            (selected_df['timing'].isna() | (selected_df['timing'] == ''))
                        ]
                        
                        if not invalid_rows.empty:
                            st.error(f"⚠️ {len(invalid_rows)} selected row(s) have missing name or timing.")
                        elif len(selected_df) == 0:
                            st.warning("⚠️ No officers selected. Please check at least one row.")
                        else:
                            # Convert to format expected by backend
                            valid_df = selected_df[
                                (selected_df['name'].notna()) & (selected_df['name'] != '') &
                                (selected_df['timing'].notna()) & (selected_df['timing'] != '')
                            ]
                            
                            if len(valid_df) > 0:
                                sos_timings_list = valid_df['timing'].tolist()
                                sos_timings_str_add = ", ".join(sos_timings_list)
                                
                                # Use the schedule manager to add SOS officers
                                try:
                                    with st.spinner("Adding SOS officers to roster..."):
                                        # Create a fresh manager
                                        manager = ScheduleManager(mode=OperationMode(operation_mode))
                                        
                                        # Get saved inputs
                                        saved_inputs_data = st.session_state.get('saved_inputs', {})
                                        
                                        # Re-run algorithm with SOS officers
                                        results = manager.run_algorithm(
                                            main_officers_reported=saved_inputs_data.get('main_officers', main_officers_reported),
                                            report_gl_counters=saved_inputs_data.get('gl_counters', report_gl_counters),
                                            sos_timings=sos_timings_str_add,
                                            ro_ra_officers=saved_inputs_data.get('ro_ra_officers', ro_ra_officers),
                                            handwritten_counters=saved_inputs_data.get('handwritten_counters', handwritten_counters),
                                            ot_counters=saved_inputs_data.get('ot_counters', OT_counters),
                                        )
                                        
                                        counter_matrix, final_counter_matrix, officer_schedule, output_text = results
                                        
                                        # Update session state with INCREMENTAL addition
                                        st.session_state['edited_schedule'] = officer_schedule
                                        st.session_state['final_counter_matrix'] = final_counter_matrix
                                        st.session_state['output_text'] = output_text
                                        st.session_state['schedule_manager'] = manager
                                        
                                        # Add to edit history with detailed info
                                        edit_entry = {
                                            'type': 'add_sos',
                                            'description': f"Added {len(sos_timings_list)} SOS officers: {', '.join(valid_df['name'].tolist()[:3])}{'...' if len(valid_df) > 3 else ''}",
                                            'timestamp': pd.Timestamp.now(),
                                            'data': {
                                                'sos_count': len(sos_timings_list),
                                                'sos_names': valid_df['name'].tolist(),
                                                'sos_timings': sos_timings_str_add
                                            }
                                        }
                                        st.session_state['edit_history'].append(edit_entry)
                                        
                                        # Save to database
                                        save_roster_edit(
                                            edit_type='add_sos',
                                            officer_id=f"{len(sos_timings_list)} SOS officers",
                                            slot_start=0,  # Not applicable for bulk SOS add
                                            slot_end=47,   # Not applicable for bulk SOS add
                                            time_start="10:00",
                                            time_end="22:00",
                                            roster_history_id=st.session_state.get('roster_history_id'),
                                            notes=f"Added {len(sos_timings_list)} SOS officers: {', '.join(valid_df['name'].tolist())}"
                                        )
                                        
                                        st.success(f"✅ Added {len(sos_timings_list)} SOS officers to roster")
                                        st.session_state.sos_extracted_data = []
                                        st.session_state.sos_confirmed = False
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error adding SOS officers: {str(e)}")
                                    if show_debug:
                                        st.exception(e)
            
        else:  # Manual Format
            sos_timings_manual = st.text_area(
                "SOS Timings (Manual Format)",
                value='(AC22)1000-1300, 1315-1430;2030-2200',
                height=100,
                help="SOS officer timings in manual format",
                key="manual_sos_add"
            )
            
            if st.button("✅ Add to Roster", use_container_width=True, type="primary", key="add_manual_sos"):
                if sos_timings_manual.strip():
                    try:
                        with st.spinner("Adding SOS officers to roster..."):
                            # Create a fresh manager
                            manager = ScheduleManager(mode=OperationMode(operation_mode))
                            
                            # Get saved inputs
                            saved_inputs_data = st.session_state.get('saved_inputs', {})
                            
                            # Re-run algorithm with SOS officers
                            results = manager.run_algorithm(
                                main_officers_reported=saved_inputs_data.get('main_officers', main_officers_reported),
                                report_gl_counters=saved_inputs_data.get('gl_counters', report_gl_counters),
                                sos_timings=sos_timings_manual.strip(),
                                ro_ra_officers=saved_inputs_data.get('ro_ra_officers', ro_ra_officers),
                                handwritten_counters=saved_inputs_data.get('handwritten_counters', handwritten_counters),
                                ot_counters=saved_inputs_data.get('ot_counters', OT_counters),
                            )
                            
                            counter_matrix, final_counter_matrix, officer_schedule, output_text = results
                            
                            # Update session state
                            st.session_state['edited_schedule'] = officer_schedule
                            st.session_state['final_counter_matrix'] = final_counter_matrix
                            st.session_state['output_text'] = output_text
                            st.session_state['schedule_manager'] = manager
                            
                            # Add to edit history
                            st.session_state['edit_history'].append(
                                f"Added SOS officers (manual)"
                            )
                            
                            # Save to database
                            save_roster_edit(
                                edit_type='add_sos',
                                officer_id=f"{len(sos_timings_list)} SOS officers",
                                slot_start=0,  # Not applicable for bulk SOS add
                                slot_end=47,   # Not applicable for bulk SOS add
                                time_start="10:00",
                                time_end="22:00",
                                roster_history_id=st.session_state.get('roster_history_id'),
                                notes=f"Added {len(sos_timings_list)} SOS officers: {', '.join(valid_df['name'].tolist())}"
                            )
                            
                            st.success(f"✅ Added SOS officers to roster")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding SOS officers: {str(e)}")
                        if show_debug:
                            st.exception(e)
                else:
                    st.warning("⚠️ Please enter SOS timings first")
    
    # TAB 2: SWAP
    with tab2:
        st.write("Swap counter assignments between two officers")
        # Refresh officer IDs to include all current officers
        officer_ids = sorted(get_all_officer_ids(st.session_state['edited_schedule']))
        col1, col2 = st.columns(2)
        with col1:
            swap_officer1 = st.selectbox("Officer 1", officer_ids, key="swap_off1")
        with col2:
            swap_officer2 = st.selectbox("Officer 2", officer_ids, 
                                         index=min(1, len(officer_ids)-1),
                                         key="swap_off2")
        
        col3, col4 = st.columns(2)
        with col3:
            swap_start = st.selectbox("From Time", time_slots, key="swap_start")
        with col4:
            swap_end = st.selectbox("To Time", time_slots,
                                    index=min(47, hhmm_to_slot(swap_start) + 3),
                                    key="swap_end")
        
        if st.button("Swap Assignments", type="primary", key="swap_btn"):
            if swap_officer1 == swap_officer2:
                st.error("❌ Cannot swap officer with themselves")
            else:
                officer1_idx = officer_ids.index(swap_officer1)
                officer2_idx = officer_ids.index(swap_officer2)
                start_idx = hhmm_to_slot(swap_start)
                end_idx = hhmm_to_slot(swap_end)
                if start_idx <= end_idx:
                    # Perform swap operation
                    temp = roster_matrix[officer1_idx, start_idx:end_idx+1].copy()
                    roster_matrix[officer1_idx, start_idx:end_idx+1] = roster_matrix[officer2_idx, start_idx:end_idx+1]
                    roster_matrix[officer2_idx, start_idx:end_idx+1] = temp
                    
                    st.session_state['edited_schedule'] = matrix_to_schedule(roster_matrix, officer_ids)
                    
                    # Save to database
                    save_roster_edit(
                        edit_type='swap',
                        officer_id=swap_officer1,
                        officer_id_2=swap_officer2,
                        slot_start=start_idx,
                        slot_end=end_idx,
                        time_start=swap_start,
                        time_end=swap_end,
                        roster_history_id=st.session_state.get('roster_history_id'),
                        notes=f"Swapped {swap_officer1} ↔ {swap_officer2} from {swap_start} to {swap_end}"
                    )


                    
                    # Add to edit history with structured data
                    edit_entry = {
                        'type': 'swap',
                        'description': f"Swapped {swap_officer1} ↔ {swap_officer2}: {swap_start}-{swap_end}",
                        'timestamp': pd.Timestamp.now(),
                        'data': {
                            'officer1': swap_officer1,
                            'officer2': swap_officer2,
                            'start_time': swap_start,
                            'end_time': get_slot_end_time(end_idx - 1),
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        }
                    }
                    st.session_state['edit_history'].append(edit_entry)
                else:
                    st.error("❌ End time must be after start time")
    
    # TAB 3: DELETE
    with tab3:
        st.write("Remove officer assignment from specified time range")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            del_officer = st.selectbox("Officer", officer_ids, key="del_officer")
        with col2:
            del_start = st.selectbox("From Time", time_slots, key="del_start")
        with col3:
            del_end = st.selectbox("To Time", time_slots, 
                                   index=min(47, hhmm_to_slot(del_start) + 3),
                                   key="del_end")
        
        if st.button("Delete Assignment", type="primary", key="del_btn"):
            officer_idx = officer_ids.index(del_officer)
            start_idx = hhmm_to_slot(del_start)
            end_idx = hhmm_to_slot(del_end)
            
            if start_idx <= end_idx:
                # Perform delete operation
                roster_matrix[officer_idx, start_idx:end_idx+1] = 0
                st.session_state['edited_schedule'] = matrix_to_schedule(roster_matrix, officer_ids)
                
                # Save to database
                save_roster_edit(
                    edit_type='delete',
                    officer_id=del_officer,
                    slot_start=start_idx,
                    slot_end=end_idx,
                    time_start=del_start,
                    time_end=del_end,
                    roster_history_id=st.session_state.get('roster_history_id'),
                    notes=f"Deleted {del_officer} from {del_start} to {del_end}"
                )
                
                # Add to edit history with structured data
                edit_entry = {
                    'type': 'delete',
                    'description': f"Deleted {del_officer}: {del_start}-{del_end}",
                    'timestamp': pd.Timestamp.now(),
                    'data': {
                        'officer': del_officer,
                        'start_time': del_start,
                        'end_time': get_slot_end_time(end_idx - 1),
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    }
                }
                st.session_state['edit_history'].append(edit_entry)
            else:
                st.error("❌ End time must be after start time")
    
    # Display updated visualizations after edits
    st.markdown("---")
    st.subheader("📊 Updated Timetables (After Edits)")
    
    # Re-initialize plotter for edited schedule
    plotter = Plotter(
        num_slots=NUM_SLOTS,
        num_counters=num_counters,
        start_hour=START_HOUR
    )
    
    # Show Final Counter Timetable (Graph B - with SOS)
    if 'final_counter_matrix' in st.session_state:
        st.markdown("### 📊 Final Counter Timetable (Including SOS Officers)")
        fig_final = plotter.plot_officer_timetable_with_labels(
            st.session_state['final_counter_matrix']
        )
        st.plotly_chart(fig_final, use_container_width=True, key="fig_final_updated")
        
        # Show updated statistics
        if 'output_text' in st.session_state and len(st.session_state['output_text']) > 1:
            st.text_area(
                "Counter Manning Statistics (with SOS)",
                value=st.session_state['output_text'][1],
                height=400,
                key="stats_text_updated"
            )
    
    # Show Officer Schedule (Graph C)
    st.markdown("### 👮 Individual Officer Schedules")
    fig_edited = plotter.plot_officer_schedule_with_labels(st.session_state['edited_schedule'])
    st.plotly_chart(fig_edited, use_container_width=True, key="fig_edited_schedule")
    
    # Show edit summary
    if st.session_state['edit_history']:
        st.info(f"✏️ Total edits made: {len(st.session_state['edit_history'])}")
    
    # Officer schedule download button
    officer_schedule_text = "Officer Schedules (After Edits)\n" + "=" * 50 + "\n\n"
    for officer_key, schedule in st.session_state['edited_schedule'].items():
        officer_schedule_text += f"{officer_key}:\n"
        schedule_str = ", ".join(
            [str(c) if c != 0 else "-" for c in schedule]
        )
        officer_schedule_text += f"  {schedule_str}\n\n"

    st.download_button(
        label="📥 Download Updated Officer Schedules",
        data=officer_schedule_text,
        file_name="officer_schedules_edited.txt",
        mime="text/plain",
        key="download_officer_schedules_edited",
        use_container_width=False
    )
    
    
    # Database edit history viewer (keep at bottom)
with st.expander("🗄️ View Database Edit History"):
    recent_edits = get_roster_edits(limit=20)
    if recent_edits:
        edit_data = []
        for edit in recent_edits:
            edit_data.append({
                'Timestamp': edit.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Type': edit.edit_type,
                'Officer': edit.officer_id if edit.officer_id else '-',
                'Officer 2': edit.officer_id_2 if edit.officer_id_2 else '-',
                'Counter': str(edit.counter_no) if edit.counter_no is not None else '-',  # ← Convert to string
                'Time Range': f"{edit.time_start if edit.time_start else '-'}-{edit.time_end if edit.time_end else '-'}",
                'Notes': edit.notes if edit.notes else '-'
            })
        st.dataframe(edit_data, use_container_width=True)
    else:
        st.info("No edit history found in database")

# === Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
        <p>AC Roster Generator | Morning Shift Planning Tool</p>
        <p>Powered by ScheduleManager OOP Architecture v3.3 🚀</p>
        <p>Now with restructured UI flow: Add SOS via Roster Editor!</p>
        <p>For issues or suggestions, please contact your system administrator.</p>
    </div>
    """,
    unsafe_allow_html=True
)