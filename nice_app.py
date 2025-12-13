"""
niceGUI web application for AC Roster Generation (Morning Shift)
"""
#nice_app.py

from nicegui import ui
import streamlit as st
import pandas as pd
import numpy as np
import re

from acroster import Plotter
from acroster.config import NUM_SLOTS, START_HOUR, MODE_CONFIG, OperationMode
from acroster.orchestrator_pipe import RosterAlgorithmOrchestrator
from acroster.schedule_utils import get_all_officer_ids, schedule_to_matrix, matrix_to_schedule
from acroster.time_utils import hhmm_to_slot, slot_to_hhmm, generate_time_slots, get_slot_end_time, clean_time, extract_officer_timings
from acroster.database import get_db_instance
from acroster.db_handlers import (
    save_last_inputs, get_last_inputs, save_roster_history,
    save_roster_edit, get_roster_edits
)


# Set page title
ui.window_title = "Generate ACar/DCar Roster Morning"

# Page title
ui.label("Generate AC/DC roster (Morning)").classes("text-2xl font-bold")

ui.label("💡 For better display on mobile, please enable Desktop site in your browser settings.")\
    .style("font-size:14px; color:gray; margin-top:-10px;")

# # Instruction box
# with ui.card().style("border: 1px solid white; border-radius: 8px; padding: 10px; margin-top: 10px;"):
#     ui.label("How to use").classes("text-lg font-semibold")
#     ui.markdown(
#         """
# - `GL counters` is only applicable for officers in main roster with S/N `4,8,12 ...` and are not assigned a counter from `1000-1115`.
# - `Handwritten counters` are counters assigned by chops room at the start of the shift as their original counter on main roster is already occupied (e.g. has OT officer). If `S/N 3` in main roster should report to `AC12` at the start of shift,  key in as `3AC12`
# - `RO/RA Officers` for `S/N 11` to `RO` at `1700` is written as `11RO1700`
# - **SOS officers can be added after generating the initial schedule using the Roster Editor below**
#         """
#     )

saved_inputs = get_last_inputs() or {}

# Helper for validation
def validate_input(value, field_name, required=False):
    """Validate input and return cleaned value"""
    if required and not value.strip():
        return None, f"⚠️ '{field_name}' is a required field."
    return value.strip(), None

def display_main_content():
    def update_step_label(event):
        step_label.set_text(f"Key in range of S/N assigned to {event.value} car roster")

    # Operation Mode select with on_change callback
    operation_mode = ui.select(
        options=[OperationMode.ARRIVAL.value, OperationMode.DEPARTURE.value],
        value=OperationMode.ARRIVAL.value,
        label='Operation Mode', 
        on_change=update_step_label
    ).props('outlined').style("width: 100%")
    
    main_officers_reported = None
    report_gl_counters = None
    handwritten_counters = None
    ot_counters = None
    ro_ra_officers = None

    
    def update_summary():
        summary_html.set_content(f'''
            <div>
                <p><strong>Main Officers:</strong> {main_officers_reported.value if main_officers_reported else ''}</p>
                <p><strong>GL Counters:</strong> {report_gl_counters.value if report_gl_counters else ''}</p>
                <p><strong>Handwritten:</strong> {handwritten_counters.value if handwritten_counters else ''}</p>
                <p><strong>OT Counters:</strong> {ot_counters.value if ot_counters else ''}</p>
                <p><strong>RO/RA:</strong> {ro_ra_officers.value if ro_ra_officers else ''}</p>
            </div>
        ''')
    # Summary card at the top
    with ui.card().classes('w-full mb-4') as summary_card:
        ui.label('Your Inputs:').classes('text-bold')
        summary_html = ui.html('', sanitize=False)
        update_summary()
    
    
    # Stepper
    with ui.stepper().props('vertical').classes('w-full') as stepper:
        with ui.step("Main Officers"):
            step_label = ui.label(f"Key in range of S/N assigned to {operation_mode.value} car")
            ui.label("E.g 1-18 or 1,3,5-10")
            main_officers_reported = ui.input('', value='1-18').style("width: 100%")
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (update_summary(), stepper.next()))
                
        with ui.step("Report to GL counters"):
            ui.label("Which counter did Chops RM assign S/N 4, 8, 12, 16... from 1000-1130? Key in as <S/N>AC<counter no.>")
            ui.label("E.g. 4AC1, 8AC11, 12AC21, 16AC31")
            report_gl_counters = ui.input('', value=saved_inputs.get('gl_counters', '4AC1, 8AC11, 12AC21, 16AC31')).style("width: 100%")
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
                
        with ui.step("Handwritten Counters (1000-1030 only)"):
            ui.label("Did Chop RM manually change some of the first counters? I.e. Is there any handwritten counter on the roster. If yes, key in as <S/N>AC<counter no.>")
            ui.label("E.g. 3AC12, 5AC13")
            handwritten_counters = ui.input("", value=saved_inputs.get('handwritten_counters', '3AC12,5AC13')).style("width: 100%")
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
                
        with ui.step("OT counters"):
            ui.label("Which counters are manned by OT staff till 1030? Key in the list of counter no. separated by commas")
            ui.label("E.g. 2,3,20")
            ot_counters = ui.input("", value=saved_inputs.get('ot_counters','2,3,20')).style("width: 100%")
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
                
        with ui.step("RO/RA officers"):
            ui.label("Which S/N is reporting late (RA) or leaving early (RO)? Key in as <S/N><RO or RA><counter no.>")
            ui.label("E.g. 3RO2100,11RO1700,15RO2130")
            ro_ra_officers = ui.input("", value=saved_inputs.get('ro_ra_officers', '3RO2100, 11RO1700,15RO2130')).style("width: 100%")
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
                
        with ui.step("Optional"):
            with ui.expansion('⚙️ Advanced Options'):
                ui.label('Beam Search Width')
                beam_width = ui.slider(
                    min=10, max=100, value=saved_inputs.get('beam_width', 20)
                )
                #ui.tooltip(beam_width, 'Higher values may produce better schedules but take longer')
                show_debug = ui.checkbox(
                    'Show Debug Information',
                    value=True
                )
            with ui.stepper_navigation():
                ui.button('Done', on_click=lambda: (update_summary(), stepper.next(),on_finish()))
                ui.button('Back', on_click=stepper.previous).props('flat')
        with ui.step("Generating schedule..."):
            ui.label('Success! Please scroll down')
            ui.button('Back', on_click=stepper.previous).props('flat')
    def on_finish():
        with result_container:
            ui.label('Plotly graph coming soon...')

    return

with ui.row().style("width: 100%"):  # full-width row
    with ui.column().style("flex: 4"):
        display_main_content()
        result_container = ui.column()
    with ui.column().style("flex: 1"):
        ui.label('side bar')
                    

ui.run(native=True, reload=False, dark = None)