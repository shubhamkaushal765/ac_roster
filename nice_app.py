"""
NiceGUI web application for AC Roster Generation (Morning Shift)
Refactored for maintainability and scalability
"""

import os
#os.environ['NICEGUI_DISABLE_CSP'] = '1'
import re
from nicegui import ui, app
import pandas as pd
import numpy as np
import traceback
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from acroster import Plotter
from acroster.config import NUM_SLOTS, START_HOUR, MODE_CONFIG, OperationMode
from acroster.orchestrator_pipe import RosterAlgorithmOrchestrator
from acroster.db_handlers import save_last_inputs, get_last_inputs
from acroster.time_utils import hhmm_to_slot, generate_time_slots, get_end_time_slots, extract_officer_timings
from acroster.schedule_utils import schedule_to_matrix, get_all_officer_ids
from acroster.statistics import StatisticsGenerator


from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class NiceGUICompatibleCSP(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Remove any restrictive CSP headers
        response.headers.pop("Content-Security-Policy", None)
        response.headers.pop("Content-Security-Policy-Report-Only", None)
        
        # Set CSP that works with NiceGUI's requirements
        # NiceGUI REQUIRES 'unsafe-inline' and 'unsafe-eval' due to Vue.js
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https: http: ws: wss: data: blob:; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https: http:; "
            "style-src 'self' 'unsafe-inline' https: http:; "
            "img-src 'self' data: blob: https: http:; "
            "font-src 'self' data: https: http:; "
            "connect-src 'self' https: http: ws: wss:; "
            "worker-src 'self' blob:;"
        )
        
        return response

# Add middleware before UI code
app.add_middleware(NiceGUICompatibleCSP)

@dataclass
class InputDefaults:
    """Default values for form inputs"""
    main_officers: str = '1-18'
    gl_counters: str = '4AC1, 8AC11, 12AC21, 16AC31'
    handwritten_counters: str = '3AC12,5AC13'
    ot_counters: str = '2,3,20'
    ro_ra_officers: str = '3RO2100, 11RO1700,15RO2130'
    beam_width: int = 20


class FormInputs:
    """Container for all form input components"""
    
    def __init__(self, saved_inputs: Dict[str, Any]):
        self.saved_inputs = saved_inputs
        self.operation_mode: Optional[ui.select] = None
        self.main_officers: Optional[ui.input] = None
        self.gl_counters: Optional[ui.input] = None
        self.handwritten_counters: Optional[ui.input] = None
        self.ot_counters: Optional[ui.input] = None
        self.ro_ra_officers: Optional[ui.input] = None
        self.beam_width: Optional[ui.slider] = None
        self.show_debug: Optional[ui.checkbox] = None
        self.form_container: Optional[ui.column] = None 
        self.toggle_form_btn: Optional[ui.button] = None 
        
    def get_values(self) -> Dict[str, Any]:
        """Extract all input values as a dictionary"""
        return {
            'operation_mode': self.operation_mode.value,
            'main_officers': self.main_officers.value.strip(),
            'gl_counters': self.gl_counters.value.strip(),
            'handwritten_counters': self.handwritten_counters.value.strip(),
            'ot_counters': self.ot_counters.value.strip(),
            'ro_ra_officers': self.ro_ra_officers.value.strip(),
            'beam_width': self.beam_width.value,
            'show_debug': self.show_debug.value,
        }
    
    def get_summary_html(self) -> str:
        """Generate HTML summary of current inputs"""
        # Return empty if inputs not yet initialized
        if not self.main_officers:
            return '<div><p><em>Fill in the form to see your inputs here</em></p></div>'
        
        return f'''
            <div>
                <p><strong>Main Officers:</strong> {self.main_officers.value}</p>
                <p><strong>GL Counters:</strong> {self.gl_counters.value}</p>
                <p><strong>Handwritten:</strong> {self.handwritten_counters.value}</p>
                <p><strong>OT Counters:</strong> {self.ot_counters.value}</p>
                <p><strong>RO/RA:</strong> {self.ro_ra_officers.value}</p>
            </div>
        '''


class RosterGenerationUI:
    """Main UI class for roster generation"""
    
    def __init__(self):
        self.saved_inputs = get_last_inputs() or {}
        self.defaults = InputDefaults()
        self.inputs = FormInputs(self.saved_inputs)
        self.result_container: Optional[ui.column] = None
        self.summary_html: Optional[ui.html] = None
        self.spinner: Optional[ui.spinner] = None
        self.sidebar_container: Optional[ui.column] = None
        self.edited_schedule: Optional[Dict] = None
        self.current_orchestrator = None
        self.current_values = None
        self.timetable_history = []  # List of (fig, stats, timestamp, description) tuples
        self.schedule_history = []   # List of (fig, timestamp, descriptio
        self.timetable_carousel: Optional[ui.carousel] = None
        self.schedule_carousel: Optional[ui.carousel] = None

        
    def render(self):
        """Render the complete UI"""
        self._render_header()

        with ui.row().style("width: 100%"):
            with ui.column().style("flex: 3"):
                self._render_main_form()
                self.result_container = ui.column()
            with ui.column().style("flex: 2"):
                self.sidebar_container = ui.column().style("width: 100%")
                self._render_sidebar()
    
    def _render_header(self):
        """Render page header"""
        ui.label("Generate AC/DC roster (Morning)").classes("text-2xl font-bold")
        ui.label("💡 For better display on mobile, please enable Desktop site in your browser settings.")\
            .style("font-size:14px; color:gray; margin-top:-10px;")
    
    def _render_sidebar(self):  
        """Render sidebar content with horizontal tabs and sub-tabs"""
        with self.sidebar_container:
            ui.label('🗂️ Roster Editor').classes('text-lg font-bold')

            if self.edited_schedule is None:
                ui.label(
                    "ℹ️ No schedule generated yet. Please generate a schedule first."
                ).style("color: gray; font-size: 14px;")
            else:
                
                def natural_sort_key(s):
                    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

                officer_ids = sorted(get_all_officer_ids(self.edited_schedule), key=natural_sort_key)
                time_slots = generate_time_slots(START_HOUR, NUM_SLOTS)

                # Horizontal tabs for Add / Swap / Delete
                with ui.tabs().props('dense align=justify').classes('w-full') as main_tabs:
                    add_tab = ui.tab('➕ Add')
                    swap_tab = ui.tab('🔄 Swap')
                    delete_tab = ui.tab('🗑️ Delete')

                with ui.tab_panels(main_tabs, value=add_tab, animated=False).classes('w-full'):
                    # ---------------- Add SOS Officers Panel ----------------
                    with ui.tab_panel(add_tab):
                        ui.label("Paste list of SOS officers given by Ops Rm here")
                        with ui.tabs().props('dense align=justify').classes('w-full') as add_subtabs:
                            raw_tab = ui.tab('Text Input (Recommended)')
                            manual_tab = ui.tab('Manual Input')
                        
                        with ui.tab_panels(add_subtabs, value=raw_tab, animated = False).classes('w-full'):
                            # Raw Input Panel
                            with ui.tab_panel(raw_tab):
                                raw_sos_input = ui.textarea(
                                    label='Paste SOS timings (raw format)',
                                    placeholder='ACAR SOS AM\n02 x GC\n...',
                                    value='',
                                ).classes('w-full').props('rows=8')
                                ui.button(
                                    '🔍 Extract SOS Officers',
                                    on_click=lambda: self._extract_sos_officers(raw_sos_input.value)
                                ).props('color=primary')
                            
                            # Manual Input Panel
                            with ui.tab_panel(manual_tab):
                                ui.markdown(
                                    """
                                **Example:** (AC22)1000-1300, 2000-2200, 1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200  
                                **Format:** (`<optional counter no. at 1000>)<sos_timing>`  
                                If an officer has multiple SOS timings, separate them with semicolons `;`.  
                                Optional pre-assigned counters must be enclosed in parentheses `()` before the time.
                                """
                                )
                                manual_sos_input = ui.textarea(
                                    label='Manual SOS Timings',
                                    placeholder='(AC22)1000-1300;1315-1430,...',
                                    value='',
                                ).classes('w-full').props('rows=4')
                                ui.button(
                                    '✅ Add Manual SOS Officers',
                                    on_click=lambda: self._add_manual_sos(manual_sos_input.value)
                                ).props('color=secondary')
                    
                    # ---------------- Swap Assignments Panel ----------------
                    with ui.tab_panel(swap_tab):
                        swap_officer1 = ui.select(
                            label='Officer 1',
                            options=officer_ids,
                            value=officer_ids[0] if officer_ids else None
                        ).classes('w-full')
                        swap_officer2 = ui.select(
                            label='Officer 2',
                            options=officer_ids,
                            value=officer_ids[1] if len(officer_ids) > 1 else officer_ids[0]
                        ).classes('w-full')
                        swap_start = ui.select(
                            label='From Time',
                            options=time_slots,
                            value=time_slots[0]
                        ).classes('w-full')
                        swap_end = ui.select(
                            label='To Time',
                            options=get_end_time_slots(time_slots),
                            value=get_end_time_slots(time_slots)[1]
                        ).classes('w-full')
                        
                        ui.button(
                            'Swap Assignments',
                            on_click=lambda: self._swap_assignments(
                                swap_officer1.value,
                                swap_officer2.value,
                                swap_start.value,
                                swap_end.value
                            )
                        ).props('color=primary')
                    
                    # ---------------- Delete Assignments Panel ----------------
                    with ui.tab_panel(delete_tab):
                        del_officer = ui.select(
                            label='Officer',
                            options=officer_ids,
                            value=officer_ids[0] if officer_ids else None
                        ).classes('w-full')
                        del_start = ui.select(
                            label='From Time',
                            options=time_slots,
                            value=time_slots[0]
                        ).classes('w-full')
                        del_end = ui.select(
                            label='To Time',
                            options=get_end_time_slots(time_slots),
                            value=get_end_time_slots(time_slots)[1]
                        ).classes('w-full')
                        
                        ui.button(
                            'Delete Assignment',
                            on_click=lambda: self._delete_assignment(
                                del_officer.value,
                                del_start.value,
                                del_end.value
                            )
                        ).props('color=negative')
    def _render_main_form(self):
        """Render the main form with stepper"""
        # Summary card with toggle button
        with ui.card().classes('w-full mb-4'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Your Inputs:').classes('text-bold')
                self.toggle_form_btn = ui.button(
                    icon='expand_less',
                    on_click=self._toggle_form_visibility
                ).props('flat dense').tooltip('Show/Hide Form')
            
            self.summary_html = ui.html('', sanitize=False)
            self._update_summary()
        
        # Stepper in a collapsible container
        self.form_container = ui.column().classes('w-full')
        with self.form_container:
            with ui.stepper().props('vertical').classes('w-full') as stepper:
                self._render_step_main_officers(stepper)
                self._render_step_gl_counters(stepper)
                self._render_step_handwritten(stepper)
                self._render_step_ot_counters(stepper)
                self._render_step_ro_ra(stepper)
                self._render_step_optional(stepper)
                self._render_step_generate(stepper)
    def _render_step_main_officers(self, stepper):
        """Step 1: Main Officers"""
        with ui.step("Main Officers"):
            step_label = ui.label(f"Key in range of S/N assigned to {OperationMode.ARRIVAL.value} car")
            
            self.inputs.operation_mode = ui.select(
                options=[OperationMode.ARRIVAL.value, OperationMode.DEPARTURE.value],
                value=OperationMode.ARRIVAL.value,
                label='Operation Mode',
                on_change=lambda e: step_label.set_text(
                    f"Key in range of S/N assigned to {e.value} car roster"
                )
            ).props('outlined').style("width: 100%")
            
            ui.label("E.g 1-18 or 1,3,5-10")
            self.inputs.main_officers = ui.input(
                '', 
                value=self.defaults.main_officers
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self._update_summary(), stepper.next()))
                ui.button(
                    '🚀 Quick Generate Schedule',
                    on_click=self._run_generation,
                    color='primary'
                ).classes('w-full mb-2')

    
    def _render_step_gl_counters(self, stepper):
        """Step 2: GL Counters"""
        with ui.step("Report to GL counters"):
            ui.label("Which counter did Chops RM assign S/N 4, 8, 12, 16... from 1000-1130? Key in as <S/N>AC<counter no.>")
            ui.label("E.g. 4AC1, 8AC11, 12AC21, 16AC31")
            self.inputs.gl_counters = ui.input(
                '', 
                value=self.saved_inputs.get('gl_counters', self.defaults.gl_counters)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self._update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_handwritten(self, stepper):
        """Step 3: Handwritten Counters"""
        with ui.step("Handwritten Counters (1000-1030 only)"):
            ui.label("Did Chop RM manually change some of the first counters? Key in as <S/N>AC<counter no.>")
            ui.label("E.g. 3AC12, 5AC13")
            self.inputs.handwritten_counters = ui.input(
                "", 
                value=self.saved_inputs.get('handwritten_counters', self.defaults.handwritten_counters)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self._update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_ot_counters(self, stepper):
        """Step 4: OT Counters"""
        with ui.step("OT counters"):
            ui.label("Which counters are manned by OT staff till 1030? Key in the list of counter no. separated by commas")
            ui.label("E.g. 2,3,20")
            self.inputs.ot_counters = ui.input(
                "", 
                value=self.saved_inputs.get('ot_counters', self.defaults.ot_counters)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self._update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_ro_ra(self, stepper):
        """Step 5: RO/RA Officers"""
        with ui.step("RO/RA officers"):
            ui.label("Which S/N is reporting late (RA) or leaving early (RO)? Key in as <S/N><RO or RA><counter no.>")
            ui.label("E.g. 3RO2100,11RO1700,15RO2130")
            self.inputs.ro_ra_officers = ui.input(
                "", 
                value=self.saved_inputs.get('ro_ra_officers', self.defaults.ro_ra_officers)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self._update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_optional(self, stepper):
        """Step 6: Optional Settings"""
        with ui.step("Optional"):
            with ui.expansion('⚙️ Advanced Options'):
                ui.label('Beam Search Width')
                self.inputs.beam_width = ui.slider(
                    min=10, max=100, 
                    value=self.saved_inputs.get('beam_width', self.defaults.beam_width)
                )
                self.inputs.show_debug = ui.checkbox('Show Debug Information', value=True)
            
            with ui.stepper_navigation():
                ui.button('Done', on_click=lambda: (self._update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_generate(self, stepper):
        """Step 7: Generate Schedule"""
        with ui.step("Generate Schedule"):
            ui.label('Click the button below to generate the schedule')
            
            with ui.row().classes('w-full justify-center'):
                ui.button(
                    '🚀 Generate Schedule',
                    on_click=self._run_generation,
                    color='primary'
                ).classes('w-1/2')
            
            self.spinner = ui.spinner(size='lg').props('color=primary')
            self.spinner.visible = False
            
            with ui.stepper_navigation():
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _update_summary(self):
        """Update the summary card with current input values"""
        if self.summary_html:
            self.summary_html.set_content(self.inputs.get_summary_html())
    
    def _run_generation(self):
        """Execute the roster generation algorithm"""
        self.spinner.visible = True
        self.result_container.clear()
        
        try:
            values = self.inputs.get_values()
            
            # Validate required fields
            if not values['main_officers']:
                ui.notify("⚠️ 'Main Officers Reported' is required", type='warning')
                self.spinner.visible = False
                return
            
            # Create orchestrator
            orchestrator = RosterAlgorithmOrchestrator(
                mode=OperationMode(values['operation_mode'])
            )
            
            # Run algorithm
            counter_matrix, final_counter_matrix, officer_schedule, output_text = orchestrator.run(
                main_officers_reported=values['main_officers'],
                report_gl_counters=values['gl_counters'],
                sos_timings="",
                ro_ra_officers=values['ro_ra_officers'],
                handwritten_counters=values['handwritten_counters'],
                ot_counters=values['ot_counters'],
            )
            
            self.spinner.visible = False
            ui.notify('✅ Schedule generated successfully!', type='positive')
            self._hide_form_after_generation()
            # Store the schedule and orchestrator for editing
            self.edited_schedule = officer_schedule.copy()
            self.current_orchestrator = orchestrator
            self.current_values = values
            
            # Save inputs
            save_last_inputs({
                'main_officers': values['main_officers'],
                'gl_counters': values['gl_counters'],
                'handwritten_counters': values['handwritten_counters'],
                'ot_counters': values['ot_counters'],
                'ro_ra_officers': values['ro_ra_officers'],
                'beam_width': values['beam_width'],
            })
            
            # Render results
            with self.result_container:
                self._render_results(
                    orchestrator, 
                    counter_matrix, 
                    final_counter_matrix, 
                    officer_schedule, 
                    output_text,
                    values
                )
            
            # Update sidebar to show roster editor
            self.sidebar_container.clear()
            with self.sidebar_container:
                self._render_sidebar()
        
        except Exception as e:
            self.spinner.visible = False
            ui.notify(f'❌ {str(e)}', type='negative')
            
            if self.inputs.show_debug.value:
                with self.result_container:
                    ui.code(traceback.format_exc())
    
    def _render_results(self, orchestrator, counter_matrix, final_counter_matrix, 
                    officer_schedule, output_text, values):
        """Render all result visualizations and data"""
        config = MODE_CONFIG[OperationMode(values['operation_mode'])]
        
        # Officer counts
        self._render_officer_metrics(orchestrator)
        
        # Optimization penalty
        if orchestrator.penalty is not None:
            ui.notify(f"🎯 Optimization Penalty: {orchestrator.penalty:.2f}", type='positive')
        
        # Initialize plotter
        plotter = Plotter(
            num_slots=NUM_SLOTS,
            num_counters=config['num_counters'],
            start_hour=START_HOUR,
        )

        timestamp = datetime.now().strftime("%H%M")
        
        # Graph 1: Counter Timetable (starts with counter_matrix)
        fig1 = plotter.plot_officer_timetable_with_labels(counter_matrix)
        description1 = "Initial schedule generated"
        self.timetable_history = [(fig1, output_text[0], timestamp, description1)]
        self._render_timetable_gallery()
        
        # Graph 2: Officer Schedules
        fig2 = plotter.plot_officer_schedule_with_labels(officer_schedule)
        description2 = "Initial schedule generated"
        self.schedule_history = [(fig2, None, timestamp, description2)]
        self._render_schedule_gallery()
        
        # Debug info
        if values['show_debug']:
            self._render_debug_info(counter_matrix, final_counter_matrix, 
                                officer_schedule, orchestrator)
                                
    def _render_officer_metrics(self, orchestrator):
        """Render officer count metrics"""
        counts = orchestrator.get_officer_counts()
        
        with ui.row().classes('w-full gap-4'):
            for title, value in [
                ('👮 Main Officers', counts['main']),
                ('🆘 SOS Officers', counts['sos']),
                ('⏰ OT Officers', counts['ot']),
                ('📊 Total Officers', counts['total'])
            ]:
                with ui.column().style("flex: 1"):
                    with ui.card().classes('w-full text-center p-4'):
                        ui.label(title).classes('text-sm text-gray-500')
                        ui.label(str(value)).classes('text-3xl font-bold')
    def _render_timetable_gallery(self):

        """Render carousel gallery for counter timetable"""
        ui.separator()
        ui.label('📊 Counter Timetable History').classes('text-lg font-bold')
        ui.label(f'Showing {len(self.timetable_history)} version(s) - Swipe to see history').classes('text-sm text-gray-500')
        
        with ui.carousel(animated=True, arrows=True, navigation=True, value=f'timetable_{len(self.timetable_history) - 1}').props('height=1000px').classes('w-full'):
            for idx, (fig, stats, timestamp, description) in enumerate(self.timetable_history):
                with ui.carousel_slide(name=f'timetable_{len(self.timetable_history) - idx - 1}'):
                    with ui.column().classes('w-full'):
                        # Version header
                        if idx == 0:
                            ui.label(f'📌 Latest - {timestamp}').classes('text-lg font-bold mb-1 text-primary')
                        else:
                            ui.label(f'{timestamp}').classes('text-lg font-bold mb-1 text-gray-600')
                        
                        # Description
                        ui.markdown(description).classes('text-sm text-gray-700 mb-3 whitespace-pre-line')
                        
                        # Graph (fixed height, no scroll needed)
                        ui.plotly(fig).classes('w-full')
                        
                        # Stats (scrollable if needed)
                        ui.textarea(
                            label='Counter Manning Statistics',
                            value=stats,
                        ).classes('w-full').props('rows=10')

    def _render_schedule_gallery(self):
        """Render carousel gallery for officer schedules"""
        ui.separator()
        ui.label('👮 Officer Schedule History').classes('text-lg font-bold')
        ui.label(f'Showing {len(self.schedule_history)} version(s) - Swipe to see history').classes('text-sm text-gray-500')
        
        with ui.carousel(animated=True, arrows=True, navigation=True, value=f'schedule_{len(self.schedule_history) - 1}').props('height=700px').classes('w-full'):
            for idx, (fig, stats_nil, timestamp, description) in enumerate(self.schedule_history):
                with ui.carousel_slide(name=f'schedule_{idx}'):
                    with ui.column().classes('w-full'):
                        # Version header
                        if idx == len(self.schedule_history)-1:
                            ui.label(f'📌 Latest - {timestamp}').classes('text-lg font-bold mb-1 text-primary')
                        else:
                            ui.label(f'{timestamp}').classes('text-lg font-bold mb-1 text-gray-600')
                        
                        # Description
                        ui.markdown(description).classes('text-sm text-gray-700 mb-3 whitespace-pre-line')
                        
                        # Graph 2 height, no scroll needed since no stats)
                        ui.plotly(fig).classes('w-full').style('height: 600px;')

    def _render_merged_gallery(self):
        """Render unified carousel for timetable + schedule"""
        ui.separator()
        ui.label('📊 Counter Timetable + Officer Schedule History').classes('text-lg font-bold')
        ui.label(f'Showing {len(self.timetable_history)} version(s)').classes('text-sm text-gray-500')

        with ui.carousel(
            animated=True,
            arrows=True,
            navigation=True,
            value=f'slide_{len(self.timetable_history) - 1}',
        ).props('height=1000px').classes('w-full'):

            # zip the two histories together
            for idx, (tt_entry, sch_entry) in enumerate(zip(self.timetable_history, self.schedule_history)):
                fig1, stats1, timestamp1, description1 = tt_entry
                fig2, stats2, timestamp2, description2 = sch_entry

                with ui.carousel_slide(name=f'slide_{idx}'):
                    with ui.column().classes('w-full'):

                        # --- Timetable graph + stats ---
                        ui.plotly(fig1).classes('w-full')
                        ui.textarea(
                            label='Counter Manning Statistics',
                            value=stats1
                        ).classes('w-full').props('rows=10')

                        # --- Schedule graph ---
                        ui.plotly(fig2).classes('w-full').style('height: 600px;')

    
    def _render_debug_info(self, counter_matrix, final_counter_matrix, 
                        officer_schedule, orchestrator):
        """Render debug information"""
        ui.separator()
        ui.label('🔍 Debug Information').classes('text-lg font-bold')
        
        with ui.expansion('View Raw Data'):
            ui.code(f'Counter Matrix Shape: {counter_matrix.shape}')
            ui.code(f'Final Counter Matrix Shape: {final_counter_matrix.shape}')
            ui.code(f'Number of Officers: {len(officer_schedule)}')
            
            ui.label('Counter Matrix (first 5 rows)')
            ui.table(rows=pd.DataFrame(counter_matrix[:5, :10]).to_dict('records'))
            
            ui.label('Final Counter Matrix (first 5 rows)')
            ui.table(rows=pd.DataFrame(final_counter_matrix[:5, :10]).to_dict('records'))
        
        with ui.expansion('📊 Orchestrator State & Officer Details'):
            ui.code(str(orchestrator))
            
            ui.label('Main Officers')
            for k, o in list(orchestrator.get_main_officers().items())[:5]:
                ui.label(f'{k}: non-zero slots={(o.schedule != 0).sum()}')
            
            ui.label('SOS Officers')
            for o in orchestrator.get_sos_officers():
                ui.label(f'{o.officer_key} → {o.pre_assigned_counter}')
            
            ui.label('OT Officers')
            for o in orchestrator.get_ot_officers():
                ui.label(f'{o.officer_key} → counter {o.counter_no}')
    
    # === Roster Editor Methods ===
    
    def _extract_sos_officers(self, raw_text: str):
        """Extract SOS officers from raw text format"""
        if not raw_text.strip():
            ui.notify("⚠️ Please paste SOS timings", type='warning')
            return
        
        try:
            
            
            # Extract timings from raw text
            sos_timings_str = extract_officer_timings(raw_text)
            
            if not sos_timings_str:
                ui.notify("⚠️ No valid SOS timings found", type='warning')
                return
            
            # Re-run orchestrator with SOS timings
            self._rerun_with_sos(sos_timings_str)
            
        except Exception as e:
            ui.notify(f"❌ Error extracting SOS: {str(e)}", type='negative')
            if self.inputs.show_debug.value:
                ui.code(traceback.format_exc())
    
    def _add_manual_sos(self, manual_text: str):
        """Add manually entered SOS officers"""
        if not manual_text.strip():
            ui.notify("⚠️ Please enter SOS timings", type='warning')
            return
        
        try:
            self._rerun_with_sos(manual_text.strip())
        except Exception as e:
            ui.notify(f"❌ Error adding SOS: {str(e)}", type='negative')
            if self.inputs.show_debug.value:
                ui.code(traceback.format_exc())
    
    def _rerun_with_sos(self, sos_timings: str):
        """Re-run the orchestrator with SOS timings"""
        if not self.current_orchestrator or not self.current_values:
            ui.notify("⚠️ No schedule to update", type='warning')
            return
        
        self.spinner.visible = True
        
        try:
            
            
            # Create new orchestrator
            orchestrator = RosterAlgorithmOrchestrator(
                mode=OperationMode(self.current_values['operation_mode'])
            )
            
            # Run with SOS timings
            counter_matrix, final_counter_matrix, officer_schedule, output_text = orchestrator.run(
                main_officers_reported=self.current_values['main_officers'],
                report_gl_counters=self.current_values['gl_counters'],
                sos_timings=sos_timings,
                ro_ra_officers=self.current_values['ro_ra_officers'],
                handwritten_counters=self.current_values['handwritten_counters'],
                ot_counters=self.current_values['ot_counters'],
            )
            
            self.spinner.visible = False
            ui.notify('✅ Schedule updated with SOS officers!', type='positive')
            
            # Update stored schedule
            self.edited_schedule = officer_schedule.copy()
            self.current_orchestrator = orchestrator
            
            # Create plotter
            config = MODE_CONFIG[OperationMode(self.current_values['operation_mode'])]
            plotter = Plotter(
                num_slots=NUM_SLOTS,
                num_counters=config['num_counters'],
                start_hour=START_HOUR,
            )
            
            # Generate new figures
            fig1 = plotter.plot_officer_timetable_with_labels(final_counter_matrix)
            fig2 = plotter.plot_officer_schedule_with_labels(officer_schedule)
            
            # Calculate stats
            stats_generator = StatisticsGenerator(OperationMode(self.current_values['operation_mode']))
            stats = stats_generator.generate_statistics(final_counter_matrix)
            
            # Get timestamp
            timestamp = datetime.now().strftime("%H%M")
            
            # Build description with SOS officer details
            sos_officers = orchestrator.get_sos_officers()
            if sos_officers:
                description = f"Added {len(sos_officers)} SOS officer(s):\n"
                for sos in sos_officers:
                    description += f"• {sos.officer_key} at counter {sos.pre_assigned_counter}\n"
            else:
                description = "Added SOS officers"
            
            # Add to history
            self.timetable_history.append((fig1, stats, timestamp, description))
            self.schedule_history.append((fig2, None, timestamp, description))
            
            # Re-render galleries
            self.result_container.clear()
            with self.result_container:
                self._render_officer_metrics(orchestrator)
                self._render_timetable_gallery()
                self._render_schedule_gallery()
            
        except Exception as e:
            self.spinner.visible = False
            ui.notify(f"❌ Error updating schedule: {str(e)}", type='negative')
            if self.inputs.show_debug.value:
                ui.code(traceback.format_exc())

    def _swap_assignments(self, officer1: str, officer2: str, start_time: str, end_time: str):
        """Swap counter assignments between two officers"""
        if not all([officer1, officer2, start_time, end_time]):
            ui.notify("⚠️ Please select all fields", type='warning')
            return
        
        if officer1 == officer2:
            ui.notify("⚠️ Please select different officers", type='warning')
            return
        
        try:
            # ADD THIS DEBUG CODE HERE:
            print("=" * 80)
            print("DEBUG INFO BEFORE SWAP:")
            print(f"Number of officers in edited_schedule: {len(self.edited_schedule)}")
            print(f"Officer IDs: {list(self.edited_schedule.keys())}")
            print(f"NUM_SLOTS: {NUM_SLOTS}")
            config = MODE_CONFIG[OperationMode(self.current_values['operation_mode'])]
            print(f"config['num_counters']: {config['num_counters']}")
            print("=" * 80)
            
            start_slot = hhmm_to_slot(start_time)
            end_slot = hhmm_to_slot(end_time)
            
            if start_slot >= end_slot:
                ui.notify("⚠️ Start time must be before end time", type='negative')
                return
            
            # Swap assignments in edited_schedule
            if officer1 in self.edited_schedule and officer2 in self.edited_schedule:
                temp = self.edited_schedule[officer1][start_slot:end_slot].copy()
                self.edited_schedule[officer1][start_slot:end_slot] = self.edited_schedule[officer2][start_slot:end_slot]
                self.edited_schedule[officer2][start_slot:end_slot] = temp
                
                ui.notify(f"✅ Swapped {officer1} ↔ {officer2} from {start_time} to {end_time}", type='positive')
                
                # Create description
                description = f"Swapped officers {officer1} and {officer2} from {start_time} to {end_time}"

                # Re-render visualizations
                self._update_visualizations()
            else:
                ui.notify("⚠️ Officer not found in schedule", type='warning')
                
        except Exception as e:
            ui.notify(f"❌ Error swapping: {str(e)}", type='negative')
    
    def _delete_assignment(self, officer: str, start_time: str, end_time: str):
        """Delete counter assignments for an officer"""
        if not all([officer, start_time, end_time]):
            ui.notify("⚠️ Please select all fields", type='warning')
            return
        
        try:
            
            start_slot = hhmm_to_slot(start_time)
            end_slot = hhmm_to_slot(end_time)
            
            if start_slot >= end_slot:
                ui.notify("⚠️ Start time must be before end time", type='warning')
                return
            
            # Delete assignments
            if officer in self.edited_schedule:
                self.edited_schedule[officer][start_slot:end_slot-1] = 0
                
                ui.notify(f"✅ Deleted assignments for {officer} from {start_time} to {end_time}", type='positive')
                
                # Create description
                description = f"Deleted officer {officer} from {start_time} to {end_time}"
            
                # Re-render visualizations
                self._update_visualizations(description)
            else:
                ui.notify("⚠️ Officer not found in schedule", type='warning')
                
        except Exception as e:
            ui.notify(f"❌ Error deleting: {str(e)}", type='negative')
    
    def _update_visualizations(self, description: str = "Manual edit"):
        """Update the schedule visualizations after editing"""
        try:
            
            
            config = MODE_CONFIG[OperationMode(self.current_values['operation_mode'])]
            
            # Convert edited schedule back to matrix
            edited_counter_matrix = np.zeros((config['num_counters'], NUM_SLOTS), dtype=object)

            # Fill the counter matrix from officer schedules
            for officer_id, schedule in self.edited_schedule.items():
                for slot_idx, counter_no in enumerate(schedule):
                    if counter_no != 0:
                        counter_idx = counter_no - 1  # Counter numbers are 1-indexed
                        if 0 <= counter_idx < config['num_counters']:
                            edited_counter_matrix[counter_idx, slot_idx] = officer_id


            # ADD THIS DEBUG CODE HERE:
            print("=" * 80)
            print("DEBUG INFO IN UPDATE_VISUALIZATIONS:")
            print(f"edited_counter_matrix shape: {edited_counter_matrix.shape}")
            print(f"Number of officers: {len(self.edited_schedule)}")
            print(f"NUM_SLOTS: {NUM_SLOTS}")
            print(f"config['num_counters']: {config['num_counters']}")
            print(f"Expected matrix shape should be: (num_officers={len(self.edited_schedule)}, num_slots={NUM_SLOTS})")
            print("=" * 80)
            # Create plotter
            plotter = Plotter(
                num_slots=NUM_SLOTS,
                num_counters=config['num_counters'],
                start_hour=START_HOUR,
            )
            
            # Generate BOTH figures
            print("DEBUG: Step 4 - Calling plot_officer_timetable_with_labels")
            fig1 = plotter.plot_officer_timetable_with_labels(edited_counter_matrix)
            print("DEBUG: Step 4 - COMPLETED")
            
            print("DEBUG: Step 5 - Calling plot_officer_schedule_with_labels")
            fig2 = plotter.plot_officer_schedule_with_labels(self.edited_schedule)
            print("DEBUG: Step 5 - COMPLETED")

            # Calculate stats for Graph 1
            print("DEBUG: Step 6 - Generating statistics")
            stats_generator = StatisticsGenerator(OperationMode(self.current_values['operation_mode']))
            stats = stats_generator.generate_statistics(edited_counter_matrix)
            print("DEBUG: Step 6 - COMPLETED")

            # Get timestamp
            print("DEBUG: Step 7 - Getting timestamp")
            timestamp = datetime.now().strftime("%H%M")

            # Add BOTH to history
            print("DEBUG: Step 8 - Adding to history")
            self.timetable_history.append((fig1, stats, timestamp, description))
            self.schedule_history.append((fig2, None, timestamp, description))

            # Re-render BOTH galleries
            print("DEBUG: Step 9 - Clearing result container")
            self.result_container.clear()
            
            print("DEBUG: Step 10 - Rendering galleries")
            with self.result_container:
                print("DEBUG: Step 10a - Rendering officer metrics")
                self._render_officer_metrics(self.current_orchestrator)
                
                print("DEBUG: Step 10b - Rendering timetable gallery")
                self._render_timetable_gallery()
                
                print("DEBUG: Step 10c - Rendering schedule gallery")
                self._render_schedule_gallery()
            
            print("DEBUG: ALL STEPS COMPLETED!")
            ui.notify('✅ Visualizations updated!', type='positive')
            
        except Exception as e:
            error_details = traceback.format_exc()
            print("=" * 80)
            print("ERROR IN _UPDATE_VISUALIZATIONS:")
            print(error_details)
            print("=" * 80)
            ui.notify(f"❌ Error updating visualizations: {str(e)}", type='negative')
            
    def _toggle_form_visibility(self):
        """Toggle form visibility"""
        if self.form_container.visible:
            self.form_container.visible = False
            self.toggle_form_btn.props('icon=expand_more')
        else:
            self.form_container.visible = True
            self.toggle_form_btn.props('icon=expand_less')

    def _hide_form_after_generation(self):
        """Hide form after successful generation"""
        if hasattr(self, 'form_container'):
            self.form_container.visible = False
            self.toggle_form_btn.props('icon=expand_more')



def main():
    """Main entry point"""
    ui.window_title = "Generate ACar/DCar Roster Morning"
    myapp = RosterGenerationUI()
    myapp.render()
    # app.add_static_files('/_nicegui', 'nicegui')
    # app.on_startup(lambda: None)
    ui.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 3000)),
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()