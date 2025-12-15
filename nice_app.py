"""
NiceGUI web application for AC Roster Generation (Morning Shift)
Refactored for maintainability and scalability
"""

from nicegui import ui, app
import pandas as pd
import traceback
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from acroster import Plotter
from acroster.config import NUM_SLOTS, START_HOUR, MODE_CONFIG, OperationMode
from acroster.orchestrator_pipe import RosterAlgorithmOrchestrator
from acroster.db_handlers import save_last_inputs, get_last_inputs


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
        
    def render(self):
        """Render the complete UI"""
        self._render_header()
        
        with ui.row().style("width: 100%"):
            with ui.column().style("flex: 4"):
                self._render_main_form()
                self.result_container = ui.column()
            with ui.column().style("flex: 1"):
                self._render_sidebar()
    
    def _render_header(self):
        """Render page header"""
        ui.label("Generate AC/DC roster (Morning)").classes("text-2xl font-bold")
        ui.label("💡 For better display on mobile, please enable Desktop site in your browser settings.")\
            .style("font-size:14px; color:gray; margin-top:-10px;")
    
    def _render_sidebar(self):
        """Render sidebar content"""
        ui.label('Side bar')
    
    def _render_main_form(self):
        """Render the main form with stepper"""
        # Summary card
        with ui.card().classes('w-full mb-4'):
            ui.label('Your Inputs:').classes('text-bold')
            self.summary_html = ui.html('', sanitize=False)
            self._update_summary()
        
        # Stepper
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
        
        # Main timetable
        self._render_main_timetable(plotter, counter_matrix, output_text[0])
        
        # Final timetable
        self._render_final_timetable(plotter, final_counter_matrix, output_text[1])
        
        # Officer schedules
        self._render_officer_schedules(plotter, officer_schedule)
        
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
    
    def _render_main_timetable(self, plotter, counter_matrix, stats_text):
        """Render main officer timetable"""
        ui.separator()
        ui.label('📊 Timetable (Main Officers Only)').classes('text-lg font-bold')
        
        fig = plotter.plot_officer_timetable_with_labels(counter_matrix)
        ui.plotly(fig).classes('w-full')
        
        ui.textarea(
            label='Counter Manning Statistics',
            value=stats_text,
        ).classes('w-full h-64')
    
    def _render_final_timetable(self, plotter, final_counter_matrix, stats_text):
        """Render final timetable with SOS"""
        ui.separator()
        ui.label('📊 Timetable (Including SOS Officers)').classes('text-lg font-bold')
        ui.label('ℹ️ No SOS officers added yet. Use the Roster Editor below to add SOS officers.')
        
        fig = plotter.plot_officer_timetable_with_labels(final_counter_matrix)
        ui.plotly(fig).classes('w-full')
        
        ui.textarea(
            label='Counter Manning Statistics (with SOS)',
            value=stats_text,
        ).classes('w-full h-64')
    
    def _render_officer_schedules(self, plotter, officer_schedule):
        """Render individual officer schedules"""
        ui.separator()
        ui.label('👮 Individual Officer Schedules').classes('text-lg font-bold')
        
        fig = plotter.plot_officer_schedule_with_labels(officer_schedule)
        ui.plotly(fig).classes('w-full')
    
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


def main():
    """Main entry point"""
    ui.window_title = "Generate ACar/DCar Roster Morning"
    my_app = RosterGenerationUI()
    my_app.render()
    ui.run()


if __name__ in {"__main__", "__mp_main__"}:
    main()