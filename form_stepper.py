"""
Form stepper component for roster input
"""

from nicegui import ui
from typing import Callable
from acroster.config import OperationMode
from form_models import FormInputs, InputDefaults


class FormStepperUI:
    """Multi-step form for roster generation inputs"""
    
    def __init__(
        self, 
        inputs: FormInputs, 
        defaults: InputDefaults,
        on_generate: Callable,
        on_update_summary: Callable
    ):
        self.inputs = inputs
        self.defaults = defaults
        self.on_generate = on_generate
        self.on_update_summary = on_update_summary
    
    def render(self):
        """Render the complete stepper form"""
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
                ui.button('Next', on_click=lambda: (self.on_update_summary(), stepper.next()))
                ui.button(
                    '🚀 Quick Generate Schedule',
                    on_click=self.on_generate,
                    color='primary'
                ).classes('w-full mb-2')
    
    def _render_step_gl_counters(self, stepper):
        """Step 2: GL Counters"""
        with ui.step("Report to GL counters"):
            ui.label("Which counter did Chops RM assign S/N 4, 8, 12, 16... from 1000-1130? Key in as <S/N>AC<counter no.>")
            ui.label("E.g. 4AC1, 8AC11, 12AC21, 16AC31")
            self.inputs.gl_counters = ui.input(
                '', 
                value=self.inputs.saved_inputs.get('gl_counters', self.defaults.gl_counters)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self.on_update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_handwritten(self, stepper):
        """Step 3: Handwritten Counters"""
        with ui.step("Handwritten Counters (1000-1030 only)"):
            ui.label("Did Chop RM manually change some of the first counters? Key in as <S/N>AC<counter no.>")
            ui.label("E.g. 3AC12, 5AC13")
            self.inputs.handwritten_counters = ui.input(
                "", 
                value=self.inputs.saved_inputs.get('handwritten_counters', self.defaults.handwritten_counters)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self.on_update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_ot_counters(self, stepper):
        """Step 4: OT Counters"""
        with ui.step("OT counters"):
            ui.label("Which counters are manned by OT staff till 1030? Key in the list of counter no. separated by commas")
            ui.label("E.g. 2,3,20")
            self.inputs.ot_counters = ui.input(
                "", 
                value=self.inputs.saved_inputs.get('ot_counters', self.defaults.ot_counters)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self.on_update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_ro_ra(self, stepper):
        """Step 5: RO/RA Officers"""
        with ui.step("RO/RA officers"):
            ui.label("Which S/N is reporting late (RA) or leaving early (RO)? Key in as <S/N><RO or RA><counter no.>")
            ui.label("E.g. 3RO2100,11RO1700,15RO2130")
            self.inputs.ro_ra_officers = ui.input(
                "", 
                value=self.inputs.saved_inputs.get('ro_ra_officers', self.defaults.ro_ra_officers)
            ).style("width: 100%")
            
            with ui.stepper_navigation():
                ui.button('Next', on_click=lambda: (self.on_update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_optional(self, stepper):
        """Step 6: Optional Settings"""
        with ui.step("Optional"):
            with ui.expansion('⚙️ Advanced Options'):
                ui.label('Beam Search Width')
                self.inputs.beam_width = ui.slider(
                    min=10, max=100, 
                    value=self.inputs.saved_inputs.get('beam_width', self.defaults.beam_width)
                )
                self.inputs.show_debug = ui.checkbox('Show Debug Information', value=True)
            
            with ui.stepper_navigation():
                ui.button('Done', on_click=lambda: (self.on_update_summary(), stepper.next()))
                ui.button('Back', on_click=stepper.previous).props('flat')
    
    def _render_step_generate(self, stepper):
        """Step 7: Generate Schedule"""
        with ui.step("Generate Schedule"):
            ui.label('Click the button below to generate the schedule')
            
            with ui.row().classes('w-full justify-center'):
                ui.button(
                    '🚀 Generate Schedule',
                    on_click=self.on_generate,
                    color='primary'
                ).classes('w-1/2')
            
            with ui.stepper_navigation():
                ui.button('Back', on_click=stepper.previous).props('flat')