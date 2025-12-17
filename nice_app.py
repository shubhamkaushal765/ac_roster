"""
NiceGUI web application for AC Roster Generation (Morning Shift)
Refactored for maintainability and scalability
"""

import os
import traceback
from nicegui import ui, app
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from acroster.db_handlers import get_last_inputs
from form_models import FormInputs, InputDefaults
from form_stepper import FormStepperUI
from roster_editor_ui import RosterEditorUI
from roster_operations import RosterOperations
from ui_components import MetricsCard, MergedHistoryCarousel


class NiceGUICompatibleCSP(BaseHTTPMiddleware):
    """Middleware for Content Security Policy"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        response.headers.pop("Content-Security-Policy", None)
        response.headers.pop("Content-Security-Policy-Report-Only", None)
        
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


class RosterGenerationUI:
    """Main UI controller for roster generation"""
    
    def __init__(self):
        # Data models
        self.saved_inputs = get_last_inputs() or {}
        self.defaults = InputDefaults()
        self.inputs = FormInputs(self.saved_inputs)
        
        # Business logic
        self.operations = RosterOperations()
        
        # UI containers
        self.result_container: Optional[ui.column] = None
        self.summary_html: Optional[ui.html] = None
        self.spinner: Optional[ui.spinner] = None
        self.sidebar_container: Optional[ui.column] = None
        
    def render(self):
        """Render the complete UI"""
        self._render_header()

        with ui.row().style("width: 100%"):
            # Main content area
            with ui.column().style("flex: 3"):
                self._render_main_form()
                self.result_container = ui.column()
            
            # Sidebar
            with ui.column().style("flex: 2"):
                self.sidebar_container = ui.column().style("width: 100%")
                self._render_sidebar()
    
    def _render_header(self):
        """Render page header"""
        ui.label("Generate AC/DC roster (Morning)").classes("text-2xl font-bold")
        ui.label("💡 For better display on mobile, please enable Desktop site in your browser settings.")\
            .style("font-size:14px; color:gray; margin-top:-10px;")
    
    def _render_main_form(self):
        """Render the main form with summary and stepper"""
        # Summary card with toggle button
        with ui.card().classes('w-full mb-4'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Your Inputs:').classes('text-bold')
                self.inputs.toggle_form_btn = ui.button(
                    icon='expand_less',
                    on_click=self._toggle_form_visibility
                ).props('flat dense').tooltip('Show/Hide Form')
            
            self.summary_html = ui.html('', sanitize=False)
            self._update_summary()
        
        # Stepper in a collapsible container
        self.inputs.form_container = ui.column().classes('w-full')
        with self.inputs.form_container:
            form_stepper = FormStepperUI(
                inputs=self.inputs,
                defaults=self.defaults,
                on_generate=self._run_generation,
                on_update_summary=self._update_summary
            )
            form_stepper.render()
            
            # Spinner
            self.spinner = ui.spinner(size='lg').props('color=primary')
            self.spinner.visible = False
    
    def _render_sidebar(self):
        """Render sidebar with roster editor"""
        with self.sidebar_container:
            editor = RosterEditorUI(
                edited_schedule=self.operations.edited_schedule,
                on_extract_sos=self._handle_extract_sos,
                on_add_manual_sos=self._handle_add_manual_sos,
                on_swap_assignments=self._handle_swap_assignments,
                on_delete_assignment=self._handle_delete_assignment
            )
            editor.render()
    
    def _update_summary(self):
        """Update the summary card with current input values"""
        if self.summary_html:
            self.summary_html.set_content(self.inputs.get_summary_html())
    
    def _toggle_form_visibility(self):
        """Toggle form visibility"""
        if self.inputs.form_container.visible:
            self.inputs.form_container.visible = False
            self.inputs.toggle_form_btn.props('icon=expand_more')
        else:
            self.inputs.form_container.visible = True
            self.inputs.toggle_form_btn.props('icon=expand_less')
    
    def _hide_form_after_generation(self):
        """Hide form after successful generation"""
        if hasattr(self.inputs, 'form_container'):
            self.inputs.form_container.visible = False
            self.inputs.toggle_form_btn.props('icon=expand_more')
    
    # === Generation Logic ===
    
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
            
            # Generate schedule
            counter_matrix, final_counter_matrix, officer_schedule, output_text = \
                self.operations.generate_schedule(values)
            
            self.spinner.visible = False
            ui.notify('✅ Schedule generated successfully!', type='positive')
            self._hide_form_after_generation()
            
            # Render results
            with self.result_container:
                self._render_results()
            
            # Update sidebar
            self.sidebar_container.clear()
            with self.sidebar_container:
                self._render_sidebar()
        
        except Exception as e:
            self.spinner.visible = False
            ui.notify(f'❌ {str(e)}', type='negative')
            
            if self.inputs.show_debug.value:
                with self.result_container:
                    ui.code(traceback.format_exc())
    
    def _render_results(self):
        """Render all result visualizations and data"""
        # Officer counts
        counts = self.operations.get_officer_counts()
        
        with ui.row().classes('w-full gap-4'):
            MetricsCard.render('👮 Main Officers', counts['main'])
            MetricsCard.render('🆘 SOS Officers', counts['sos'])
            MetricsCard.render('⏰ OT Officers', counts['ot'])
            MetricsCard.render('📊 Total Officers', counts['total'])
        
        # Optimization penalty
        if self.operations.current_orchestrator.penalty is not None:
            ui.notify(
                f"🎯 Optimization Penalty: {self.operations.current_orchestrator.penalty:.2f}", 
                type='positive'
            )
        
        # Merged gallery with both timetable and schedule
        carousel = MergedHistoryCarousel()
        carousel.render(
            self.operations.timetable_history,
            self.operations.schedule_history
        )
    
    # === Roster Editor Handlers ===
    
    def _handle_extract_sos(self, raw_text: str):
        """Handle SOS extraction from raw text"""
        if not raw_text.strip():
            ui.notify("⚠️ Please paste SOS timings", type='warning')
            return
        
        try:
            sos_timings_str = self.operations.extract_sos_from_text(raw_text)
            
            if not sos_timings_str:
                ui.notify("⚠️ No valid SOS timings found", type='warning')
                return
            
            self._rerun_with_sos(sos_timings_str)
            
        except Exception as e:
            ui.notify(f"❌ Error extracting SOS: {str(e)}", type='negative')
            if self.inputs.show_debug.value:
                ui.code(traceback.format_exc())
    
    def _handle_add_manual_sos(self, manual_text: str):
        """Handle manually entered SOS officers"""
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
        if not self.operations.current_orchestrator:
            ui.notify("⚠️ No schedule to update", type='warning')
            return
        
        self.spinner.visible = True
        
        try:
            counter_matrix, officer_schedule, stats, description = \
                self.operations.add_sos_officers(sos_timings)
            
            self.spinner.visible = False
            ui.notify('✅ Schedule updated with SOS officers!', type='positive')
            
            # Re-render results
            self.result_container.clear()
            with self.result_container:
                self._render_results()
            
        except Exception as e:
            self.spinner.visible = False
            ui.notify(f"❌ Error updating schedule: {str(e)}", type='negative')
            if self.inputs.show_debug.value:
                ui.code(traceback.format_exc())
    
    def _handle_swap_assignments(self, officer1: str, officer2: str, 
                                 start_time: str, end_time: str):
        """Handle swapping counter assignments"""
        if not all([officer1, officer2, start_time, end_time]):
            ui.notify("⚠️ Please select all fields", type='warning')
            return
        
        if officer1 == officer2:
            ui.notify("⚠️ Please select different officers", type='warning')
            return
        
        try:
            description = self.operations.swap_assignments(
                officer1, officer2, start_time, end_time
            )
            
            ui.notify(f"✅ {description}", type='positive')
            
            # Re-render results
            self.result_container.clear()
            with self.result_container:
                self._render_results()
                
        except Exception as e:
            ui.notify(f"❌ Error swapping: {str(e)}", type='negative')
    
    def _handle_delete_assignment(self, officer: str, start_time: str, end_time: str):
        """Handle deleting counter assignments"""
        if not all([officer, start_time, end_time]):
            ui.notify("⚠️ Please select all fields", type='warning')
            return
        
        try:
            description = self.operations.delete_assignment(
                officer, start_time, end_time
            )
            
            ui.notify(f"✅ {description}", type='positive')
            
            # Re-render results
            self.result_container.clear()
            with self.result_container:
                self._render_results()
                
        except Exception as e:
            ui.notify(f"❌ Error deleting: {str(e)}", type='negative')


def main():
    """Main entry point"""
    # Add CSP middleware
    #app.add_middleware(NiceGUICompatibleCSP)
    
    # Set window title
    ui.window_title = "Generate ACar/DCar Roster Morning"
    
    # Create and render app
    myapp = RosterGenerationUI()
    myapp.render()
    
    # Run server
    ui.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 3000)),
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()