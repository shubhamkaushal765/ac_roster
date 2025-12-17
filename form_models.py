"""
Data models for form inputs and configuration
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from nicegui import ui


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