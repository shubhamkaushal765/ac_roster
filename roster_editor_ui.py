"""
Roster editor UI component for adding SOS officers and editing schedules
"""

import re
from nicegui import ui
from typing import Dict, Optional, Callable
from acroster.config import NUM_SLOTS, START_HOUR
from acroster.time_utils import generate_time_slots, get_end_time_slots
from acroster.schedule_utils import get_all_officer_ids


class RosterEditorUI:
    """UI component for editing rosters (Add SOS, Swap, Delete)"""
    
    def __init__(
        self, 
        edited_schedule: Optional[Dict],
        on_extract_sos: Callable,
        on_add_manual_sos: Callable,
        on_swap_assignments: Callable,
        on_delete_assignment: Callable
    ):
        self.edited_schedule = edited_schedule
        self.on_extract_sos = on_extract_sos
        self.on_add_manual_sos = on_add_manual_sos
        self.on_swap_assignments = on_swap_assignments
        self.on_delete_assignment = on_delete_assignment
    
    def render(self):
        """Render the roster editor sidebar"""
        ui.label('🗂️ Roster Editor').classes('text-lg font-bold')

        if self.edited_schedule is None:
            ui.label(
                "ℹ️ No schedule generated yet. Please generate a schedule first."
            ).style("color: gray; font-size: 14px;")
            return
        
        officer_ids = self._get_sorted_officer_ids()
        time_slots = generate_time_slots(START_HOUR, NUM_SLOTS)
        
        # Horizontal tabs for Add / Swap / Delete
        with ui.tabs().props('dense align=justify').classes('w-full') as main_tabs:
            add_tab = ui.tab('➕ Add')
            swap_tab = ui.tab('🔄 Swap')
            delete_tab = ui.tab('🗑️ Delete')

        with ui.tab_panels(main_tabs, value=add_tab, animated=False).classes('w-full'):
            self._render_add_sos_panel(add_tab)
            self._render_swap_panel(swap_tab, officer_ids, time_slots)
            self._render_delete_panel(delete_tab, officer_ids, time_slots)
    
    def _get_sorted_officer_ids(self):
        """Get naturally sorted list of officer IDs"""
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() 
                    for text in re.split('([0-9]+)', s)]
        
        officer_ids = get_all_officer_ids(self.edited_schedule)
        return sorted(officer_ids, key=natural_sort_key)
    
    def _render_add_sos_panel(self, tab):
        """Render Add SOS Officers panel"""
        with ui.tab_panel(tab):
            ui.label("Paste list of SOS officers given by Ops Rm here")
            
            with ui.tabs().props('dense align=justify').classes('w-full') as add_subtabs:
                raw_tab = ui.tab('Text Input (Recommended)')
                manual_tab = ui.tab('Manual Input')
            
            with ui.tab_panels(add_subtabs, value=raw_tab, animated=False).classes('w-full'):
                # Raw Input Panel
                with ui.tab_panel(raw_tab):
                    raw_sos_input = ui.textarea(
                        label='Paste SOS timings (raw format)',
                        placeholder='ACAR SOS AM\n02 x GC\n...',
                        value='',
                    ).classes('w-full').props('rows=8')
                    ui.button(
                        '🔍 Extract SOS Officers',
                        on_click=lambda: self.on_extract_sos(raw_sos_input.value)
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
                        on_click=lambda: self.on_add_manual_sos(manual_sos_input.value)
                    ).props('color=secondary')
    
    def _render_swap_panel(self, tab, officer_ids, time_slots):
        """Render Swap Assignments panel"""
        with ui.tab_panel(tab):
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
                on_click=lambda: self.on_swap_assignments(
                    swap_officer1.value,
                    swap_officer2.value,
                    swap_start.value,
                    swap_end.value
                )
            ).props('color=primary')
    
    def _render_delete_panel(self, tab, officer_ids, time_slots):
        """Render Delete Assignments panel"""
        with ui.tab_panel(tab):
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
                on_click=lambda: self.on_delete_assignment(
                    del_officer.value,
                    del_start.value,
                    del_end.value
                )
            ).props('color=negative')