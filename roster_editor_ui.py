"""
Roster editor UI component for adding SOS officers and editing schedules
"""

import re
from nicegui import ui
from typing import Dict, Optional, Callable
from acroster.config import NUM_SLOTS, START_HOUR
from acroster.time_utils import generate_time_slots, get_end_time_slots, clean_time
from acroster.schedule_utils import get_all_officer_ids
from ui_components import copyable_label


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

        # Raw SOS extraction state
        self.sos_extracted_data = []
        self.sos_confirmed = False

        # UI refs (set later)
        self.table_container = None
        self.reset_btn = None
        self.confirm_btn = None
        self.sos_table = None

    
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
                    ui.markdown("**Example**")
                    example_raw_text = '''ACAR SOS AM  
02 x GC  
- officer A (1000-1200)  
- officer B (2000-2200)  

03 x Bikes 2 (1300-1430,2030-2200)  
- officer C  
- officer D  
- officer E (1000-1130)'''
                    copyable_label(example_raw_text)
                    raw_sos_input = ui.textarea(
                        label='Paste here',
                        placeholder='ACAR SOS AM\n02 x GC\n...',
                        value='',
                    ).classes('w-full').props('rows=8')
                    with ui.row().classes('gap-2'):
                        ui.button(
                            '🔍 Extract SOS Officers',
                            on_click=lambda: self.extract_raw_sos(raw_sos_input.value)
                        ).props('color=primary')

                        self.reset_btn = ui.button(
                            '🔄 Reset',
                            on_click=self.reset_raw_sos
                        ).props('color=secondary')
                        self.reset_btn.disable()
                    
                    # Container for the table (will be populated after extraction)
                    self.table_container = ui.column().classes('w-full')
                
                # Manual Input Panel
                with ui.tab_panel(manual_tab):
                    manual_sos_example = "(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200"
                    ui.markdown("""
**Format:** `<optional counter no. at 1000 enclosed><sos_timing>`  
If an officer has multiple SOS timings, separate them with semicolons `;`.  
Optional pre-assigned counters must be enclosed in parentheses `()` before the time.

**Example**   
""")
                    copyable_label(manual_sos_example)
                    manual_sos_input = ui.textarea(
                        label='Type here',
                        placeholder='(AC22)1000-1300;1315-1430,...',
                        value='',
                    ).classes('w-full').props('rows=4')
                    ui.button(
                        '✅ Add Manual SOS Officers',
                        on_click=lambda: self.on_add_manual_sos(manual_sos_input.value)
                    ).props('color=secondary')
                
    def extract_raw_sos(self, raw_text: str):
        """Extract SOS officers from raw text and display editable table."""
        if not raw_text.strip():
            ui.notify('⚠️ Please enter raw text first', type='warning')
            return

        # Clear previous table completely
        if self.table_container:
            self.table_container.clear()
        
        # Reset state
        self.sos_extracted_data = []
        self.sos_table = None
        self.confirm_btn = None

        try:
            extracted_data = extract_officer_timings_with_deployment(raw_text)

            if extracted_data:
                # Ensure each row has a 'selected' checkbox
                for i, row in enumerate(extracted_data):
                    row.setdefault('id', i)
                    row.setdefault('selected', True)

                self.sos_extracted_data = extracted_data
                self.sos_confirmed = False

                # Notify success
                ui.notify(f'✅ Extracted {len(extracted_data)} officer records', type='positive')

                # Render editable table
                self.render_editable_table(extracted_data)

                # Enable Reset button
                if self.reset_btn:
                    self.reset_btn.enable()

            else:
                ui.notify('⚠️ No officer data could be extracted. Please check the format.', type='warning')

        except Exception as e:
            ui.notify(f'❌ Extraction failed: {str(e)}', type='negative')


    def render_editable_table(self, data: list[dict]):
        """Render editable table with Deployment, Selected, Name, Timing columns."""
        if not self.table_container:
            return

        with self.table_container:
            # Create the table
            self.sos_table = ui.table(
                columns=[
                    {'name': 'selected', 'label': '✓ Include', 'field': 'selected', 'align': 'center'},
                    {'name': 'deployment', 'label': 'Deployment', 'field': 'deployment', 'align': 'left'},
                    {'name': 'name', 'label': 'Officer Name', 'field': 'name', 'align': 'left'},
                    {'name': 'timing', 'label': 'Timing', 'field': 'timing', 'align': 'left'},
                ],
                rows=data,
                row_key='id'
            ).classes('w-full')

            # Body slot for editable cells and checkbox
            self.sos_table.add_slot('body', r'''
                <q-tr :props="props">
                    <q-td key="selected" :props="props" class="text-center">
                        <q-checkbox v-model="props.row.selected" dense @update:model-value="$parent.$emit('update:selected', props.row)"/>
                    </q-td>
                    <q-td key="deployment" :props="props">
                        {{ props.row.deployment }}
                        <q-popup-edit v-model="props.row.deployment" v-slot="scope"
                            @update:model-value="$parent.$emit('update:deployment', props.row)">
                            <q-input v-model="scope.value" dense autofocus counter @keyup.enter="scope.set"/>
                        </q-popup-edit>
                    </q-td>
                    <q-td key="name" :props="props">
                        {{ props.row.name }}
                        <q-popup-edit v-model="props.row.name" v-slot="scope"
                            @update:model-value="$parent.$emit('update:name', props.row)">
                            <q-input v-model="scope.value" dense autofocus counter @keyup.enter="scope.set"/>
                        </q-popup-edit>
                    </q-td>
                    <q-td key="timing" :props="props">
                        {{ props.row.timing }}
                        <q-popup-edit v-model="props.row.timing" v-slot="scope"
                            @update:model-value="$parent.$emit('update:timing', props.row)">
                            <q-input v-model="scope.value" dense autofocus counter @keyup.enter="scope.set"/>
                        </q-popup-edit>
                    </q-td>
                </q-tr>
            ''')

            # Event handlers to sync Python-side state
            self.sos_table.on('update:selected', lambda e: self._sync_table_row(e))
            self.sos_table.on('update:deployment', lambda e: self._sync_table_row(e))
            self.sos_table.on('update:name', lambda e: self._sync_table_row(e))
            self.sos_table.on('update:timing', lambda e: self._sync_table_row(e))

            # Instruction label
            ui.label('✏️ Double-click cells to edit before confirming').classes('text-sm text-gray-500 mt-1')

            # Confirm button
            self.confirm_btn = ui.button(
                '✅ Confirm & Add to Roster',
                on_click=self._confirm_add_sos
            ).classes('w-full mt-2').props('color=primary')

    def _confirm_add_sos(self):
        """Confirm and add selected SOS officers to roster"""
        selected_rows = [r for r in self.sos_extracted_data if r.get('selected')]
        
        if not selected_rows:
            ui.notify("⚠️ No officers selected. Please check at least one row.", type='warning')
            return

        invalid = [r for r in selected_rows if not r.get('timing')]
        if invalid:
            ui.notify(f"⚠️ {len(invalid)} selected row(s) have missing Timing.", type='warning')
            return

        # Build the SOS timings string with optional name/deployment
        # Format: (DEPLOYMENT|NAME)timing1;timing2, (DEPLOYMENT2|NAME2)timing3;timing4
        # If no name: just timing
        # If name only: (|NAME)timing
        # If deployment and name: (DEPLOYMENT|NAME)timing
        sos_timings_parts = []
        for row in selected_rows:
            timing = row.get('timing', '')
            deployment = row.get('deployment', '').strip()
            name = row.get('name', '').strip()
            
            # Build prefix based on what's available
            if deployment and name:
                # Both deployment and name: (DEP|NAME)timing
                prefix = f"({deployment}|{name})"
            elif name:
                # Name only: (|NAME)timing
                prefix = f"(|{name})"
            elif deployment:
                # Deployment only: (DEP|)timing
                prefix = f"({deployment}|)"
            else:
                # Neither: just timing
                prefix = ""
            
            sos_timings_parts.append(f"{prefix}{timing}")
        
        sos_timings_str = ', '.join(sos_timings_parts)
        
        # Call the backend handler to actually add the SOS officers
        self.on_add_manual_sos(sos_timings_str)
        
        # Clear the table and reset state
        self.reset_raw_sos()


    def _sync_table_row(self, e):
        """Update Python state when a row is edited (checkbox or cell)"""
        row_data = e.args  # Extract the actual dict from the event
        for i, r in enumerate(self.sos_extracted_data):
            if r['id'] == row_data['id']:
                self.sos_extracted_data[i] = row_data
                break


    def reset_raw_sos(self):
        """Reset the SOS extraction state"""
        self.sos_extracted_data = []
        self.sos_confirmed = False

        if self.table_container:
            self.table_container.clear()
        
        self.sos_table = None
        self.confirm_btn = None

        if self.reset_btn:
            self.reset_btn.disable()
        
        ui.notify('Reset completed', type='info')


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


# -------------------------------------------
# Helper function to extract SOS timings
# -------------------------------------------

def extract_officer_timings_with_deployment(text: str):
    """
    Extract officer timings from raw SOS text, including deployment column.
    """
    final_records = []
    text = text.replace('\t', ' ')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    current_deployment = None
    base_times = []
    officer_lines = []
    
    for line in lines + [""]:  # add empty line at end to flush last block
        if not re.match(r'^[-*]\s*', line):  # header line
            # Process previous block
            if officer_lines and current_deployment:
                for officer in officer_lines:
                    name = re.sub(r'\(.*?\)', '', officer).strip()
                    extra_match = re.search(r'\(([^)]*?)\)', officer)
                    extra_times = []
                    if extra_match:
                        extra_times = [t.strip() for t in extra_match.group(1).split(',') if t.strip()]
                    combined_times = base_times + extra_times
                    if combined_times:
                        timing_str = ';'.join(combined_times)
                        final_records.append({
                            'deployment': current_deployment,
                            'name': name,
                            'timing': timing_str
                        })
                officer_lines = []
                base_times = []
            
            # Parse new deployment header
            current_deployment = None
            # Remove leading "NN x " if present
            dep_line = re.sub(r'^\d+\s*x\s*', '', line, flags=re.IGNORECASE)
            # Keep text before parentheses or punctuation
            dep_line = re.split(r'\(|:|;', dep_line)[0].strip()
            # Skip headers with no officers (e.g., ACAR SOS AM)
            current_deployment = dep_line if dep_line and not re.search(r'ACAR SOS AM', dep_line, re.IGNORECASE) else None
            
            # Extract base times if header has parentheses
            base_match = re.search(r'\(([^)]*?)\)', line)
            if base_match:
                base_times = [t.strip() for t in base_match.group(1).split(',') if t.strip()]
            else:
                base_times = []

        else:
            # Officer line
            officer_line = re.sub(r'^[-*]\s*', '', line)
            officer_lines.append(officer_line)
    
    return final_records