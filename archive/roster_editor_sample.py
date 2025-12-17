"""
roster_editor.py - Add to acroster/ directory
Handles the UI and logic for ad-hoc roster editing
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple

class RosterEditor:
    def __init__(self, roster_matrix: np.ndarray, officer_ids: list):
        """
        Initialize the roster editor
        
        Args:
            roster_matrix: mxn matrix where m=officers, n=48 time slots
            officer_ids: List of officer identifiers
        """
        self.roster = roster_matrix.copy()
        self.officer_ids = officer_ids
        self.time_slots = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(48)]
        
    def delete_officer_slots(self, officer_idx: int, slot_start: int, slot_end: int):
        """Delete officer assignment from slot_start to slot_end (inclusive)"""
        if 0 <= officer_idx < len(self.officer_ids) and 0 <= slot_start <= slot_end < 48:
            self.roster[officer_idx, slot_start:slot_end+1] = 0  # 0 means unassigned
            return True
        return False
    
    def swap_officers_slots(self, officer1_idx: int, officer2_idx: int, 
                           slot_start: int, slot_end: int):
        """Swap counter assignments between two officers for specified slots"""
        if (0 <= officer1_idx < len(self.officer_ids) and 
            0 <= officer2_idx < len(self.officer_ids) and
            0 <= slot_start <= slot_end < 48):
            
            temp = self.roster[officer1_idx, slot_start:slot_end+1].copy()
            self.roster[officer1_idx, slot_start:slot_end+1] = self.roster[officer2_idx, slot_start:slot_end+1]
            self.roster[officer2_idx, slot_start:slot_end+1] = temp
            return True
        return False
    
    def add_officer_slots(self, officer_idx: int, counter_no: int, 
                         slot_start: int, slot_end: int):
        """Assign officer to a counter for specified slots"""
        if (0 <= officer_idx < len(self.officer_ids) and 
            counter_no > 0 and 
            0 <= slot_start <= slot_end < 48):
            
            self.roster[officer_idx, slot_start:slot_end+1] = counter_no
            return True
        return False
    
    def get_roster_dataframe(self) -> pd.DataFrame:
        """Convert roster matrix to a formatted DataFrame for display"""
        df = pd.DataFrame(
            self.roster,
            index=self.officer_ids,
            columns=self.time_slots
        )
        return df
    
    def get_roster_matrix(self) -> np.ndarray:
        """Return the current roster matrix"""
        return self.roster.copy()


def display_roster_editor(roster_matrix: np.ndarray, officer_ids: list):
    """
    Streamlit UI for roster editing
    
    Args:
        roster_matrix: Initial roster matrix
        officer_ids: List of officer IDs
    """
    st.title("🗓️ Roster Editor")
    
    # Initialize editor in session state
    if 'editor' not in st.session_state:
        st.session_state.editor = RosterEditor(roster_matrix, officer_ids)
        st.session_state.edit_history = []
    
    editor = st.session_state.editor
    
    # Display current roster
    st.subheader("Current Roster")
    df = editor.get_roster_dataframe()
    
    # Color code the roster display
    def highlight_roster(val):
        if val == 0:
            return 'background-color: #f0f0f0'
        else:
            colors = ['#ffcccc', '#ccffcc', '#ccccff', '#ffffcc', '#ffccff', '#ccffff']
            return f'background-color: {colors[int(val) % len(colors)]}'
    
    st.dataframe(
        df.style.applymap(highlight_roster),
        height=400,
        use_container_width=True
    )
    
    st.divider()
    
    # Edit operations
    st.subheader("Edit Operations")
    
    tab1, tab2, tab3 = st.tabs(["🗑️ Delete", "🔄 Swap", "➕ Add"])
    
    # Helper function to convert time to slot index
    def time_to_slot(time_str: str) -> int:
        return editor.time_slots.index(time_str)
    
    # TAB 1: DELETE
    with tab1:
        st.write("Remove officer assignment from specified time range")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            del_officer = st.selectbox("Officer", officer_ids, key="del_officer")
        with col2:
            del_start = st.selectbox("From Time", editor.time_slots, key="del_start")
        with col3:
            del_end = st.selectbox("To Time", editor.time_slots, 
                                   index=min(47, time_to_slot(del_start) + 3),
                                   key="del_end")
        
        if st.button("Delete Assignment", type="primary", key="del_btn"):
            officer_idx = officer_ids.index(del_officer)
            start_idx = time_to_slot(del_start)
            end_idx = time_to_slot(del_end)
            
            if start_idx <= end_idx:
                success = editor.delete_officer_slots(officer_idx, start_idx, end_idx)
                if success:
                    st.success(f"✅ Deleted {del_officer} from {del_start} to {del_end}")
                    st.session_state.edit_history.append(
                        f"Deleted {del_officer}: {del_start}-{del_end}"
                    )
                    st.rerun()
                else:
                    st.error("❌ Invalid operation")
            else:
                st.error("❌ End time must be after start time")
    
    # TAB 2: SWAP
    with tab2:
        st.write("Swap counter assignments between two officers")
        
        col1, col2 = st.columns(2)
        with col1:
            swap_officer1 = st.selectbox("Officer 1", officer_ids, key="swap_off1")
        with col2:
            swap_officer2 = st.selectbox("Officer 2", officer_ids, 
                                         index=min(1, len(officer_ids)-1),
                                         key="swap_off2")
        
        col3, col4 = st.columns(2)
        with col3:
            swap_start = st.selectbox("From Time", editor.time_slots, key="swap_start")
        with col4:
            swap_end = st.selectbox("To Time", editor.time_slots,
                                    index=min(47, time_to_slot(swap_start) + 3),
                                    key="swap_end")
        
        if st.button("Swap Assignments", type="primary", key="swap_btn"):
            if swap_officer1 == swap_officer2:
                st.error("❌ Cannot swap officer with themselves")
            else:
                officer1_idx = officer_ids.index(swap_officer1)
                officer2_idx = officer_ids.index(swap_officer2)
                start_idx = time_to_slot(swap_start)
                end_idx = time_to_slot(swap_end)
                
                if start_idx <= end_idx:
                    success = editor.swap_officers_slots(officer1_idx, officer2_idx, 
                                                        start_idx, end_idx)
                    if success:
                        st.success(f"✅ Swapped {swap_officer1} ↔ {swap_officer2} from {swap_start} to {swap_end}")
                        st.session_state.edit_history.append(
                            f"Swapped {swap_officer1} ↔ {swap_officer2}: {swap_start}-{swap_end}"
                        )
                        st.rerun()
                    else:
                        st.error("❌ Invalid operation")
                else:
                    st.error("❌ End time must be after start time")
    
    # TAB 3: ADD
    with tab3:
        st.write("Assign officer to a counter for specified time range")
        
        col1, col2 = st.columns(2)
        with col1:
            add_officer = st.selectbox("Officer", officer_ids, key="add_officer")
        with col2:
            add_counter = st.number_input("Counter No.", min_value=1, max_value=20, 
                                         value=1, key="add_counter")
        
        col3, col4 = st.columns(2)
        with col3:
            add_start = st.selectbox("From Time", editor.time_slots, key="add_start")
        with col4:
            add_end = st.selectbox("To Time", editor.time_slots,
                                   index=min(47, time_to_slot(add_start) + 3),
                                   key="add_end")
        
        if st.button("Add Assignment", type="primary", key="add_btn"):
            officer_idx = officer_ids.index(add_officer)
            start_idx = time_to_slot(add_start)
            end_idx = time_to_slot(add_end)
            
            if start_idx <= end_idx:
                success = editor.add_officer_slots(officer_idx, add_counter, 
                                                   start_idx, end_idx)
                if success:
                    st.success(f"✅ Assigned {add_officer} to Counter {add_counter} from {add_start} to {add_end}")
                    st.session_state.edit_history.append(
                        f"Added {add_officer} to Counter {add_counter}: {add_start}-{add_end}"
                    )
                    st.rerun()
                else:
                    st.error("❌ Invalid operation")
            else:
                st.error("❌ End time must be after start time")
    
    # Edit history sidebar
    with st.sidebar:
        st.subheader("📝 Edit History")
        if st.session_state.edit_history:
            for i, edit in enumerate(reversed(st.session_state.edit_history[-10:]), 1):
                st.text(f"{i}. {edit}")
        else:
            st.info("No edits yet")
        
        if st.button("Reset Roster", type="secondary"):
            st.session_state.editor = RosterEditor(roster_matrix, officer_ids)
            st.session_state.edit_history = []
            st.rerun()
        
        if st.button("Export Roster"):
            st.download_button(
                label="Download CSV",
                data=df.to_csv(),
                file_name="edited_roster.csv",
                mime="text/csv"
            )