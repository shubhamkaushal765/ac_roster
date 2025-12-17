"""
Reusable UI components for the roster generation application
"""

from nicegui import ui
from typing import List, Tuple, Optional
import plotly.graph_objects as go


class MetricsCard:
    """Reusable metrics card component"""
    
    @staticmethod
    def render(title: str, value: int):
        """Render a single metrics card"""
        with ui.column().style("flex: 1"):
            with ui.card().classes('w-full text-center p-4'):
                ui.label(title).classes('text-sm text-gray-500')
                ui.label(str(value)).classes('text-3xl font-bold')


class HistoryCarousel:
    """Generic carousel component for displaying history of visualizations"""
    
    def __init__(self, title: str, carousel_id: str, height: str = "1000px"):
        self.title = title
        self.carousel_id = carousel_id
        self.height = height
    
    def render(self, history: List[Tuple], show_stats: bool = True):
        """
        Render carousel with history
        
        Args:
            history: List of (fig, stats, timestamp, description) tuples
            show_stats: Whether to show statistics textarea
        """
        if not history:
            ui.label("No history available").classes('text-gray-500')
            return
        
        ui.separator()
        ui.label(self.title).classes('text-lg font-bold')
        ui.label(f'Showing {len(history)} version(s) - Swipe to see history').classes('text-sm text-gray-500')
        
        with ui.carousel(
            animated=True, 
            arrows=True, 
            navigation=True, 
            value=f'{self.carousel_id}_{len(history) - 1}'
        ).props(f'height={self.height}').classes('w-full'):
            
            for idx, entry in enumerate(history):
                fig, stats, timestamp, description = entry
                
                with ui.carousel_slide(name=f'{self.carousel_id}_{idx}'):
                    with ui.column().classes('w-full'):
                        # Version header
                        if idx == len(history) - 1:
                            ui.label(f'📌 Latest - {timestamp}').classes('text-lg font-bold mb-1 text-primary')
                        else:
                            ui.label(f'{timestamp}').classes('text-lg font-bold mb-1 text-gray-600')
                        
                        # Description
                        ui.markdown(description).classes('text-sm text-gray-700 mb-3 whitespace-pre-line')
                        
                        # Graph
                        ui.plotly(fig).classes('w-full')
                        
                        # Stats (optional)
                        if show_stats and stats:
                            ui.textarea(
                                label='Counter Manning Statistics',
                                value=stats,
                            ).classes('w-full').props('rows=10')


class MergedHistoryCarousel:
    """Carousel showing both timetable and schedule together"""
    
    def __init__(self, title: str = '📊 Counter Timetable + Officer Schedule History'):
        self.title = title
    
    def render(self, timetable_history: List[Tuple], schedule_history: List[Tuple]):
        """
        Render merged carousel with both timetable and schedule
        
        Args:
            timetable_history: List of (fig, stats, timestamp, description) for timetables
            schedule_history: List of (fig, stats, timestamp, description) for schedules
        """
        if not timetable_history or not schedule_history:
            ui.label("No history available").classes('text-gray-500')
            return
        
        ui.separator()
        ui.label(self.title).classes('text-lg font-bold')
        ui.label(f'Showing {len(timetable_history)} version(s) - Swipe to see history').classes('text-sm text-gray-500')

        with ui.carousel(
            animated=True,
            arrows=True,
            navigation=True,
            value=f'merged_{len(timetable_history) - 1}',
        ).props('height=1600px').classes('w-full'):

            for idx, (tt_entry, sch_entry) in enumerate(zip(timetable_history, schedule_history)):
                fig1, stats1, timestamp1, description1 = tt_entry
                fig2, stats2, timestamp2, description2 = sch_entry

                with ui.carousel_slide(name=f'merged_{idx}'):
                    with ui.column().classes('w-full'):
                        # Version header
                        if idx == len(timetable_history) - 1:
                            ui.label(f'📌 Latest - {timestamp1}').classes('text-lg font-bold mb-1 text-primary')
                        else:
                            ui.label(f'{timestamp1}').classes('text-lg font-bold mb-1 text-gray-600')
                        
                        ui.markdown(description1).classes('text-sm text-gray-700 mb-3 whitespace-pre-line')
                        
                        # Counter Timetable
                        ui.label('📊 Counter Timetable').classes('text-md font-bold mt-4')
                        ui.plotly(fig1).classes('w-full')
                        
                        if stats1:
                            ui.textarea(
                                label='Counter Manning Statistics',
                                value=stats1
                            ).classes('w-full').props('rows=10')
                        
                        # Officer Schedule
                        ui.label('👮 Officer Schedule').classes('text-md font-bold mt-6')
                        ui.plotly(fig2).classes('w-full').style('height: 600px;')


class StepperNavigation:
    """Reusable stepper navigation component"""
    
    @staticmethod
    def render(stepper, show_next: bool = True, show_back: bool = True, 
               next_label: str = 'Next', on_next=None):
        """Render stepper navigation buttons"""
        with ui.stepper_navigation():
            if show_next:
                if on_next:
                    ui.button(next_label, on_click=on_next)
                else:
                    ui.button(next_label, on_click=stepper.next)
            if show_back:
                ui.button('Back', on_click=stepper.previous).props('flat')


def copyable_label(text_to_copy: str):
    with ui.row().classes('items-center gap-2'):
        ui.markdown(text_to_copy).classes('font-mono')
        ui.button(
            icon='content_copy',
            on_click=lambda: (
                ui.clipboard.write(text_to_copy),
                ui.notify('Copied to clipboard', type='positive')
            )
        ).props('flat dense')