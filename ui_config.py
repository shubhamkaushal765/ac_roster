"""
UI configuration constants and settings
"""

from enum import Enum


class UIMessages:
    """User-facing messages"""
    
    # Success messages
    SCHEDULE_GENERATED = '✅ Schedule generated successfully!'
    SOS_ADDED = '✅ Schedule updated with SOS officers!'
    VISUALIZATIONS_UPDATED = '✅ Visualizations updated!'
    
    # Warning messages
    NO_SOS_TIMINGS = "⚠️ Please paste SOS timings"
    NO_MANUAL_SOS = "⚠️ Please enter SOS timings"
    MAIN_OFFICERS_REQUIRED = "⚠️ 'Main Officers Reported' is required"
    SELECT_ALL_FIELDS = "⚠️ Please select all fields"
    SELECT_DIFFERENT_OFFICERS = "⚠️ Please select different officers"
    NO_SCHEDULE_TO_UPDATE = "⚠️ No schedule to update"
    NO_VALID_SOS = "⚠️ No valid SOS timings found"
    OFFICER_NOT_FOUND = "⚠️ Officer not found in schedule"
    START_BEFORE_END = "⚠️ Start time must be before end time"
    
    # Error messages
    ERROR_EXTRACTING_SOS = "❌ Error extracting SOS: {}"
    ERROR_ADDING_SOS = "❌ Error adding SOS: {}"
    ERROR_UPDATING = "❌ Error updating schedule: {}"
    ERROR_SWAPPING = "❌ Error swapping: {}"
    ERROR_DELETING = "❌ Error deleting: {}"
    ERROR_GENERATION = "❌ {}"
    
    # Info messages
    NO_SCHEDULE_YET = "ℹ️ No schedule generated yet. Please generate a schedule first."
    NO_HISTORY = "No history available"


class UILabels:
    """UI labels and titles"""
    
    # Main headers
    APP_TITLE = "Generate AC/DC roster (Morning)"
    MOBILE_TIP = "💡 For better display on mobile, please enable Desktop site in your browser settings."
    
    # Form steps
    STEP_MAIN_OFFICERS = "Main Officers"
    STEP_GL_COUNTERS = "Report to GL counters"
    STEP_HANDWRITTEN = "Handwritten Counters (1000-1030 only)"
    STEP_OT_COUNTERS = "OT counters"
    STEP_RO_RA = "RO/RA officers"
    STEP_OPTIONAL = "Optional"
    STEP_GENERATE = "Generate Schedule"
    
    # Sidebar
    ROSTER_EDITOR = '🗂️ Roster Editor'
    
    # Tabs
    TAB_ADD = '➕ Add'
    TAB_SWAP = '🔄 Swap'
    TAB_DELETE = '🗑️ Delete'
    
    # Metrics
    METRIC_MAIN = '👮 Main Officers'
    METRIC_SOS = '🆘 SOS Officers'
    METRIC_OT = '⏰ OT Officers'
    METRIC_TOTAL = '📊 Total Officers'
    
    # History
    HISTORY_TIMETABLE = '📊 Counter Timetable History'
    HISTORY_SCHEDULE = '👮 Officer Schedule History'
    HISTORY_MERGED = '📊 Counter Timetable + Officer Schedule History'
    HISTORY_LATEST = '📌 Latest - {}'
    HISTORY_SWIPE = 'Showing {} version(s) - Swipe to see history'


class UIStyles:
    """CSS styling constants"""
    
    FLEX_ROW_FULL = "width: 100%"
    FLEX_COL_3 = "flex: 3"
    FLEX_COL_2 = "flex: 2"
    FLEX_1 = "flex: 1"
    WIDTH_100 = "width: 100%"
    
    # Colors
    COLOR_GRAY = "color: gray; font-size: 14px;"
    COLOR_PRIMARY = "text-primary"
    COLOR_GRAY_600 = "text-gray-600"
    COLOR_GRAY_500 = "text-gray-500"
    COLOR_GRAY_700 = "text-gray-700"


class FormPlaceholders:
    """Placeholder text for form inputs"""
    
    SOS_RAW = 'ACAR SOS AM\n02 x GC\n...'
    SOS_MANUAL = '(AC22)1000-1300;1315-1430,...'


class FormExamples:
    """Example text for form hints"""
    
    MAIN_OFFICERS = "E.g 1-18 or 1,3,5-10"
    GL_COUNTERS = "E.g. 4AC1, 8AC11, 12AC21, 16AC31"
    HANDWRITTEN = "E.g. 3AC12, 5AC13"
    OT_COUNTERS = "E.g. 2,3,20"
    RO_RA = "E.g. 3RO2100,11RO1700,15RO2130"


class FormInstructions:
    """Instruction text for forms"""
    
    MAIN_OFFICERS = "Key in range of S/N assigned to {} car roster"
    GL_COUNTERS = "Which counter did Chops RM assign S/N 4, 8, 12, 16... from 1000-1130? Key in as <S/N>AC<counter no.>"
    HANDWRITTEN = "Did Chop RM manually change some of the first counters? Key in as <S/N>AC<counter no.>"
    OT_COUNTERS = "Which counters are manned by OT staff till 1030? Key in the list of counter no. separated by commas"
    RO_RA = "Which S/N is reporting late (RA) or leaving early (RO)? Key in as <S/N><RO or RA><counter no.>"
    
    SOS_PASTE = "Paste list of SOS officers given by Ops Rm here"
    SOS_MANUAL = """
**Example:** (AC22)1000-1300, 2000-2200, 1315-1430;2030-2200, (AC23)1000-1130;1315-1430;2030-2200  
**Format:** (`<optional counter no. at 1000>)<sos_timing>`  
If an officer has multiple SOS timings, separate them with semicolons `;`.  
Optional pre-assigned counters must be enclosed in parentheses `()` before the time.
"""


class ButtonLabels:
    """Button text labels"""
    
    NEXT = 'Next'
    BACK = 'Back'
    DONE = 'Done'
    GENERATE = '🚀 Generate Schedule'
    QUICK_GENERATE = '🚀 Quick Generate Schedule'
    EXTRACT_SOS = '🔍 Extract SOS Officers'
    ADD_MANUAL_SOS = '✅ Add Manual SOS Officers'
    SWAP = 'Swap Assignments'
    DELETE = 'Delete Assignment'


class CarouselConfig:
    """Carousel configuration"""
    
    HEIGHT_TIMETABLE = '1000px'
    HEIGHT_SCHEDULE = '700px'
    HEIGHT_MERGED = '1600px'
    HEIGHT_GRAPH2 = 'height: 600px;'


class SpinnerConfig:
    """Spinner configuration"""
    
    SIZE = 'lg'
    COLOR = 'primary'


class SliderConfig:
    """Slider configuration"""
    
    BEAM_WIDTH_MIN = 10
    BEAM_WIDTH_MAX = 100
    BEAM_WIDTH_DEFAULT = 20