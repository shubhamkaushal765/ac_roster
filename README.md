# ACar/DCar Roster Generator - Morning Shift

A sophisticated scheduling system that allocates break time and counters for each SOS officer. Ensures maximum running counters while fulfilling break constraints (no more than 2.5h at any counter).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Flow](#system-flow)
- [Module Documentation](#module-documentation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

- **48 time slots per shift**: 15-minute intervals from 10:00 to 22:00
- **Two operation modes**: Arrival (41 counters) and Departure (37 counters)
- **Three officer types**: Main officers (fixed templates), SOS officers (flexible timimg), OT officers (1000-1030)
- **Intelligent optimization**: Find the best break timing for each officers to optimize running counters
- **Interactive editing**: Handles manual counter amendments with full edit history

### Key Capabilities

- ✅ Automated roster generation from simple text inputs
- ✅ Constraint-aware break scheduling for SOS officers (max 10 consecutive working slots)
- ✅ Counter assignment maximizes running counters, with new counters opened according to the priority list
- ✅ Real-time visualization with interactive Plotly charts
- ✅ Edit history tracking and undo functionality
- ✅ SQLite persistence for audit trails and history

## ✨ Features

### Core Functionality

- **Template-Based Main Rosters**: Pre-defined for main officers
- **Late Arrival/Early Departure Handling**: `RA` (Report At) and `RO` (Report Off)
- **Accepts Manual Counter Allocation**: Allows user to update counters assigned by Chops Room
- **OT Officer Integration**: Add counters occupied by OT officers since the previous shift

### Advanced Scheduling

- **SOS Officer Optimization**: Beam search algorithm minimizes manning fluctuations
- **Break Schedule Generation**: Recursive algorithm generates valid break combinations
- **Connected Interval Packing**: Smart counter assignment prioritizing continuity

### User Interface (Streamlit)

- **Three Visualization Modes**:
  - Counter timetable (which officer at each counter)
  - Officer timetable (which counter each officer is at)
  - Manning statistics (coverage by zone and time)
- **Interactive Roster Editor**: Swap, delete, and add assignments post-generation
- **Raw Text Parsing**: Auto-extract officer timings from operations room messages

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  Layer 1: UI/Presentation                   │
│  app.py                                     │  ← Entry point
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│  Layer 2: Orchestration                     │
│  orchestrator_pipe.py                  │  ← Coordinates services
└──────────────┬──────────────────────────────┘
               │
               ├──────────────────────────────┐
               │                              │
┌──────────────▼──────────────┐  ┌───────────▼──────────────┐
│  Layer 3: Domain Services   │  │  Layer 3: Domain Services│
│  roster_builder.py          │  │  sos_scheduler.py        │
│  assignment_engine.py       │  │  optimization.py         │
│  statistics.py              │  │  plotter.py              │
└──────────────┬──────────────┘  └────────────┬─────────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 4: Domain Models (Data + Behavior)                   │
│  officer.py, counter.py                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│  Layer 5: Infrastructure/Utilities                          │
│  config.py, time_utils.py, database.py                      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd acroster
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.17.0
sqlalchemy>=2.0.0
```

## 🎬 Quick Start

### Running the Web Application

```bash
streamlit run app.py
```

### Basic Usage Flow

1. **Select Operation Mode**: Choose Arrival or Departure
2. **Key in Officers' S/N in Main Roster**: e.g., `1-18`
3. **Configure Counters**:
   - GL counters: `4AC1, 8AC11, 12AC21, 16AC31`
   - OT counters: `2,20,40`
4. **Add Adjustments**: e.g., `3RO2100` (officer 3 reports off at 21:00)
5. **Generate Schedule**: Click "Generate Schedule" button
6. **Add SOS Officers** (optional): Use Roster Editor to add support officers
7. **Make Additional Edits** (optional): Swap, delete, or modify assignments

```python
from acroster.orchestrator import RosterAlgorithmOrchestrator

from acroster.config import OperationMode

# Create orchestrator
orchestrator = RosterAlgorithmOrchestrator(mode=OperationMode.ARRIVAL)

# Generate roster
results = orchestrator.run(
    main_officers_reported="1-18",
    report_gl_counters="4AC1, 8AC11, 12AC21, 16AC31",
    sos_timings="",  # Add SOS later via editor
    ro_ra_officers="3RO2100, 11RO1700",
    handwritten_counters="3AC12,5AC13",
    ot_counters="2,20,40"
)

# Access results
main_matrix, final_matrix, officer_schedule, statistics = results

```

## 🔄 Pipeline

```
1. app.py (User clicks "Generate Schedule")
   ↓
2. algorithm_orchestrator.py → RosterAlgorithmOrchestrator.run()
   ↓
3. config.py → Load MODE_CONFIG, roster_templates
   ↓
4. roster_builder.py → RosterBuilder.build_main_officers()
   │  ├─→ time_utils.py → hhmm_to_slot() for RA/RO parsing
   │  └─→ officer.py → Create MainOfficer instances
   ↓
5. roster_builder.py → LastCounterAssigner.assign_last_counters()
   │  ├─→ assignment_engine.py → Convert to CounterMatrix
   │  └─→ counter.py → Query empty counters
   ↓
6. assignment_engine.py → CounterAssignmentEngine.assign_ot_officers()
   │  └─→ officer.py → Create OTOfficer instances
   │  └─→ counter.py → Update CounterMatrix
   ↓
7. sos_scheduler.py → SOSOfficerBuilder.build_sos_officers()
   │  ├─→ time_utils.py → Parse time ranges
   │  └─→ officer.py → Create SOSOfficer instances
   ↓
8. sos_scheduler.py → BreakScheduleGenerator.generate_break_schedules()
   │  └─→ Recursive break placement algorithm
   ↓
9. optimization.py → ScheduleOptimizer.optimize()
    │  └─→ Beam search to select best break schedules
    ↓
10. assignment_engine.py → SOSAssignmentEngine.assign_sos_officers()
    │  └─→ counter.py → Interval packing algorithm
    ↓
11. statistics.py → StatisticsGenerator.generate_statistics()
    │  └─→ time_utils.py → slot_to_hhmm() for display
    ↓
12. orchestrator_pipe.py → Return results to app
    ↓
13. plotter.py → Generate visualizations
    │  └─→ time_utils.py → Format time labels
    ↓
14. app.py → Display results to user
```

### Data Transformation

```
Raw User Input (strings)
  ↓ [roster_builder.py]
Officer Objects (domain models)
  ↓ [assignment_engine.py]
CounterMatrix (structured data)
  ↓ [sos_scheduler.py + optimization.py]
CounterMatrix with SOS (enhanced data)
  ↓ [statistics.py]
Text Statistics (formatted output)
  ↓ [plotter.py]
Plotly Figures (visualizations)
  ↓ [app.py]
User sees results
```

## 📚 Module Documentation

#### `app.py`

**Streamlit web application UI**

- Three-panel visualization (counter timetable, officer timetable, statistics)
- Roster editor to `swap`/`delete`/`add` officers
- History sidebar with 🗑️ icon
- Real-time chart updates with Plotly
- Mobile-responsive layout

---

#### `orchestrator_pipe.py`

**Pipeline**:

1. Build main officer rosters
2. Assign last counters (for officers with unassigned last counters in main roster)
3. Add overtime officers
4. Build and optimize SOS officer schedules
5. Assign SOS officers to counters
6. Generate statistics

---

#### `roster_builder.py`

\*\*Initialize counter timetable from main officers only"

- Validates RA/RO adjustments (late arrival/early departure)
- Handles ad-hoc counter changes from Chops Room

**Key Classes**:

- `RosterBuilder`: Main officer roster construction
- `LastCounterAssigner`: Assigns last counters to officers in S/N `3,7,11...` from 2030-2200

**Input Formats**:

```python
main_officers_reported = "1,3,5-10,15"    # S/N
ro_ra_officers = "3RO2100, 11RA1030"      # RO = report off, RA = report at
report_gl_counters = "4AC1, 8AC11"        # Officer 4 at counter 1 from 1000 to 1115
handwritten_counters = "3AC12"            # Ad-hoc counter changes from Chops Rooms (assumes 1000-1030 only)
```

#### `sos_scheduler.py`

- Parses SOS officers' availability time range
- Generates valid break schedules with constraints
- Supports pre-assigned counter specifications

**Key Classes**:

- `AvailabilityParser`: Converts timing strings to binary arrays
- `SOSOfficerBuilder`: Creates SOS officer objects
- `BreakScheduleGenerator`: Recursive break placement algorithm

**Break Constraints**:

- Maximum 10 consecutive working slots without break
- Minimum 4-slot gap between breaks
- Break patterns: 36+ slots = 3 breaks, 20-35 slots = 2 breaks, 11-19 slots = 1 break

**Input Format**:

```python
# Basic timing
"1000-1300, 2000-2200"

# Multiple ranges with semicolon
"1315-1430;2030-2200"

# Pre-assigned counter
"(AC22)1000-1300, 2000-2200"
```

#### `optimization.py`

- Optimizes SOS officer break schedule selection
- Uses beam search to find best schedule combination
- Minimizes manning fluctuation (penalty)
- Maximizes coverage (reward)

**Key Classes**:

- `ScheduleOptimizer`: Beam search coordinator
- `SegmentTree`: Scoring helper (penalty/reward computation)

**Optimization Metrics**:

- **Penalty**: Number of manning level changes (lower is better)
- **Reward**: Sum of gaps from maximum manning (lower is better)
- **Combined Score**: α × penalty - β × reward

**Beam Width**: Controls search breadth (20-100 typical range)

- Lower = faster but may miss optimal solution
- Higher = slower but more thorough search

#### `assignment_engine.py`

- Assigns SOS officers to available counters
- Converts officer schedules to counter matrix format
- Adds OT officers to counters specified by users

**Key Classes**:

- `CounterAssignmentEngine`: Main/OT officer assignments
- `SOSAssignmentEngine`: SOS officer interval packing
- `MatrixConverter`: Format conversions

**SOS Assignment Priority**:

1. Pre-assigned counters (if specified)
2. Partial counters with connected intervals
3. Already-used SOS counters with connections
4. Assign new counters (starting with the highest counter number for each zone)

#### `statistics.py`

**Output Format**:

```
ACar

1000: 28/1
8/7/7/6

1030: 30/1
8/8/8/6

[Car counters / Motor counter]
[Zone1 / Zone2 / Zone3 / Zone4]
```

#### `plotter.py`

- Counter timetable: Shows officer at each counter over time
- Officer timetable: Shows counter for each officer over time
- Hover tooltips with time/counter/officer info on plotly
- Graph-paper background to improve chart readability

#### `officer.py`

**Classes**:

- `Officer` (abstract): Base class with schedule array
- `MainOfficer`: Fixed template + adjustments (RA/RO/last counter)
- `SOSOfficer`: Flexible availability + break schedules
- `OTOfficer`: Overtime officer (first 2 slots at specific counter)

**Key Operations**:

```python
officer.assign_counter(slot, counter)
officer.apply_late_arrival(slot)
officer.apply_early_departure(slot)
officer.get_working_intervals()  # SOS only
```

#### `counter.py`

- Represents a single counter across 48 time slots
- Queries for emptiness, fullness, connectivity (i.e. running counters)

**Classes**:

- `Counter`: Single counter with 48 slots
- `CounterMatrix`: Manages all counters collectively

**Key Operations**:

```python
counter.assign_officer(officer_key, start_slot, end_slot)
counter.is_empty(start_slot, end_slot)
counter.is_connected(start_slot, end_slot)

matrix.get_partial_empty_counters()
matrix.get_counters_with_prefix("S")  # Find SOS officers
matrix.merge_with(other_matrix, priority="other")
```

#### `config.py`

- Operation mode definitions (Arrival/Departure)
- Roster templates for 40 main officers
- Zone definitions and counter priority lists

**Key Constants**:

```python
NUM_SLOTS = 48          # 15-minute intervals
START_HOUR = 10         # 10:00 AM start
OperationMode.ARRIVAL   # 41 counters
OperationMode.DEPARTURE # 37 counters
```

**Mode Configuration**:

- Number of counters
- Zone boundaries (counter ranges)
- Roster templates (predefined patterns)
- Counter priority lists (for assignment)

#### `time_utils.py`

- Strips punctuation from input ("14:30" → "1430")
- Validates hour/minute ranges
- Generates time slot lists dynamically

**Key Functions**:

```python
hhmm_to_slot("1430") → 18      # Converts time to slot index
slot_to_hhmm(18) → "1430"      # Converts slot to time string
generate_time_slots() → ["1000", "1015", ...]
```

#### `database.py` & `db_handlers.py`

- Stores roster generation history
- Tracks edits
- Saves last user inputs for convenience

**Tables**:

- `roster_history`: Each roster generation with inputs/results
- `last_inputs`: Most recent user inputs (single row)
- `roster_edits`: Individual edit operations (swap/delete/add)

**Key Functions**:

```python
save_last_inputs(inputs_dict)
get_last_inputs() → dict
save_roster_history(inputs, results) → history_id
save_roster_edit(edit_type, officer_id, ...)
get_roster_edits(limit=20) → list
```

## 💡 Usage Examples

### Step 1: Generate Basic Roster

```python
from acroster.orchestrator import RosterAlgorithmOrchestrator

from acroster.config import OperationMode

orchestrator = RosterAlgorithmOrchestrator(mode=OperationMode.ARRIVAL)

results = orchestrator.run(
    main_officers_reported="1-18",
    report_gl_counters="4AC1, 8AC11, 12AC21, 16AC31",
    sos_timings="",
    ro_ra_officers="",
    handwritten_counters="",
    ot_counters="2,20,40"
)

orchestrator.print_summary()
```

**Output**:

```
======================================================================
SCHEDULING SUMMARY
======================================================================

Officers Scheduled:
  Main Officers (M):    18
  SOS Officers (S):      0
  OT Officers (OT):      3
  ─────────────────────────
  Total:                21

Counter Matrix Shape: (41, 48)
  Counters: 41
  Time Slots: 48
```

### Step 2: Add Late Arrival/Early Departure

```python
results = orchestrator.run(
    main_officers_reported="1-20",
    report_gl_counters="4AC1, 8AC11, 12AC21, 16AC31",
    sos_timings="",
    ro_ra_officers="3RO2100, 11RO1700, 15RA1030",  # Officer 3 leaves at 21:00, etc.
    handwritten_counters="3AC12",
    ot_counters="2,20,40"
)

```

### Step 3: Add SOS Officers

```python
results = orchestrator.run(
    main_officers_reported="1-18",
    report_gl_counters="4AC1, 8AC11, 12AC21, 16AC31",
    sos_timings="(AC22)1000-1300;2000-2200, 1315-1430;2030-2200, 1200-1800",
    ro_ra_officers="3RO2100",
    handwritten_counters="",
    ot_counters="2,20,40"
)

# Access results
main_matrix, final_matrix, officer_schedule, stats = results
```

### Step 4: Export Results

```python
# Export to dictionary
export_data = orchestrator.export_schedules_to_dict()

# Save to JSON
import json
with open('roster_output.json', 'w') as f:
    json.dump(export_data, f, indent=2)

# Access specific data
print(f"Mode: {export_data['mode']}")
print(f"Total officers: {export_data['officer_counts']['total']}")
print(f"Optimization penalty: {export_data['optimization_penalty']}")
```

### Alternatively, use the Orchestrator directly

```python
from acroster.algorithm_orchestrator import RosterAlgorithmOrchestrator
from acroster.config import OperationMode

# Lower-level API (stateless)
orchestrator = RosterAlgorithmOrchestrator(mode=OperationMode.ARRIVAL)

results = orchestrator.run(
    main_officers_reported="1-18",
    report_gl_counters="4AC1, 8AC11",
    sos_timings="1000-1300, 2000-2200",
    ro_ra_officers="3RO2100",
    handwritten_counters="",
    ot_counters="2,20,40"
)

# Unpack results
main_matrix, final_matrix, officer_schedule, statistics = results
```

## ⚙️ Configuration

### Operation Modes

**Arrival Mode**:

- 41 counters total (40 car + 1 motor)
- Zones: 1-10, 11-20, 21-30, 31-40, 41
- Counter priority: [41, 40, 30, 20, 39, 29, 19, ...]

**Departure Mode**:

- 37 counters total (36 car + 1 motor)
- Zones: 1-8, 9-18, 19-28, 29-36, 37
- Counter priority: [37, 36, 35, ...]

### Adjusting Optimization Parameters

```python
orchestrator = RosterAlgorithmOrchestrator(mode=OperationMode.ARRIVAL)

# In algorithm_orchestrator.py, modify:
self.optimizer = ScheduleOptimizer(
    beam_width=20,    # Increase for better results (slower)
    alpha=0.1,        # Weight for smoothness penalty
    beta=1.0          # Weight for coverage reward
)
```

### Break Schedule Constraints

Modify `sos_scheduler.py`:

```python
class BreakScheduleGenerator:
    MAX_CONSECUTIVE_SLOTS = 10  # Change max working slots without break

    # Modify break patterns in _place_breaks()
    if stretch_len >= 36:
        pattern = [2, 3, 3]  # [break1_length, break2_length, break3_length]
    elif stretch_len >= 20:
        pattern = [2, 3]
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Commit: `git commit -m "Add feature X"`
5. Push and create a pull request

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Optimization takes too long"

- **Solution**: Reduce beam width in Advanced Options (try 10-20)

### Debug Mode

Enable debug information in the Streamlit app:

1. Expand "Advanced Options"
2. Check "Show Debug Information"
3. View raw matrices, officer details, and export data structures

## 🔄 Version History

### v2.0 (Current)

- ✨ OOP refactor with cleaner architecture
- ✨ Enhanced error handling and validation
- ✨ Improved edit history tracking

### v1.2

- Improved SOS officer break optimization with beam search
- Added database persistence

### v1.1

- Added Streamlit web interface
- Added visualization with plotly

### v1.0

- Initial command-line roster generator
- Basic SOS officer scheduling on top of main officers' timetable

---

**Built with ❤️ for efficient workforce management**
