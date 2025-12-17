"""
Time conversion utilities for roster scheduling system.

Handles conversion between time slots and time strings (HHMM format).
"""

from datetime import datetime, timedelta
import string
import re
from typing import List

from acroster.config import NUM_SLOTS, START_HOUR


class TimeConverter:
    """Handles time conversions between slots and HHMM format."""

    def __init__(self, start_hour: int = START_HOUR, num_slots: int = NUM_SLOTS):
        """
        Initialize time converter.

        Args:
            start_hour: Starting hour of the day (default: 10)
            num_slots: Total number of 15-minute slots (default: 48)
        """
        self.start_hour = start_hour
        self.num_slots = num_slots

    def hhmm_to_slot(self, hhmm: str) -> int:
        """
        Convert hhmm string to a slot index (0–47).

        Args:
            hhmm: Time string in HHMM format (e.g., '0800', '1430')

        Returns:
            Slot index (0-47)

        Raises:
            ValueError: If hhmm is empty or invalid format
        """
        hhmm = hhmm.strip()

        # Remove any punctuation characters
        hhmm = hhmm.translate(str.maketrans('', '', string.punctuation))

        if not hhmm:
            raise ValueError("Time string cannot be empty")

        if not hhmm.isdigit():
            raise ValueError(f"Time string '{hhmm}' must contain only digits")

        t = int(hhmm)
        h = t // 100
        m = t % 100

        # Validate hours and minutes
        if h < 0 or h > 23:
            raise ValueError(f"Invalid hour: {h} in time '{hhmm}'")
        if m < 0 or m > 59:
            raise ValueError(f"Invalid minute: {m} in time '{hhmm}'")

        slot = (h - self.start_hour) * 4 + (m // 15)
        return max(0, min(self.num_slots - 1, slot))

    def slot_to_hhmm(self, slot: int) -> str:
        """
        Convert slot index back to hhmm string.

        Args:
            slot: Slot index (0-47)

        Returns:
            Time string in HHMM format
        """
        h = self.start_hour + slot // 4
        m = (slot % 4) * 15
        return f"{h:02d}{m:02d}"

    def generate_time_slots(self) -> List[str]:
        """
        Generate a list of HHMM time slots dynamically.

        Returns:
            List of time strings in HHMM format
        """
        start_time = datetime.strptime(f"{self.start_hour:02d}:00", "%H:%M")
        delta = timedelta(minutes=15)

        times = []
        current = start_time
        for _ in range(self.num_slots):
            times.append(current.strftime("%H%M"))
            current += delta

        return times

def get_end_time_slots(time_slots):
    end_time_slots = []

    for t in time_slots:
        slot = hhmm_to_slot(t)
        end_slot = slot + 1

        if end_slot <= NUM_SLOTS:
            end_time_slots.append(slot_to_hhmm(end_slot))
    return (end_time_slots)



# Global instance for backward compatibility
_default_converter = TimeConverter()


def hhmm_to_slot(hhmm: str) -> int:
    """Convert hhmm string to slot index (backward compatibility)."""
    return _default_converter.hhmm_to_slot(hhmm)


def slot_to_hhmm(slot: int) -> str:
    """Convert slot index to hhmm string (backward compatibility)."""
    return _default_converter.slot_to_hhmm(slot)


def generate_time_slots(start_hour: int = START_HOUR, num_slots: int = NUM_SLOTS) -> List[str]:
    """Generate list of time slots (backward compatibility)."""
    converter = TimeConverter(start_hour, num_slots)
    return converter.generate_time_slots()

def get_slot_end_time(slot_idx: int) -> str:
    """Get the end time of a slot (start time + 15 minutes)"""
    return slot_to_hhmm(slot_idx + 1)
# === Raw Text Extraction Functions ===
def clean_time(t):
    """Cleans timing text by removing 'ish' and spaces."""
    t = t.lower().replace("ish", "")
    t = t.replace(" ", "")
    return t if re.match(r'\d{4}-\d{4}', t) else None

def extract_officer_timings(full_text):
    """Extract officer timings from raw text format."""
    blocks = re.split(r'\n(?=\d{2}\s*x\s*)', full_text.strip(), flags=re.IGNORECASE)
    final_records = []

    for block in blocks:
        if not block.strip():
            continue

        base_parentheses = re.search(r'\(([^)]*?\d{4}.*?\d{4}[^)]*?)\)', block)
        if not base_parentheses:
            continue

        base_text = base_parentheses.group(1)
        raw_base_times = re.split(r'[/,&]', base_text)
        base_times = []
        for t in raw_base_times:
            cleaned = clean_time(t)
            if cleaned:
                base_times.append(cleaned)

        officer_lines = re.findall(r'(?:[-*]\s*)?([A-Za-z0-9@_ ]+(?:\([^)]*\))?)', block)
        officer_lines = [l.strip() for l in officer_lines if l.strip() and not re.match(r'\d{2}\s*x', l)]

        for line in officer_lines:
            name = re.sub(r'\(.*?\)', '', line).strip()

            extra_match = re.search(r'\(([^)]*?)\)', line)
            if extra_match:
                extra_raw = extra_match.group(1)
                extra_clean = clean_time(extra_raw)
            else:
                extra_clean = None

            combined_times = base_times.copy()
            if extra_clean:
                combined_times.append(extra_clean)

            timing_str = ";".join(combined_times)
            final_records.append({
                "name": name,
                "timing": timing_str
            })

    return final_records

def parse_sos_timing_with_metadata(timing_str: str) -> tuple:
    """
    Parse SOS timing string with optional deployment and name metadata.
    
    Format: (DEPLOYMENT|NAME)timing or just timing
    
    Examples:
        "(GC|John Doe)1000-1200" -> ("GC", "John Doe", "1000-1200")
        "(|Jane Smith)1000-1200" -> (None, "Jane Smith", "1000-1200")
        "(GC|)1000-1200" -> ("GC", None, "1000-1200")
        "1000-1200" -> (None, None, "1000-1200")
    
    Returns:
        tuple: (deployment, name, timing)
    """
    import re
    
    # Check if starts with parentheses
    match = re.match(r'^\(([^|]*)\|([^)]*)\)(.+)$', timing_str.strip())
    
    if match:
        deployment = match.group(1).strip() or None
        name = match.group(2).strip() or None
        timing = match.group(3).strip()
        return (deployment, name, timing)
    else:
        # No metadata, just timing
        return (None, None, timing_str.strip())