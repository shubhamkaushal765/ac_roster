"""
Time conversion utilities for roster scheduling system.

Handles conversion between time slots and time strings (HHMM format).
"""

from datetime import datetime, timedelta
import string
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