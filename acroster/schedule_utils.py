import numpy as np
def get_all_officer_ids(officer_schedule: dict) -> list:
    """Extract all officer IDs from the schedule"""
    return list(officer_schedule.keys())

def schedule_to_matrix(officer_schedule: dict) -> np.ndarray:
    """Convert officer schedule dictionary to numpy matrix"""
    officer_ids = get_all_officer_ids(officer_schedule)
    matrix = np.zeros((len(officer_ids), 48), dtype=int)
    for i, officer_id in enumerate(officer_ids):
        matrix[i, :] = officer_schedule[officer_id]
    return matrix

def matrix_to_schedule(matrix: np.ndarray, officer_ids: list) -> dict:
    """Convert numpy matrix back to officer schedule dictionary"""
    return {officer_ids[i]: matrix[i, :] for i in range(len(officer_ids))}