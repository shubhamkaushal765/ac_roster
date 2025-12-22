#!/usr/bin/env python3
"""
Simple test script to verify API functionality
"""
import json

import requests

BASE_URL = "http://localhost:8000"
API_V1 = f"{BASE_URL}/api/v1"


def test_health_check():
    """Test health endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_roster_generation():
    """Test roster generation"""
    print("Testing roster generation...")

    payload = {
        "mode":                   "arrival",
        "main_officers_reported": "1-18",
        "report_gl_counters":     "4AC1, 8AC11, 12AC21, 16AC31",
        "handwritten_counters":   "3AC12,5AC13",
        "ot_counters":            "2,20,40",
        "ro_ra_officers":         "3RO2100, 11RO1700,15RO2130",
        "sos_timings":            "(AC22)1000-1300, 2000-2200, 1315-1430;2030-2200",
        "beam_width":             20,
        "save_to_history":        True
    }

    response = requests.post(
        f"{API_V1}/roster/generate",
        json=payload
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Success: True")
        print(f"Officer counts: {data.get('officer_counts')}")
        print(f"Optimization penalty: {data.get('optimization_penalty')}")
        print("Statistics preview:")
        print(data.get('statistics', {}).get('stats1', '')[:200])
    else:
        print(f"Error: {json.dumps(response.json(), indent=2)}")
    print()


def test_get_last_inputs():
    """Test get last inputs"""
    print("Testing get last inputs...")
    response = requests.get(f"{API_V1}/history/last-inputs")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)[:300]}...")
    else:
        print(f"Error: {response.json()}")
    print()


def test_get_history():
    """Test get roster history"""
    print("Testing get roster history...")
    response = requests.get(f"{API_V1}/history/history?limit=5")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Count: {data.get('count')}")
        print("Success: True")
    else:
        print(f"Error: {response.json()}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Officer Roster Optimization API - Test Suite")
    print("=" * 60)
    print()

    try:
        test_health_check()
        test_roster_generation()
        test_get_last_inputs()
        test_get_history()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Make sure the server is running.")
        print("Run: uvicorn app.main:app --reload")
    except Exception as e:
        print(f"ERROR: {str(e)}")
