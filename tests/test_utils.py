"""Tests for utility functions."""

from emotive_currents.utils import process_batch, validate_input


def test_validate_input_valid():
    """Test validation with valid input."""
    assert validate_input("valid text") is True
    assert validate_input(123) is True
    assert validate_input([1, 2, 3]) is True


def test_validate_input_invalid():
    """Test validation with invalid input."""
    assert validate_input(None) is False
    assert validate_input("") is False
    assert validate_input("   ") is False


def test_process_batch():
    """Test batch processing."""
    items = ["valid", "", "also valid", None]
    results = process_batch(items)

    assert len(results) == 4
    assert results[0]["processed"] is True
    assert results[1]["processed"] is False
    assert results[2]["processed"] is True
    assert results[3]["processed"] is False
