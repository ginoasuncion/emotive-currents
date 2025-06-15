"""Tests for core functionality."""

from emotive_currents.core import analyze_emotion, hello_world


def test_hello_world():
    """Test the hello world function."""
    result = hello_world()
    assert result == "Hello from Emotive Currents!"


def test_analyze_emotion():
    """Test emotion analysis function."""
    result = analyze_emotion("I am happy today!")

    assert "text" in result
    assert "emotions" in result
    assert "confidence" in result
    assert result["text"] == "I am happy today!"
    assert isinstance(result["emotions"], dict)
    assert isinstance(result["confidence"], int | float)


def test_analyze_emotion_empty():
    """Test emotion analysis with empty text."""
    result = analyze_emotion("")

    assert result["text"] == ""
    assert "emotions" in result
