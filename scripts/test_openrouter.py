#!/usr/bin/env python3
"""
Simple test script for OpenRouter AI API.

This script tests basic functionality of the OpenRouter client.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotive_currents.ai_client import OpenRouterClient, create_client


def test_text_chat():
    """Test basic text chat functionality."""
    print("=== Testing Text Chat ===")
    
    try:
        # Create client (will use environment variable)
        client = create_client()
        
        # Test simple chat
        prompt = "Hello! Can you tell me a short joke?"
        response = client.simple_chat(prompt, model="openai/gpt-4o")
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("✅ Text chat test passed!")
        
    except Exception as e:
        print(f"❌ Text chat test failed: {e}")
        return False
    
    return True


def test_multimodal_chat():
    """Test multimodal chat functionality."""
    print("\n=== Testing Multimodal Chat ===")
    
    try:
        client = create_client()
        
        # Test with the same image from your original code
        text_prompt = "What is in this image?"
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        
        response = client.multimodal_chat(text_prompt, image_url, model="openai/gpt-4o")
        
        print(f"Prompt: {text_prompt}")
        print(f"Image URL: {image_url}")
        print(f"Response: {response}")
        print("✅ Multimodal chat test passed!")
        
    except Exception as e:
        print(f"❌ Multimodal chat test failed: {e}")
        return False
    
    return True


def test_available_models():
    """Test fetching available models."""
    print("\n=== Testing Available Models ===")
    
    try:
        client = create_client()
        models = client.get_available_models()
        
        print(f"Found {len(models.get('data', []))} available models")
        
        # Show first few models
        for i, model in enumerate(models.get('data', [])[:5]):
            print(f"  {i+1}. {model.get('id', 'Unknown')}")
        
        print("✅ Available models test passed!")
        
    except Exception as e:
        print(f"❌ Available models test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("OpenRouter AI API Test")
    print("=" * 50)
    
    # Check if API key is set
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY environment variable not set!")
        print("Please set it with: export OPENROUTER_API_KEY='your-api-key'")
        return
    
    print("✅ API key found")
    
    # Run tests
    tests = [
        test_text_chat,
        test_multimodal_chat,
        test_available_models
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your OpenRouter setup is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main() 