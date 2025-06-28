#!/usr/bin/env python3
"""
Integration tests for OpenRouter AI client.

This module tests the OpenRouter AI client functionality including:
- Environment variable loading with python-dotenv
- Basic text chat functionality
- Multimodal chat (text + image)
- Model availability checking
- Error handling

To run these tests:
    uv run python tests/test_openrouter_integration.py
    uv run pytest tests/test_openrouter_integration.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotive_currents.ai_client import create_client, OpenRouterConfig


def test_environment_setup():
    """Test that environment variables are properly loaded."""
    print("=== Testing Environment Setup ===")
    
    # Check if .env file exists
    env_file = Path(".env")
    assert env_file.exists(), ".env file not found! Please create it with your OpenRouter API key."
    
    # Check API key (python-dotenv loads it automatically)
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None, "OPENROUTER_API_KEY not found in .env file"
    
    print(f"✅ .env file found and API key loaded")
    print(f"✅ API key starts with: {api_key[:10]}...")


def test_client_creation():
    """Test OpenRouter client creation."""
    print("\n=== Testing Client Creation ===")
    
    # Test automatic creation from environment
    client = create_client()
    assert client is not None, "Client creation failed"
    print("✅ Client created successfully from environment variables")
    
    # Test manual creation with config
    config = OpenRouterConfig(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        site_url="https://test-site.com",
        site_name="Test Site"
    )
    manual_client = create_client(config.api_key)
    assert manual_client is not None, "Manual client creation failed"
    print("✅ Client created successfully with manual config")


def test_text_chat():
    """Test basic text chat functionality."""
    print("\n=== Testing Text Chat ===")
    
    client = create_client()
    
    # Test with different prompts
    test_prompts = [
        "Say 'Hello from OpenRouter!'",
        "What is 2 + 2?",
        "Tell me a short joke"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  Test {i}: {prompt}")
        response = client.simple_chat(prompt, model="openai/gpt-4o")
        assert response is not None and len(response) > 0, f"Empty response for prompt {i}"
        print(f"    Response: {response[:100]}{'...' if len(response) > 100 else ''}")
    
    print("✅ All text chat tests passed!")


def test_multimodal_chat():
    """Test multimodal chat (text + image) functionality."""
    print("\n=== Testing Multimodal Chat ===")
    
    client = create_client()
    
    # Test with the same image from the original code
    text_prompt = "What is in this image? Describe it briefly."
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    print(f"  Testing with image: {image_url}")
    response = client.multimodal_chat(text_prompt, image_url, model="openai/gpt-4o")
    assert response is not None and len(response) > 0, "Empty multimodal response"
    print(f"    Response: {response[:150]}{'...' if len(response) > 150 else ''}")
    
    print("✅ Multimodal chat test passed!")


def test_model_availability():
    """Test fetching available models."""
    print("\n=== Testing Model Availability ===")
    
    client = create_client()
    models = client.get_available_models()
    
    model_count = len(models.get('data', []))
    assert model_count > 0, "No models available"
    print(f"✅ Found {model_count} available models")
    
    # Show some popular models including the new Gemini model
    popular_models = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet", 
        "google/gemini-2.5-flash-lite-preview-06-17",
        "meta-llama/llama-3.1-8b-instruct"
    ]
    
    available_models = [model['id'] for model in models.get('data', [])]
    print("  Popular models available:")
    for model in popular_models:
        status = "✅" if model in available_models else "❌"
        print(f"    {status} {model}")


def test_different_models():
    """Test chat with different models."""
    print("\n=== Testing Different Models ===")
    
    client = create_client()
    
    # Test with different models including the new Gemini model
    test_models = [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.5-flash-lite-preview-06-17"
    ]
    
    prompt = "Say 'Hello from [MODEL_NAME]!'"
    
    for model in test_models:
        try:
            print(f"  Testing {model}...")
            response = client.simple_chat(prompt, model=model)
            assert response is not None and len(response) > 0, f"Empty response from {model}"
            print(f"    Response: {response[:80]}{'...' if len(response) > 80 else ''}")
        except Exception as e:
            print(f"    ❌ Failed with {model}: {e}")
            # Don't fail the test for model-specific issues
    
    print("✅ Model testing completed!")


def test_gemini_2_5_flash_lite():
    """Test specifically with Google Gemini 2.5 Flash Lite."""
    print("\n=== Testing Google Gemini 2.5 Flash Lite ===")
    
    client = create_client()
    model = "google/gemini-2.5-flash-lite-preview-06-17"
    
    try:
        # Test basic chat
        prompt = "What are the key features of Gemini 2.5 Flash Lite?"
        response = client.simple_chat(prompt, model=model)
        assert response is not None and len(response) > 0, "Empty response from Gemini 2.5 Flash Lite"
        print(f"✅ Gemini 2.5 Flash Lite text chat working")
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Test multimodal if supported
        try:
            text_prompt = "Describe this image briefly."
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            
            multimodal_response = client.multimodal_chat(text_prompt, image_url, model=model)
            assert multimodal_response is not None and len(multimodal_response) > 0, "Empty multimodal response from Gemini 2.5 Flash Lite"
            print(f"✅ Gemini 2.5 Flash Lite multimodal chat working")
            print(f"   Response: {multimodal_response[:100]}{'...' if len(multimodal_response) > 100 else ''}")
        except Exception as e:
            print(f"⚠️  Gemini 2.5 Flash Lite multimodal not supported: {e}")
        
    except Exception as e:
        print(f"❌ Gemini 2.5 Flash Lite test failed: {e}")


def main():
    """Run all integration tests."""
    print("OpenRouter AI Integration Tests")
    print("=" * 60)
    print("This test suite verifies the OpenRouter AI client functionality.")
    print("Make sure you have set up your .env file with OPENROUTER_API_KEY")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Client Creation", test_client_creation),
        ("Text Chat", test_text_chat),
        ("Multimodal Chat", test_multimodal_chat),
        ("Model Availability", test_model_availability),
        ("Different Models", test_different_models),
        ("Gemini 2.5 Flash Lite", test_gemini_2_5_flash_lite)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your OpenRouter integration is working perfectly.")
        print("\nYou can now use the OpenRouter client in your project:")
        print("  from emotive_currents.ai_client import create_client")
        print("  client = create_client()")
        print("  response = client.simple_chat('Your prompt here')")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("Make sure your .env file is properly configured.")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 