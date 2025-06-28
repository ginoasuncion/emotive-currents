"""
OpenRouter AI client for basic API testing.

This module provides a simple interface for testing OpenRouter AI API calls.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


@dataclass
class OpenRouterConfig:
    """Configuration for OpenRouter AI client."""
    api_key: str
    site_url: Optional[str] = None
    site_name: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Simple client for testing OpenRouter AI API."""
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            config: Configuration object. If None, will try to load from environment.
        """
        if config is None:
            config = self._load_config_from_env()
        
        self.config = config
        self.session = requests.Session()
        
        # Set up default headers
        self.session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        })
        
        if config.site_url:
            self.session.headers["HTTP-Referer"] = config.site_url
        if config.site_name:
            self.session.headers["X-Title"] = config.site_name
    
    def _load_config_from_env(self) -> OpenRouterConfig:
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
        
        return OpenRouterConfig(
            api_key=api_key,
            site_url=os.getenv("OPENROUTER_SITE_URL"),
            site_name=os.getenv("OPENROUTER_SITE_NAME"),
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = "openai/gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenRouter.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (e.g., "openai/gpt-4o", "anthropic/claude-3.5-sonnet")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response as dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def simple_chat(
        self,
        prompt: str,
        model: str = "openai/gpt-4o",
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Simple chat interface for text-only prompts.
        
        Args:
            prompt: Text prompt to send
            model: Model to use
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages, model, temperature, **kwargs)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response content: {e}")
            logger.error(f"Response: {response}")
            raise
    
    def multimodal_chat(
        self,
        text_prompt: str,
        image_url: str,
        model: str = "openai/gpt-4o",
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Chat interface for multimodal prompts (text + image).
        
        Args:
            text_prompt: Text prompt
            image_url: URL of the image
            model: Model to use (must support vision)
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]
        
        response = self.chat_completion(messages, model, temperature, **kwargs)
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response content: {e}")
            logger.error(f"Response: {response}")
            raise
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from OpenRouter."""
        try:
            response = self.session.get(f"{self.config.base_url}/models")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch models: {e}")
            raise


# Convenience function
def create_client(api_key: Optional[str] = None) -> OpenRouterClient:
    """Create an OpenRouter client with optional API key."""
    if api_key:
        config = OpenRouterConfig(api_key=api_key)
        return OpenRouterClient(config)
    return OpenRouterClient() 