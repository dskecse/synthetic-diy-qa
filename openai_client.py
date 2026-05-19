"""
OpenAI Client Utility
Centralized configuration for OpenAI client with OpenAI-compatible API
"""

import os
from openai import OpenAI

# Initialize OpenAI client with OpenAI-compatible API configuration
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL")
)

def get_openai_client():
    """
    Get the configured OpenAI client instance

    Returns:
        OpenAI: Configured OpenAI client with OpenAI-compatible API settings
    """
    return client
