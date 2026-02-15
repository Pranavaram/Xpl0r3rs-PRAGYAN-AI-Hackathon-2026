"""
Groq LLM Integration for Triage Explanations
"""

import os
from typing import Optional
from groq import Groq


# Initialize Groq client (singleton pattern)
_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """
    Get or create a singleton Groq client instance.
    
    Returns:
        Groq client instance
    """
    global _groq_client
    
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it before using the LLM service."
            )
        _groq_client = Groq(api_key=api_key)
    
    return _groq_client


def call_llm(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Call Groq LLM to generate a natural language explanation.
    
    Args:
        prompt: The prompt string (user message) to send to the LLM
        model: Groq model name (default: llama-3.1-8b-instant)
    
    Returns:
        LLM response text (explanation)
    
    Raises:
        ValueError: If GROQ_API_KEY is not set
        Exception: If LLM call fails
    """
    client = get_groq_client()
    
    system_message = (
        "You are a concise clinical triage assistant. "
        "Explain risk and department decisions in 2–3 short sentences, "
        "using simple, non-technical language."
    )
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=0.3,  # Lower temperature for more consistent, factual responses
            max_tokens=200    # Limit response length for concise explanations
        )
        
        response_text = chat_completion.choices[0].message.content
        return response_text.strip()
    
    except Exception as e:
        # Re-raise with more context
        raise Exception(f"Failed to call Groq LLM: {str(e)}")
