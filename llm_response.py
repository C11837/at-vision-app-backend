"""
Helper functions for calling large language model APIs.

This module exposes a single function, ``get_llm_response``, which sends a
prompt to an LLM endpoint and returns the generated reply.  It reads the
required API key and endpoint URL from environment variables ``API_KEY`` and
``OPENAI_URL``.  If either variable is not set or the HTTP request fails,
the helper returns a message prefixed with ``LLM analysis unavailable`` so
callers can gracefully degrade to fallback behaviour.

The implementation uses the same payload structure as the OpenAI API.  To
adapt it for another provider adjust the ``headers`` and ``data``
construction accordingly.
"""

from __future__ import annotations

import os
import requests

API_KEY = os.environ.get("API_KEY", "")
OPENAI_URL = os.environ.get("OPENAI_URL", "")


def get_llm_response(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.0,
) -> str:
    """Call the configured LLM service and return its response.

    Parameters
    ----------
    system_prompt : str
        The instruction that sets the behaviour of the assistant.
    user_prompt : str
        The user's question or request.
    max_tokens : int, optional
        Maximum number of tokens to generate, by default 600.
    temperature : float, optional
        Sampling temperature for the model, by default 0.0 (deterministic).

    Returns
    -------
    str
        The assistant's reply, or a fallback message if an error occurs.
    """
    # If no endpoint or key is provided return an informative message.
    if not API_KEY or not OPENAI_URL:
        return "LLM analysis unavailable: API key or endpoint not configured"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Use the GPT-4o mini model; adjust as necessary when a different model
        # becomes available or if your organisation uses a custom model ID.
        "model": "gpt-4o-mini-2024-07-18-section-4omini-1",
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = requests.post(OPENAI_URL, headers=headers, json=data)
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        # Network or HTTP error
        return f"LLM analysis unavailable: {str(e)}"
    except (KeyError, IndexError) as e:
        # Unexpected response format
        return "LLM analysis unavailable due to unexpected response format"