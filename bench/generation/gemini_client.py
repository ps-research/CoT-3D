import os
import json
from google import genai
from google.genai import types


def generate(
    prompt: str,
    system_instruction: str,
    response_schema: dict = None,
    model: str = "gemini-3.5-flash",
    thinking_level: str = "LOW",
):
    """
    Call Gemini API with structured JSON output.
    Uses the exact API signature from AI Studio.

    Args:
        prompt: user message
        system_instruction: system prompt
        response_schema: JSON schema for structured output (optional)
        model: model name (DO NOT CHANGE from gemini-3.5-flash)
        thinking_level: LOW/MEDIUM/HIGH

    Returns:
        Parsed JSON dict from the structured response
    """
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    config_kwargs = {
        "thinking_config": types.ThinkingConfig(
            thinking_level=thinking_level,
        ),
        "response_mime_type": "application/json",
        "system_instruction": [
            types.Part.from_text(text=system_instruction),
        ],
    }

    if response_schema:
        config_kwargs["response_schema"] = response_schema

    generate_content_config = types.GenerateContentConfig(**config_kwargs)

    # Collect full response (not streaming, for structured output parsing)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return json.loads(response.text)
