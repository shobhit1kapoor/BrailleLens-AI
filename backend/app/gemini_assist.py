from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_MODELS = "gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite"


def _configured_models() -> list[str]:
    raw = os.getenv("GEMINI_MODELS") or os.getenv("GEMINI_MODEL") or DEFAULT_MODELS
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return models or DEFAULT_MODELS.split(",")


def gemini_is_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts if part.get("text")).strip()


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "text": cleaned,
            "confidence": 0.45,
            "warnings": ["Gemini returned non-JSON text."],
            "cells": [],
        }


def _call_gemini_model(api_key: str, model: str, body: dict[str, Any]) -> tuple[bool, dict[str, Any] | str]:
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=35,
    )
    if response.status_code >= 400:
        return False, f"{model}: {response.status_code} {response.text[:180]}"
    return True, response.json()


def scan_with_gemini(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "available": False,
            "text": "",
            "confidence": 0,
            "warnings": ["Gemini API key is not configured. Add GEMINI_API_KEY to backend/.env."],
            "cells": [],
        }

    prompt = """
You are helping an assistive technology prototype read physical Braille from a camera image.
Inspect the physical raised or printed Braille dots. Do not translate Unicode Braille; use only the visible dots in the image.

Return strict JSON only:
{
  "text": "recognized English text",
  "confidence": 0.0,
  "warnings": ["short warning if uncertain"],
  "cells": [
    {"index": 0, "pattern": "1", "char": "a", "confidence": 0.8}
  ],
  "notes": "brief explanation of visible Braille quality"
}

If the image is not readable, return text as "" and explain why in warnings.
Prefer Grade 1 English Braille. If it appears contracted Grade 2, mention that in warnings.
"""
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    errors: list[str] = []
    selected_model = ""
    payload: dict[str, Any] | None = None
    for model in _configured_models():
        ok, model_response = _call_gemini_model(api_key, model, body)
        if ok:
            selected_model = model
            payload = model_response if isinstance(model_response, dict) else None
            break
        errors.append(str(model_response))

    if payload is None:
        return {
            "available": True,
            "text": "",
            "confidence": 0,
            "warnings": [f"Gemini request failed: {' | '.join(errors)}"],
            "cells": [],
        }
    parsed_text = _extract_text(payload)
    parsed = _parse_json_text(parsed_text)
    parsed["available"] = True
    parsed["model"] = selected_model
    parsed.setdefault("warnings", [])
    parsed.setdefault("cells", [])
    parsed.setdefault("confidence", 0.5)
    parsed.setdefault("text", "")
    return parsed
