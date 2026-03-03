"""
LangChain-based LLM + (optional) direct image generation.

- Text generation uses LangChain + ChatGoogleGenerativeAI
- Image generation uses google-generativeai directly (LangChain support varies by model)

Env vars used:
- PROVIDER=gemini
- GEMINI_API_KEY=...
- GEMINI_MODEL=models/gemini-2.5-flash
- GEMINI_IMAGE_MODEL=models/gemini-2.5-flash-image
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------------
# Config loading (local + cloud)
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load .env from the SAME folder as this file (bulletproof)
# (If .env doesn't exist, this is a no-op.)
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _get_secret(key: str) -> Optional[str]:
    """
    Read a config value from:
      1) os.environ (local env, docker env, CI env)
      2) Streamlit secrets (Streamlit Cloud)
    """
    val = os.getenv(key)
    if val:
        return val

    # Streamlit Cloud secrets support (only when running under Streamlit)
    try:
        import streamlit as st  # imported lazily to avoid hard dependency

        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return None


def _require(key: str, hint: str = "") -> str:
    val = _get_secret(key)
    if not val:
        msg = f"{key} is missing."
        if hint:
            msg += f" {hint}"
        raise RuntimeError(msg)
    return val


# -----------------------------
# Gemini (LangChain) text client
# -----------------------------

def _gemini_llm(temperature: float) -> ChatGoogleGenerativeAI:
    api_key = _require("GEMINI_API_KEY", "Set it in .env (local) or Streamlit Secrets (cloud).")
    model_name = _get_secret("GEMINI_MODEL") or "models/gemini-2.5-flash"

    # LangChain wrapper expects google_api_key for auth.
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def chat_completion(system: str, user: str, *, temperature: float = 0.6) -> str:
    """LangChain pipeline: PromptTemplate -> Gemini Chat Model -> String output."""
    llm = _gemini_llm(temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{user}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user": user})


def chat_completion_json(system: str, user: str, *, temperature: float = 0.6) -> Dict[str, Any]:
    """
    Calls LLM (LangChain) and parses JSON with a safe fallback extractor.
    (Still expects the model to return a JSON object.)
    """
    raw = chat_completion(system, user, temperature=temperature)

    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError(f"Model did not return valid JSON. Raw output:\n{raw}")


# -----------------------------
# Optional: Direct image gen
# -----------------------------

def generate_image_bytes(prompt: str) -> bytes:
    """
    Direct Gemini image generation: returns PNG/JPEG bytes you can show in Streamlit.

    Requires:
      - GEMINI_API_KEY
      - GEMINI_IMAGE_MODEL=models/gemini-2.5-flash-image (or any image-capable model available to your key)
    """
    import google.generativeai as genai

    api_key = _require("GEMINI_API_KEY", "Set it in .env (local) or Streamlit Secrets (cloud).")
    genai.configure(api_key=api_key)

    model_name = _get_secret("GEMINI_IMAGE_MODEL") or "models/gemini-2.5-flash-image"
    model = genai.GenerativeModel(model_name)

    resp = model.generate_content(prompt)

    # Extract inline image bytes
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for p in parts:
            inline = getattr(p, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data  # bytes

    raise RuntimeError("No image bytes returned. Try simplifying the prompt or changing GEMINI_IMAGE_MODEL.")