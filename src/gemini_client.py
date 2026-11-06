"""
Centralized Google Gemini Foundation Model Configuration and Client Module.
Uses the official Google GenAI Python SDK (`google-genai`).

Provides:
- Secure environment key resolution via python-dotenv, os.getenv, and Streamlit secrets.
- Full API key masking and error message sanitization (preventing secret leakage).
- Client initialization, live connection validation, and model cascading.
- Production text generation with model fallbacks (e.g. 503 high-demand resilience).
- Document embedding generation (gemini-embedding-001).
- Clear, descriptive status messages for missing, invalid, or active keys.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError, ServerError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None
    APIError = Exception
    ClientError = Exception
    ServerError = Exception

# Model cascades: verified active models with official google-genai SDK
DEFAULT_TEXT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

DEFAULT_PRO_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
]

DEFAULT_EMBEDDING_MODELS = [
    "gemini-embedding-001",
    "gemini-embedding-2",
]


def get_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve Google Gemini API key securely in priority order:
    1. Explicit override_key argument (from UI input widget if provided)
    2. Streamlit Cloud Secrets (st.secrets["GEMINI_API_KEY"] or st.secrets["GOOGLE_API_KEY"])
    3. os.environ["GEMINI_API_KEY"] (loaded via .env)
    4. os.environ["GOOGLE_API_KEY"] (backwards compatibility)
    """
    if override_key and isinstance(override_key, str) and override_key.strip():
        return override_key.strip()

    # Check Streamlit secrets if running in Streamlit runtime
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
                if secret_key:
                    return secret_key
            if "GOOGLE_API_KEY" in st.secrets:
                secret_key = str(st.secrets["GOOGLE_API_KEY"]).strip()
                if secret_key:
                    return secret_key
    except Exception:
        pass

    # Check environment variables
    key = os.getenv("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    legacy_key = os.getenv("GOOGLE_API_KEY")
    if legacy_key and legacy_key.strip():
        return legacy_key.strip()

    return None


def mask_api_key(key: Optional[str]) -> str:
    """
    Mask API key for safe display in UI or logs without exposing sensitive secrets.
    Example: 'AQ.Ab8...yuIw' or '••••••••'
    """
    if not key or not str(key).strip():
        return "Not Configured"
    clean = str(key).strip()
    if len(clean) >= 12:
        return f"{clean[:6]}...{clean[-4:]}"
    return "••••••••"


def sanitize_message(message: str, raw_key: Optional[str] = None) -> str:
    """
    Scrub raw API keys or sensitive credential patterns from error messages.
    """
    if not message:
        return ""
    sanitized = str(message)
    if raw_key and raw_key in sanitized:
        sanitized = sanitized.replace(raw_key, "[REDACTED_API_KEY]")
    # Scrub common Gemini API key prefixes (AIzaSy... or AQ....)
    sanitized = re.sub(r"AIzaSy[A-Za-z0-9_-]{33}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"AQ\.[A-Za-z0-9_-]{30,60}", "[REDACTED_API_KEY]", sanitized)
    return sanitized


def get_gemini_client(api_key: Optional[str] = None) -> Optional[Any]:
    """
    Initialize and return an official google-genai Client.
    Returns None if SDK is not installed or API key is not configured.
    """
    if not GENAI_AVAILABLE:
        return None

    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return None

    try:
        return genai.Client(api_key=resolved_key)
    except Exception:
        return None


def validate_gemini_connection(
    api_key: Optional[str] = None,
    test_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Verify live API connection with the official Google GenAI client.
    Tests model response without exposing secrets.
    """
    if not GENAI_AVAILABLE:
        return {
            "is_valid": False,
            "status": "sdk_missing",
            "masked_key": "N/A",
            "message": "Official 'google-genai' SDK is not installed in the environment.",
            "sample_response": None,
            "error_details": None,
        }

    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return {
            "is_valid": False,
            "status": "missing_key",
            "masked_key": "Not Configured",
            "message": "Gemini API key is not configured. Please add GEMINI_API_KEY to your .env file or sidebar settings.",
            "sample_response": None,
            "error_details": None,
        }

    masked = mask_api_key(resolved_key)

    try:
        client = genai.Client(api_key=resolved_key)
    except Exception as exc:
        clean_err = sanitize_message(str(exc), resolved_key)
        return {
            "is_valid": False,
            "status": "client_init_failed",
            "masked_key": masked,
            "message": f"Failed to initialize Google GenAI client: {clean_err}",
            "sample_response": None,
            "error_details": clean_err,
        }

    # Model candidate order
    candidates = []
    if test_model:
        candidates.append(test_model)
    for m in DEFAULT_TEXT_MODELS:
        if m not in candidates:
            candidates.append(m)

    last_error = ""
    for model_name in candidates:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents="Ping: Please respond with the single word 'ACTIVE'.",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10,
                ) if types else None,
            )
            resp_text = resp.text.strip() if resp and resp.text else "ACTIVE"
            return {
                "is_valid": True,
                "status": "connected",
                "model": model_name,
                "masked_key": masked,
                "message": f"Google Gemini API connection verified successfully using model '{model_name}'.",
                "sample_response": resp_text,
                "error_details": None,
            }
        except Exception as exc:
            err_msg = sanitize_message(str(exc), resolved_key)
            last_error = err_msg
            # If 401 / 403 (invalid key / permission denied), stop trying other models
            if "401" in err_msg or "403" in err_msg or "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                return {
                    "is_valid": False,
                    "status": "invalid_key",
                    "masked_key": masked,
                    "message": "Google Gemini API rejected credentials. The provided API key is invalid or unauthorized.",
                    "sample_response": None,
                    "error_details": err_msg,
                }
            # Otherwise (503 demand spike, 404 retired model), continue cascade
            continue

    return {
        "is_valid": False,
        "status": "error",
        "masked_key": masked,
        "message": f"Gemini connection failed after evaluating available models. Last error: {last_error}",
        "sample_response": None,
        "error_details": last_error,
    }


def generate_gemini_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    preferred_model: Optional[str] = None,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    max_output_tokens: Optional[int] = 2048,
) -> Dict[str, Any]:
    """
    Generate text using official Google GenAI SDK with automatic model fallback
    to handle demand spikes (503) or decommissioned models (404).
    """
    if not GENAI_AVAILABLE:
        return {
            "success": False,
            "text": "",
            "model_used": "offline",
            "error": "The official 'google-genai' library is not available.",
        }

    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return {
            "success": False,
            "text": "",
            "model_used": "offline",
            "error": "Google Gemini API key is missing. Set GEMINI_API_KEY in .env or provide it in the UI settings.",
        }

    try:
        client = genai.Client(api_key=resolved_key)
    except Exception as exc:
        return {
            "success": False,
            "text": "",
            "model_used": "offline",
            "error": sanitize_message(f"Client initialization failed: {exc}", resolved_key),
        }

    # Build model cascade list
    candidates: List[str] = []
    if preferred_model:
        clean_pref = preferred_model.replace("models/", "")
        candidates.append(clean_pref)
    for m in DEFAULT_TEXT_MODELS:
        if m not in candidates:
            candidates.append(m)

    last_err = ""
    for model_name in candidates:
        try:
            config_kwargs: Dict[str, Any] = {"temperature": float(temperature)}
            if max_output_tokens:
                config_kwargs["max_output_tokens"] = int(max_output_tokens)
            if system_instruction:
                config_kwargs["system_instruction"] = str(system_instruction)

            gen_config = types.GenerateContentConfig(**config_kwargs) if types else None

            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )

            text_out = resp.text if (resp and resp.text) else ""
            if text_out:
                return {
                    "success": True,
                    "text": text_out,
                    "model_used": model_name,
                    "error": None,
                }
        except Exception as exc:
            err_str = sanitize_message(str(exc), resolved_key)
            last_err = err_str
            # If invalid credentials, don't keep trying models
            if "401" in err_str or "403" in err_str or "API_KEY_INVALID" in err_str:
                break
            continue

    return {
        "success": False,
        "text": "",
        "model_used": "offline",
        "error": f"Gemini generation error: {last_err}",
    }


def generate_gemini_embedding(
    text: str,
    model: str = "gemini-embedding-001",
    api_key: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Generate dense embedding vector using the official Google GenAI SDK.
    Falls back across embedding models if needed.
    """
    if not GENAI_AVAILABLE or not text or not str(text).strip():
        return None

    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return None

    try:
        client = genai.Client(api_key=resolved_key)
    except Exception:
        return None

    emb_models = [model] + [m for m in DEFAULT_EMBEDDING_MODELS if m != model]
    clean_text = str(text).strip()[:10000]

    for emb_m in emb_models:
        try:
            resp = client.models.embed_content(
                model=emb_m,
                contents=clean_text,
            )
            if resp:
                if hasattr(resp, "embedding") and resp.embedding and hasattr(resp.embedding, "values"):
                    return list(resp.embedding.values)
                if hasattr(resp, "embeddings") and resp.embeddings and len(resp.embeddings) > 0:
                    return list(resp.embeddings[0].values)
        except Exception:
            continue

    return None
