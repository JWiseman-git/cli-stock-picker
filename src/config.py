"""
Configuration management for LLM providers and environment variables.

Handles OpenRouter integration with LangChain's ChatOpenAI interface.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Config:
    """
    Centralized configuration for the Stock Intelligence system.

    Attributes:
        OPENROUTER_API_KEY: API key for OpenRouter
        DEFAULT_MODEL: Free-tier Gemini model via OpenRouter
        OPENROUTER_BASE_URL: API endpoint
        SQLITE_DB_PATH: Checkpointer database location
    """

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv(
        "DEFAULT_MODEL",
        "google/gemini-2.0-flash-lite-preview-02-05:free"
    )
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "./data/checkpoints.db")

    # LLM generation parameters
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    @classmethod
    def validate(cls) -> None:
        """Validate required environment variables are set."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Please set it in your .env file or environment."
            )
        logger.info(f"Configuration validated. Using model: {cls.DEFAULT_MODEL}")


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance configured for OpenRouter.

    Args:
        model: Override default model (optional)
        temperature: Override default temperature (optional)
        max_tokens: Override default max_tokens (optional)

    Returns:
        Configured ChatOpenAI instance

    Example:
        >>> llm = create_llm()
        >>> response = llm.invoke("What is stock analysis?")
    """
    Config.validate()

    return ChatOpenAI(
        model=model or Config.DEFAULT_MODEL,
        api_key=Config.OPENROUTER_API_KEY,
        base_url=Config.OPENROUTER_BASE_URL,
        temperature=temperature or Config.TEMPERATURE,
        max_tokens=max_tokens or Config.MAX_TOKENS,
    )
