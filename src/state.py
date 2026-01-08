"""
State management for the Stock Intelligence Multi-Agent System.

This module defines the AgentState schema using Pydantic BaseModel
with annotated reducers for message handling.
"""

from typing import Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(BaseModel):
    """
    Central state container for the multi-agent stock analysis workflow.

    Attributes:
        messages: Conversation history with append-only behavior via add_messages reducer
        ticker: Stock ticker symbol being analyzed (e.g., "AAPL")
        research_data: Raw data collected by researcher agent (JSON serializable dict)
        analysis_summary: Final investment recommendation from analyst agent
        user_decision: Human approval/rejection from interrupt resume
    """

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Message history with automatic appending"
    )

    ticker: Optional[str] = Field(
        default=None,
        description="Stock ticker symbol (e.g., AAPL, GOOGL)"
    )

    research_data: Optional[dict] = Field(
        default=None,
        description="Structured data from yfinance: price, fundamentals, news"
    )

    analysis_summary: Optional[str] = Field(
        default=None,
        description="Analyst agent's investment recommendation"
    )

    user_decision: Optional[str] = Field(
        default=None,
        description="Human approval/rejection decision"
    )

    class Config:
        arbitrary_types_allowed = True  # Required for BaseMessage types
