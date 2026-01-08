"""
Stock Intelligence Multi-Agent System

A professional LangGraph-based multi-agent system for stock analysis
with human-in-the-loop capabilities.
"""

__version__ = "1.0.0"
__author__ = "Jordan"

from src.state import AgentState
from src.graph import create_graph

__all__ = ["AgentState", "create_graph"]
