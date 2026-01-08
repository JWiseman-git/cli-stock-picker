"""
LangGraph workflow definition for Stock Intelligence system.

Constructs the StateGraph with nodes, edges, and checkpointing.
"""

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from src.state import AgentState
from src.agents import (
    supervisor_node,
    researcher_node,
    analyst_node,
    human_review_node
)
from src.config import Config
import logging
import os

logger = logging.getLogger(__name__)


def create_graph():
    """
    Create and compile the LangGraph StateGraph for stock analysis.

    Graph Structure:
        START -> supervisor -> researcher -> supervisor -> analyst
        -> supervisor -> human_review [interrupt] -> supervisor -> END

    Returns:
        Compiled graph with SqliteSaver checkpointer
    """
    logger.info("Creating StateGraph")

    # Initialize StateGraph with Pydantic state
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("human_review", human_review_node)

    # Add edges
    # START always goes to supervisor (entry point)
    workflow.add_edge(START, "supervisor")

    # All agents return to supervisor via Command pattern
    # (Conditional routing handled by Command.goto in node functions)

    # Note: With Command pattern, edges are implicit in the Command returns
    # The supervisor's Command.goto determines the next node

    # Setup checkpointer for persistence
    db_path = Config.SQLITE_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    checkpointer = SqliteSaver.from_conn_string(db_path)
    logger.info(f"Checkpointer initialized at {db_path}")

    # Compile graph
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("Graph compiled successfully")
    return graph


def create_thread_config(thread_id: str) -> dict:
    """
    Create configuration for a conversation thread.

    Args:
        thread_id: Unique identifier for the conversation thread

    Returns:
        Configuration dict for graph invocation

    Example:
        >>> config = create_thread_config("user-123-session-1")
        >>> graph.invoke(input_state, config)
    """
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }
