"""
Agent node implementations for the multi-agent stock analysis system.

Each agent is a pure function that receives state, performs actions,
and returns state updates (following LangGraph best practices).
"""

from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from src.state import AgentState
from src.tools import fetch_stock_data, format_research_summary, format_comparison_summary
from src.config import create_llm
import logging

logger = logging.getLogger(__name__)


def supervisor_node(state: AgentState) -> Command[Literal["researcher", "analyst", "human_review", "__end__"]]:
    """
    Supervisor agent that orchestrates the workflow.

    Responsibilities:
    - Routes initial request to researcher
    - Routes research data to analyst
    - Routes final recommendation to human review
    - Ends workflow after human approval/rejection

    Args:
        state: Current AgentState

    Returns:
        Command specifying next agent to invoke
    """
    is_comparison = state.mode == "comparison"

    # Determine if research is complete based on mode
    if is_comparison:
        research_complete = state.research_data_a is not None and state.research_data_b is not None
    else:
        research_complete = state.research_data is not None

    # Initial request - route to researcher
    if not research_complete:
        logger.info("Supervisor: Routing to researcher agent")
        mode_msg = "comparison" if is_comparison else "single stock"
        return Command(
            goto="researcher",
            update={
                "messages": [
                    SystemMessage(content=f"Supervisor: Initiating {mode_msg} research.")
                ]
            }
        )

    # Research complete but no analysis - route to analyst
    if research_complete and not state.analysis_summary:
        logger.info("Supervisor: Routing to analyst agent")
        return Command(
            goto="analyst",
            update={
                "messages": [
                    SystemMessage(content="Supervisor: Research complete, analyzing data.")
                ]
            }
        )

    # Analysis complete but no human decision - route to human review
    if state.analysis_summary and not state.user_decision:
        logger.info("Supervisor: Routing to human review")
        return Command(
            goto="human_review",
            update={
                "messages": [
                    SystemMessage(content="Supervisor: Analysis ready for human review.")
                ]
            }
        )

    # Human decision received - end workflow
    logger.info("Supervisor: Workflow complete")
    return Command(
        goto="__end__",
        update={
            "messages": [
                AIMessage(content=f"Workflow complete. User decision: {state.user_decision}")
            ]
        }
    )


def researcher_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Researcher agent that fetches stock data using yfinance.

    Responsibilities:
    - Extract ticker symbol from user messages
    - Fetch comprehensive stock data
    - Store data in state.research_data (single) or research_data_a/b (comparison)
    - Return to supervisor

    Args:
        state: Current AgentState

    Returns:
        Command routing back to supervisor with research data
    """
    logger.info("Researcher agent: Starting research")

    # Check if we're in comparison mode
    if state.mode == "comparison":
        return _research_comparison(state)
    else:
        return _research_single(state)


def _research_single(state: AgentState) -> Command[Literal["supervisor"]]:
    """Handle single stock research."""
    # Extract ticker from latest user message
    ticker = state.ticker
    if not ticker:
        # Parse from messages if not explicitly set
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                # Simple extraction (could use LLM for more complex parsing)
                words = msg.content.upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        ticker = word
                        break
                if ticker:
                    break

    if not ticker:
        error_msg = "Could not identify stock ticker. Please specify a ticker symbol (e.g., AAPL)."
        logger.error(error_msg)
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=error_msg)],
                "research_data": {"error": error_msg}
            }
        )

    # Fetch data
    try:
        logger.info(f"Fetching data for {ticker}")
        research_data = fetch_stock_data(ticker)
        summary = format_research_summary(research_data)

        return Command(
            goto="supervisor",
            update={
                "ticker": ticker,
                "research_data": research_data,
                "messages": [
                    AIMessage(
                        content=f"Research complete for {ticker}. Key data collected:\n{summary}"
                    )
                ]
            }
        )
    except Exception as e:
        error_msg = f"Research failed: {str(e)}"
        logger.error(error_msg)
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=error_msg)],
                "research_data": {"error": error_msg}
            }
        )


def _research_comparison(state: AgentState) -> Command[Literal["supervisor"]]:
    """Handle comparison mode research for two stocks."""
    ticker_a = state.ticker_a
    ticker_b = state.ticker_b

    if not ticker_a or not ticker_b:
        error_msg = "Comparison mode requires two ticker symbols (ticker_a and ticker_b)."
        logger.error(error_msg)
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=error_msg)],
                "research_data_a": {"error": error_msg},
                "research_data_b": {"error": error_msg}
            }
        )

    try:
        # Fetch data for both stocks sequentially
        logger.info(f"Fetching data for {ticker_a}")
        research_data_a = fetch_stock_data(ticker_a)

        logger.info(f"Fetching data for {ticker_b}")
        research_data_b = fetch_stock_data(ticker_b)

        # Format comparison summary
        comparison_summary = format_comparison_summary(research_data_a, research_data_b)

        return Command(
            goto="supervisor",
            update={
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "research_data_a": research_data_a,
                "research_data_b": research_data_b,
                "messages": [
                    AIMessage(
                        content=f"Research complete for {ticker_a} vs {ticker_b}. Comparison data collected:\n{comparison_summary}"
                    )
                ]
            }
        )
    except Exception as e:
        error_msg = f"Research failed: {str(e)}"
        logger.error(error_msg)
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=error_msg)],
                "research_data_a": {"error": error_msg},
                "research_data_b": {"error": error_msg}
            }
        )


def analyst_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Analyst agent that synthesizes research into investment recommendation.

    Responsibilities:
    - Analyze research data using LLM
    - Generate investment summary (buy/hold/sell recommendation)
    - Consider fundamentals, technicals, and news sentiment
    - Return structured recommendation to supervisor

    Args:
        state: Current AgentState with populated research_data

    Returns:
        Command routing back to supervisor with analysis
    """
    logger.info("Analyst agent: Starting analysis")

    # Check if we're in comparison mode
    if state.mode == "comparison":
        return _analyze_comparison(state)
    else:
        return _analyze_single(state)


def _analyze_single(state: AgentState) -> Command[Literal["supervisor"]]:
    """Handle single stock analysis."""
    if not state.research_data or "error" in state.research_data:
        return Command(
            goto="supervisor",
            update={
                "messages": [
                    AIMessage(content="Cannot analyze: No valid research data available.")
                ]
            }
        )

    # Format research for LLM
    research_summary = format_research_summary(state.research_data)

    # Create analysis prompt
    system_prompt = """You are an expert financial analyst. Analyze the provided stock research data and provide:

1. Investment Recommendation: BUY, HOLD, or SELL
2. Confidence Level: High, Medium, or Low
3. Key Rationale: 3-5 bullet points explaining your recommendation
4. Risk Factors: 2-3 potential concerns
5. Price Target: 6-month price estimate

Be objective, data-driven, and clearly explain your reasoning. Format your response clearly."""

    user_prompt = f"""Analyze this stock research and provide your investment recommendation:

{research_summary}

Provide a comprehensive investment analysis following the structured format."""

    try:
        llm = create_llm(temperature=0.3)  # Lower temperature for analytical tasks

        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages_for_llm)
        analysis = response.content

        logger.info("Analysis complete")
        return Command(
            goto="supervisor",
            update={
                "analysis_summary": analysis,
                "messages": [
                    AIMessage(
                        content=f"Investment Analysis for {state.ticker}:\n\n{analysis}"
                    )
                ]
            }
        )

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=error_msg)]
            }
        )


def _analyze_comparison(state: AgentState) -> Command[Literal["supervisor"]]:
    """Handle comparative analysis of two stocks."""
    # Validate we have data for both stocks
    if not state.research_data_a or "error" in state.research_data_a:
        return Command(
            goto="supervisor",
            update={
                "messages": [
                    AIMessage(content=f"Cannot analyze: No valid research data for {state.ticker_a}.")
                ]
            }
        )

    if not state.research_data_b or "error" in state.research_data_b:
        return Command(
            goto="supervisor",
            update={
                "messages": [
                    AIMessage(content=f"Cannot analyze: No valid research data for {state.ticker_b}.")
                ]
            }
        )

    # Format comparison summary for LLM
    comparison_summary = format_comparison_summary(state.research_data_a, state.research_data_b)

    # Create comparative analysis prompt
    system_prompt = """You are an expert financial analyst specializing in comparative stock analysis. Analyze the two stocks and provide:

1. **Winner Pick**: Which stock is the better investment right now and why (one clear choice)
2. **Head-to-Head Analysis**: Compare the two stocks across key dimensions:
   - Valuation (P/E, PEG, forward P/E)
   - Growth prospects (revenue growth, earnings trends)
   - Financial health (margins, balance sheet strength)
   - Risk profile (beta, volatility)
3. **Key Differentiators**: 3-4 factors that most distinguish these two investments
4. **Risk Factors for Each**: 2-3 specific concerns for each stock
5. **Investment Scenarios**: When would you prefer Stock A over Stock B, and vice versa?

Be objective, data-driven, and clearly explain your reasoning. Make a definitive recommendation."""

    user_prompt = f"""Compare these two stocks and determine which is the better investment:

{comparison_summary}

Provide a comprehensive head-to-head analysis and pick a winner."""

    try:
        llm = create_llm(temperature=0.3)

        messages_for_llm = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages_for_llm)
        analysis = response.content

        logger.info("Comparison analysis complete")
        return Command(
            goto="supervisor",
            update={
                "analysis_summary": analysis,
                "messages": [
                    AIMessage(
                        content=f"Comparative Analysis: {state.ticker_a} vs {state.ticker_b}:\n\n{analysis}"
                    )
                ]
            }
        )

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=error_msg)]
            }
        )


def human_review_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Human-in-the-loop review node using interrupt().

    This node triggers an interrupt to pause execution and collect
    human approval/rejection. Execution resumes via Command(resume=...).

    Args:
        state: Current AgentState with analysis_summary

    Returns:
        Command routing back to supervisor after human input
    """
    from langgraph.types import interrupt

    logger.info("Human review: Triggering interrupt for approval")

    # Create header based on mode
    if state.mode == "comparison":
        header = f"Comparison: {state.ticker_a} vs {state.ticker_b}"
    else:
        header = f"Analysis: {state.ticker}"

    # Present analysis and request approval
    prompt = f"""
### {header}

{state.analysis_summary}

---

Do you approve this investment recommendation?
Please respond with:
- 'APPROVE' to accept the recommendation
- 'REJECT' to decline the recommendation
- Or provide specific feedback
"""

    # interrupt() will pause execution here until Command(resume=...) is invoked
    user_input = interrupt(prompt)

    logger.info(f"Human review received: {user_input}")

    # Process user decision
    decision = user_input.strip().upper()

    if "APPROVE" in decision:
        outcome = "APPROVED"
        message = "Thank you! The investment recommendation has been approved."
    elif "REJECT" in decision:
        outcome = "REJECTED"
        message = "Understood. The investment recommendation has been rejected."
    else:
        outcome = "FEEDBACK"
        message = f"Thank you for your feedback: {user_input}"

    return Command(
        goto="supervisor",
        update={
            "user_decision": outcome,
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content=message)
            ]
        }
    )
