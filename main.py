#!/usr/bin/env python3
"""
Stock Intelligence Multi-Agent System - CLI Entry Point

Professional CLI interface using Rich library for beautiful terminal output.
"""

import sys
import uuid
import os
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich import box
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import logging

from src.graph import create_graph, create_thread_config
from src.state import AgentState
from src.config import Config

# Enable UTF-8 mode on Windows to handle Unicode characters from LLM responses
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_intelligence.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Rich console - force_terminal=True uses ANSI sequences instead of legacy Windows API
console = Console(force_terminal=True)


def print_banner():
    """Display application banner."""
    banner = """
# Stock Intelligence Multi-Agent System

**Powered by LangGraph | OpenRouter | yfinance**

A professional multi-agent system for intelligent stock analysis with human-in-the-loop decision making.
"""
    console.print(Panel(Markdown(banner), box=box.DOUBLE, border_style="cyan", title="[bold cyan]STOCK INTELLIGENCE[/bold cyan]"))


def print_agent_message(agent: str, message: str):
    """
    Print agent messages with visual separation.

    Args:
        agent: Agent name (supervisor, researcher, analyst)
        message: Message content
    """
    colors = {
        "supervisor": "yellow",
        "researcher": "green",
        "analyst": "blue",
        "system": "magenta"
    }
    color = colors.get(agent.lower(), "white")

    # Agent icons using ASCII-safe symbols
    icons = {
        "supervisor": "[S]",
        "researcher": "[R]",
        "analyst": "[A]",
        "system": "[*]"
    }
    icon = icons.get(agent.lower(), ">")

    console.print(f"\n[bold {color}]{icon} {agent.upper()}[/bold {color}]")
    console.print(Panel(message, border_style=color, box=box.ROUNDED))


def get_analysis_mode() -> tuple:
    """
    Prompt user to choose analysis mode and collect ticker(s).

    Returns:
        Tuple of (mode, ticker_or_none, ticker_a_or_none, ticker_b_or_none)
    """
    console.print("\n[bold]Choose analysis mode:[/bold]")
    console.print("  [cyan]1[/cyan] - Single stock analysis")
    console.print("  [cyan]2[/cyan] - Compare two stocks\n")

    choice = Prompt.ask("[cyan]Selection[/cyan]", choices=["1", "2"], default="1")

    if choice == "1":
        console.print("\n[bold]Enter a stock ticker symbol to analyze[/bold]")
        console.print("[dim]Examples: AAPL, GOOGL, MSFT, TSLA[/dim]\n")

        ticker = Prompt.ask("[cyan]Ticker[/cyan]").strip().upper()

        if not ticker:
            console.print("[red]Error: Ticker cannot be empty[/red]")
            sys.exit(1)

        return ("single", ticker, None, None)

    else:
        console.print("\n[bold]Enter two stock ticker symbols to compare[/bold]")
        console.print("[dim]Examples: AAPL vs GOOGL, MSFT vs AMZN[/dim]\n")

        ticker_a = Prompt.ask("[cyan]First ticker[/cyan]").strip().upper()
        if not ticker_a:
            console.print("[red]Error: First ticker cannot be empty[/red]")
            sys.exit(1)

        ticker_b = Prompt.ask("[cyan]Second ticker[/cyan]").strip().upper()
        if not ticker_b:
            console.print("[red]Error: Second ticker cannot be empty[/red]")
            sys.exit(1)

        if ticker_a == ticker_b:
            console.print("[red]Error: Please enter two different tickers[/red]")
            sys.exit(1)

        return ("comparison", None, ticker_a, ticker_b)


def stream_graph_execution(graph, input_state: dict, config: dict) -> Optional[dict]:
    """
    Execute graph with streaming output display.

    Args:
        graph: Compiled LangGraph
        input_state: Initial state
        config: Thread configuration

    Returns:
        Final state after execution (or tuple with interrupt info if interrupted)
    """
    try:
        final_state = None

        for event in graph.stream(input_state, config, stream_mode="updates"):
            if "__interrupt__" in event:
                # Interrupt encountered - return interrupt tuple
                return ("__interrupt__", event["__interrupt__"])

            # Display agent updates
            for node_name, node_state in event.items():
                if node_name in ["supervisor", "researcher", "analyst"]:
                    messages = node_state.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        print_agent_message(node_name, last_message.content)

                final_state = node_state

        return final_state

    except Exception as e:
        console.print(f"[bold red]Error during execution: {e}[/bold red]")
        logger.error(f"Graph execution error: {e}", exc_info=True)
        return None


def handle_interrupt(graph, interrupt_data, config: dict) -> Optional[dict]:
    """
    Handle human-in-the-loop interrupt.

    Args:
        graph: Compiled LangGraph
        interrupt_data: Data from interrupt event (can be list/tuple of Interrupt objects)
        config: Thread configuration

    Returns:
        Final state after resume
    """
    # Extract prompt text from interrupt data
    # LangGraph returns a list/tuple of Interrupt objects with a 'value' attribute
    if isinstance(interrupt_data, (list, tuple)) and len(interrupt_data) > 0:
        first_interrupt = interrupt_data[0]
        prompt_text = getattr(first_interrupt, 'value', str(first_interrupt))
    elif hasattr(interrupt_data, 'value'):
        prompt_text = interrupt_data.value
    elif isinstance(interrupt_data, dict):
        prompt_text = interrupt_data.get("value", "Awaiting user input...")
    else:
        prompt_text = str(interrupt_data)

    console.print("\n" + "="*80, style="yellow")
    console.print("[bold yellow]>> HUMAN REVIEW REQUIRED[/bold yellow]")
    console.print("="*80 + "\n", style="yellow")

    console.print(Panel(
        prompt_text,
        title="Investment Recommendation",
        border_style="yellow",
        box=box.DOUBLE
    ))

    # Get user decision
    console.print("\n[bold]Options:[/bold]")
    console.print("  • Type [green]APPROVE[/green] to accept the recommendation")
    console.print("  • Type [red]REJECT[/red] to decline the recommendation")
    console.print("  • Or provide custom feedback\n")

    user_input = Prompt.ask("[bold cyan]Your decision[/bold cyan]")

    # Resume execution with user input
    console.print("\n[dim]Resuming workflow with your decision...[/dim]\n")

    try:
        # Resume using Command(resume=...)
        final_state = None
        for event in graph.stream(
            Command(resume=user_input),
            config,
            stream_mode="updates"
        ):
            for node_name, node_state in event.items():
                if node_name in ["supervisor", "human_review"]:
                    messages = node_state.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        print_agent_message(node_name, last_message.content)

                final_state = node_state

        return final_state

    except Exception as e:
        console.print(f"[bold red]Error during resume: {e}[/bold red]")
        logger.error(f"Resume error: {e}", exc_info=True)
        return None


def run_analysis(mode: str, ticker: Optional[str] = None, ticker_a: Optional[str] = None, ticker_b: Optional[str] = None):
    """
    Run complete stock analysis workflow.

    Args:
        mode: Analysis mode ('single' or 'comparison')
        ticker: Stock ticker symbol (for single mode)
        ticker_a: First stock ticker (for comparison mode)
        ticker_b: Second stock ticker (for comparison mode)
    """
    if mode == "comparison":
        console.print(f"\n[bold cyan]Starting comparison: {ticker_a} vs {ticker_b}...[/bold cyan]\n")
    else:
        console.print(f"\n[bold cyan]Starting analysis for {ticker}...[/bold cyan]\n")

    try:
        # Validate configuration
        Config.validate()

        # Create graph
        with console.status("[bold green]Initializing multi-agent system...", spinner="dots"):
            graph = create_graph()

        # Create thread config
        thread_id = f"session-{uuid.uuid4().hex[:8]}"
        config = create_thread_config(thread_id)
        logger.info(f"Session thread: {thread_id}")

        # Build initial input based on mode
        if mode == "comparison":
            input_state = {
                "messages": [HumanMessage(content=f"Compare stocks {ticker_a} vs {ticker_b}")],
                "mode": "comparison",
                "ticker_a": ticker_a,
                "ticker_b": ticker_b
            }
        else:
            input_state = {
                "messages": [HumanMessage(content=f"Analyze stock {ticker}")],
                "mode": "single",
                "ticker": ticker
            }

        # Execute graph with streaming
        result = stream_graph_execution(graph, input_state, config)

        # Handle interrupt if present
        if result and isinstance(result, tuple) and result[0] == "__interrupt__":
            interrupt_data = result[1]
            result = handle_interrupt(graph, interrupt_data, config)

        # Display final result
        if result and isinstance(result, dict):
            console.print("\n" + "="*80, style="green")
            console.print("[bold green][OK] ANALYSIS COMPLETE[/bold green]")
            console.print("="*80 + "\n", style="green")

            if result.get("user_decision"):
                console.print(Panel(
                    f"User Decision: {result['user_decision']}",
                    border_style="green",
                    box=box.DOUBLE
                ))

    except ValueError as e:
        console.print(f"\n[bold red]Configuration Error:[/bold red] {e}")
        console.print("[dim]Please check your .env file and ensure OPENROUTER_API_KEY is set.[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected Error:[/bold red] {e}")
        logger.error(f"Analysis error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    print_banner()

    # Get analysis mode and ticker(s) from user
    mode, ticker, ticker_a, ticker_b = get_analysis_mode()

    # Run analysis based on mode
    run_analysis(mode=mode, ticker=ticker, ticker_a=ticker_a, ticker_b=ticker_b)

    # Ask to analyze another
    console.print()
    if Confirm.ask("[cyan]Run another analysis?[/cyan]"):
        main()
    else:
        console.print("\n[bold green]Thank you for using Stock Intelligence![/bold green]\n")


if __name__ == "__main__":
    main()
