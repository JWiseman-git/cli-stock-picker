#!/usr/bin/env python3
"""
Stock Intelligence Multi-Agent System - CLI Entry Point

Professional CLI interface using Rich library for beautiful terminal output.
"""

import sys
import uuid
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_intelligence.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Rich console
console = Console()


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


def stream_graph_execution(graph, input_state: dict, config: dict) -> Optional[dict]:
    """
    Execute graph with streaming output display.

    Args:
        graph: Compiled LangGraph
        input_state: Initial state
        config: Thread configuration

    Returns:
        Final state after execution (or None if interrupted)
    """
    try:
        final_state = None

        for event in graph.stream(input_state, config, stream_mode="updates"):
            if "__interrupt__" in event:
                # Interrupt encountered - return control to caller
                return event["__interrupt__"]

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


def handle_interrupt(graph, interrupt_data: dict, config: dict) -> Optional[dict]:
    """
    Handle human-in-the-loop interrupt.

    Args:
        graph: Compiled LangGraph
        interrupt_data: Data from interrupt event
        config: Thread configuration

    Returns:
        Final state after resume
    """
    # Display interrupt prompt
    prompt_text = interrupt_data.get("value", "Awaiting user input...")

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


def run_analysis(ticker: str):
    """
    Run complete stock analysis workflow.

    Args:
        ticker: Stock ticker symbol
    """
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

        # Initial input
        input_state = {
            "messages": [HumanMessage(content=f"Analyze stock {ticker}")],
            "ticker": ticker
        }

        # Execute graph with streaming
        result = stream_graph_execution(graph, input_state, config)

        # Handle interrupt if present
        if result and isinstance(result, dict) and "__interrupt__" in str(result):
            result = handle_interrupt(graph, result, config)

        # Display final result
        if result:
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

    # Get ticker from user
    console.print("\n[bold]Enter a stock ticker symbol to analyze[/bold]")
    console.print("[dim]Examples: AAPL, GOOGL, MSFT, TSLA[/dim]\n")

    ticker = Prompt.ask("[cyan]Ticker[/cyan]").strip().upper()

    if not ticker:
        console.print("[red]Error: Ticker cannot be empty[/red]")
        sys.exit(1)

    run_analysis(ticker)

    # Ask to analyze another
    console.print()
    if Confirm.ask("[cyan]Analyze another stock?[/cyan]"):
        main()
    else:
        console.print("\n[bold green]Thank you for using Stock Intelligence![/bold green]\n")


if __name__ == "__main__":
    main()
