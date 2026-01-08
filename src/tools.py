"""
Financial data tools using yfinance.

Provides structured data fetching for stock analysis including:
- Real-time price data
- Company fundamentals
- Financial statements
- Recent news headlines
"""

import yfinance as yf
from typing import Dict, Any
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _convert_to_native_types(obj):
    """Convert numpy types to native Python types for serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native_types(item) for item in obj]
    return obj


def fetch_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch comprehensive stock data for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")

    Returns:
        Dictionary containing:
            - price_data: Current price, day range, 52-week range
            - fundamentals: P/E ratio, market cap, dividend yield
            - financials: Revenue, earnings, cash flow summaries
            - news: Recent news headlines (last 7 days)
            - historical: 90-day price history for trend analysis

    Raises:
        ValueError: If ticker is invalid or data unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Validate ticker exists
        if not info or "regularMarketPrice" not in info:
            raise ValueError(f"Invalid ticker symbol: {ticker}")

        # Price data
        price_data = {
            "current_price": info.get("regularMarketPrice"),
            "previous_close": info.get("previousClose"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
        }

        # Fundamental data
        fundamentals = {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "eps": info.get("trailingEps"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
        }

        # Company metadata
        company_info = {
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": info.get("longBusinessSummary"),
            "website": info.get("website"),
        }

        # Historical data (90 days for trend analysis)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        history = stock.history(start=start_date, end=end_date)

        historical_data = {
            "90_day_return": (
                (history["Close"].iloc[-1] - history["Close"].iloc[0])
                / history["Close"].iloc[0] * 100
            ) if len(history) > 0 else None,
            "volatility": history["Close"].pct_change().std() * 100 if len(history) > 1 else None,
            "avg_price_90d": history["Close"].mean() if len(history) > 0 else None,
        }

        # Recent news (last 5 articles)
        news_items = []
        try:
            news = stock.news[:5]  # Top 5 recent articles
            for article in news:
                news_items.append({
                    "title": article.get("title"),
                    "publisher": article.get("publisher"),
                    "link": article.get("link"),
                })
        except Exception as e:
            logger.warning(f"Could not fetch news for {ticker}: {e}")
            news_items = []

        # Convert all numpy types to native Python types for serialization
        result = {
            "ticker": ticker.upper(),
            "fetch_timestamp": datetime.now().isoformat(),
            "company_info": company_info,
            "price_data": price_data,
            "fundamentals": fundamentals,
            "historical_data": historical_data,
            "news": news_items,
        }
        return _convert_to_native_types(result)

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")


def format_research_summary(data: Dict[str, Any]) -> str:
    """
    Format research data into human-readable summary for LLM consumption.

    Args:
        data: Output from fetch_stock_data()

    Returns:
        Formatted string summary suitable for LLM analysis
    """
    company = data["company_info"]
    price = data["price_data"]
    fundamentals = data["fundamentals"]
    historical = data["historical_data"]

    # Helper functions to format values safely
    def fmt_currency(val):
        return f"${val:.2f}" if val else "N/A"

    def fmt_pct(val):
        return f"{val:.2f}%" if val else "N/A"

    def fmt_num(val):
        return f"{val:,}" if val else "N/A"

    def fmt_float(val):
        return f"{val:.2f}" if val else "N/A"

    summary = f"""
## Research Summary for {data['ticker']} - {company['name']}

### Company Overview
- Sector: {company['sector']}
- Industry: {company['industry']}
- Description: {company['description'][:200] if company['description'] else 'N/A'}...

### Current Price Data
- Current Price: {fmt_currency(price['current_price'])}
- Day Range: {fmt_currency(price['day_low'])} - {fmt_currency(price['day_high'])}
- 52-Week Range: {fmt_currency(price['52_week_low'])} - {fmt_currency(price['52_week_high'])}
- Volume: {fmt_num(price['volume'])} (Avg: {fmt_num(price['avg_volume'])})

### Fundamental Metrics
- Market Cap: {fmt_num(fundamentals['market_cap'])}
- P/E Ratio: {fmt_float(fundamentals['pe_ratio'])}
- Dividend Yield: {fmt_pct(fundamentals['dividend_yield']*100) if fundamentals['dividend_yield'] else 'N/A'}
- Beta: {fmt_float(fundamentals['beta'])}
- Profit Margin: {fmt_pct(fundamentals['profit_margin']*100) if fundamentals['profit_margin'] else 'N/A'}

### Performance Trends
- 90-Day Return: {fmt_pct(historical['90_day_return'])}
- Volatility: {fmt_pct(historical['volatility'])}

### Recent News
"""

    for i, article in enumerate(data['news'], 1):
        summary += f"{i}. {article['title']} ({article['publisher']})\n"

    if not data['news']:
        summary += "No recent news available.\n"

    return summary
