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

logger = logging.getLogger(__name__)


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

        return {
            "ticker": ticker.upper(),
            "fetch_timestamp": datetime.now().isoformat(),
            "company_info": company_info,
            "price_data": price_data,
            "fundamentals": fundamentals,
            "historical_data": historical_data,
            "news": news_items,
        }

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

    summary = f"""
## Research Summary for {data['ticker']} - {company['name']}

### Company Overview
- Sector: {company['sector']}
- Industry: {company['industry']}
- Description: {company['description'][:200] if company['description'] else 'N/A'}...

### Current Price Data
- Current Price: ${price['current_price']:.2f if price['current_price'] else 0}
- Day Range: ${price['day_low']:.2f if price['day_low'] else 0} - ${price['day_high']:.2f if price['day_high'] else 0}
- 52-Week Range: ${price['52_week_low']:.2f if price['52_week_low'] else 0} - ${price['52_week_high']:.2f if price['52_week_high'] else 0}
- Volume: {price['volume']:,} (Avg: {price['avg_volume']:,})

### Fundamental Metrics
- Market Cap: ${fundamentals['market_cap']:,.0f if fundamentals['market_cap'] else 0}
- P/E Ratio: {fundamentals['pe_ratio']:.2f if fundamentals['pe_ratio'] else 'N/A'}
- Dividend Yield: {fundamentals['dividend_yield']*100:.2f if fundamentals['dividend_yield'] else 0}%
- Beta: {fundamentals['beta']:.2f if fundamentals['beta'] else 'N/A'}
- Profit Margin: {fundamentals['profit_margin']*100:.2f if fundamentals['profit_margin'] else 0}%

### Performance Trends
- 90-Day Return: {historical['90_day_return']:.2f if historical['90_day_return'] else 0}%
- Volatility: {historical['volatility']:.2f if historical['volatility'] else 0}%

### Recent News
"""

    for i, article in enumerate(data['news'], 1):
        summary += f"{i}. {article['title']} ({article['publisher']})\n"

    if not data['news']:
        summary += "No recent news available.\n"

    return summary
