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


def format_comparison_summary(data_a: Dict[str, Any], data_b: Dict[str, Any]) -> str:
    """
    Format comparison data into a side-by-side summary for LLM consumption.

    Args:
        data_a: Output from fetch_stock_data() for first stock
        data_b: Output from fetch_stock_data() for second stock

    Returns:
        Formatted string with side-by-side comparison suitable for LLM analysis
    """
    # Helper functions to format values safely
    def fmt_currency(val):
        return f"${val:.2f}" if val else "N/A"

    def fmt_pct(val):
        return f"{val:.2f}%" if val else "N/A"

    def fmt_num(val):
        if val is None:
            return "N/A"
        if val >= 1_000_000_000_000:
            return f"${val/1_000_000_000_000:.2f}T"
        elif val >= 1_000_000_000:
            return f"${val/1_000_000_000:.2f}B"
        elif val >= 1_000_000:
            return f"${val/1_000_000:.2f}M"
        return f"{val:,}"

    def fmt_float(val):
        return f"{val:.2f}" if val else "N/A"

    def fmt_pct_mult(val):
        return f"{val*100:.2f}%" if val else "N/A"

    ticker_a = data_a['ticker']
    ticker_b = data_b['ticker']
    company_a = data_a['company_info']
    company_b = data_b['company_info']
    price_a = data_a['price_data']
    price_b = data_b['price_data']
    fund_a = data_a['fundamentals']
    fund_b = data_b['fundamentals']
    hist_a = data_a['historical_data']
    hist_b = data_b['historical_data']

    summary = f"""
## Stock Comparison: {ticker_a} vs {ticker_b}

### Company Overview
| Metric | {ticker_a} | {ticker_b} |
|--------|------------|------------|
| Name | {company_a['name']} | {company_b['name']} |
| Sector | {company_a['sector']} | {company_b['sector']} |
| Industry | {company_a['industry']} | {company_b['industry']} |

### Price Data
| Metric | {ticker_a} | {ticker_b} |
|--------|------------|------------|
| Current Price | {fmt_currency(price_a['current_price'])} | {fmt_currency(price_b['current_price'])} |
| Day Range | {fmt_currency(price_a['day_low'])} - {fmt_currency(price_a['day_high'])} | {fmt_currency(price_b['day_low'])} - {fmt_currency(price_b['day_high'])} |
| 52-Week Range | {fmt_currency(price_a['52_week_low'])} - {fmt_currency(price_a['52_week_high'])} | {fmt_currency(price_b['52_week_low'])} - {fmt_currency(price_b['52_week_high'])} |
| Volume | {fmt_num(price_a['volume'])} | {fmt_num(price_b['volume'])} |
| Avg Volume | {fmt_num(price_a['avg_volume'])} | {fmt_num(price_b['avg_volume'])} |

### Fundamental Metrics
| Metric | {ticker_a} | {ticker_b} |
|--------|------------|------------|
| Market Cap | {fmt_num(fund_a['market_cap'])} | {fmt_num(fund_b['market_cap'])} |
| P/E Ratio | {fmt_float(fund_a['pe_ratio'])} | {fmt_float(fund_b['pe_ratio'])} |
| Forward P/E | {fmt_float(fund_a['forward_pe'])} | {fmt_float(fund_b['forward_pe'])} |
| PEG Ratio | {fmt_float(fund_a['peg_ratio'])} | {fmt_float(fund_b['peg_ratio'])} |
| Dividend Yield | {fmt_pct_mult(fund_a['dividend_yield'])} | {fmt_pct_mult(fund_b['dividend_yield'])} |
| Beta | {fmt_float(fund_a['beta'])} | {fmt_float(fund_b['beta'])} |
| EPS | {fmt_currency(fund_a['eps'])} | {fmt_currency(fund_b['eps'])} |
| Profit Margin | {fmt_pct_mult(fund_a['profit_margin'])} | {fmt_pct_mult(fund_b['profit_margin'])} |
| Revenue Growth | {fmt_pct_mult(fund_a['revenue_growth'])} | {fmt_pct_mult(fund_b['revenue_growth'])} |

### Performance Trends
| Metric | {ticker_a} | {ticker_b} |
|--------|------------|------------|
| 90-Day Return | {fmt_pct(hist_a['90_day_return'])} | {fmt_pct(hist_b['90_day_return'])} |
| Volatility | {fmt_pct(hist_a['volatility'])} | {fmt_pct(hist_b['volatility'])} |

### Recent News - {ticker_a}
"""
    for i, article in enumerate(data_a['news'], 1):
        summary += f"{i}. {article['title']} ({article['publisher']})\n"
    if not data_a['news']:
        summary += "No recent news available.\n"

    summary += f"\n### Recent News - {ticker_b}\n"
    for i, article in enumerate(data_b['news'], 1):
        summary += f"{i}. {article['title']} ({article['publisher']})\n"
    if not data_b['news']:
        summary += "No recent news available.\n"

    return summary
