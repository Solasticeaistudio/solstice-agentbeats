"""
Tool definitions and execution for ReAct-style reasoning.

Tools are defined as metadata dicts. The LLM sees tool descriptions in
the system prompt and emits JSON tool calls. parse_tool_call() extracts
them, execute_tool() routes to the appropriate handler.

Computation tier:
  REAL MATH  — pure Python/numpy/scipy, zero LLM cost, deterministic
  LIVE DATA  — yfinance / Pinecone, real data, minor latency
  LLM-BACKED — delegated to Solstice specialists for narrative/reasoning
"""

import json
import logging
import math
import re
from typing import Any, Dict, Optional

log = logging.getLogger("agentbeats.tools")

# ── Tool call extraction ──────────────────────────────────────────────

TOOL_CALL_PATTERN = re.compile(
    r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"parameters"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a tool call from LLM output.

    Returns {"name": "tool_name", "parameters": {...}} or None.
    """
    text = text.strip()

    # Try full JSON parse first (entire response is a tool call)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool" in parsed:
            return {
                "name": parsed["tool"],
                "parameters": parsed.get("parameters", {}),
            }
    except (json.JSONDecodeError, TypeError):
        pass

    # Try regex extraction (tool call embedded in text)
    match = TOOL_CALL_PATTERN.search(text)
    if match:
        try:
            return {
                "name": match.group(1),
                "parameters": json.loads(match.group(2)),
            }
        except json.JSONDecodeError:
            pass

    return None


# ── Black-Scholes (inline — no Plutus needed) ─────────────────────────

def _black_scholes(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call",
) -> Dict[str, Any]:
    """
    Black-Scholes option pricing with full Greeks.

    S      = current stock price
    K      = strike price
    T      = time to expiration (years)
    r      = risk-free rate (annual, decimal)
    sigma  = implied volatility (annual, decimal)
    """
    from scipy.stats import norm

    if T <= 0:
        intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        return {"price": round(intrinsic, 4), "delta": 1.0 if intrinsic > 0 else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "intrinsic_value": round(intrinsic, 4), "time_value": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * discount * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * discount * norm.cdf(d2)
        ) / 365
    else:  # put
        price = K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * discount * norm.cdf(-d2)
        ) / 365

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1% vol change
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)

    return {
        "price": round(price, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "intrinsic_value": round(intrinsic, 4),
        "time_value": round(price - intrinsic, 4),
        "d1": round(d1, 4),
        "d2": round(d2, 4),
        "implied_volatility": round(sigma * 100, 2),
        "option_type": option_type,
    }


# ── Parametric risk metrics (inline — pure numpy/scipy) ───────────────

def _portfolio_risk_metrics(
    annual_return: float,
    annual_volatility: float,
    risk_free_rate: float = 0.045,
    portfolio_value: float = 1_000_000,
    holding_period_days: int = 1,
) -> Dict[str, Any]:
    """Parametric VaR, CVaR, Sharpe, Sortino — no data feed needed."""
    from scipy.stats import norm
    import numpy as np

    hp_return = annual_return * (holding_period_days / 252)
    hp_vol = annual_volatility * math.sqrt(holding_period_days / 252)

    var_95 = portfolio_value * (hp_return - norm.ppf(0.95) * hp_vol)
    var_99 = portfolio_value * (hp_return - norm.ppf(0.99) * hp_vol)

    # CVaR (Expected Shortfall) — analytical for normal distribution
    cvar_95 = portfolio_value * (hp_return - (norm.pdf(norm.ppf(0.05)) / 0.05) * hp_vol)
    cvar_99 = portfolio_value * (hp_return - (norm.pdf(norm.ppf(0.01)) / 0.01) * hp_vol)

    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

    # Sortino (assumes downside std ≈ vol * sqrt(0.5) for symmetric normal)
    downside_vol = annual_volatility * math.sqrt(0.5)
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    # Calmar approximation (using 2-sigma max drawdown estimate)
    max_drawdown_est = 2 * annual_volatility * math.sqrt(1)
    calmar = annual_return / max_drawdown_est if max_drawdown_est > 0 else 0

    return {
        f"var_95_{holding_period_days}d": round(abs(var_95), 2),
        f"var_99_{holding_period_days}d": round(abs(var_99), 2),
        f"cvar_95_{holding_period_days}d": round(abs(cvar_95), 2),
        f"cvar_99_{holding_period_days}d": round(abs(cvar_99), 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio_approx": round(calmar, 3),
        "max_drawdown_estimate_pct": round(max_drawdown_est * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "annual_volatility_pct": round(annual_volatility * 100, 2),
        "portfolio_value": portfolio_value,
        "holding_period_days": holding_period_days,
    }


# ── Tool execution ────────────────────────────────────────────────────

async def execute_tool(
    tool_call: Dict[str, Any],
    bridge: Any,
) -> Dict[str, Any]:
    """
    Execute a tool call against the Solstice bridge.

    Returns a dict with results or error information.
    """
    name = tool_call["name"]
    params = tool_call.get("parameters", {})

    try:
        # ── REAL MATH TOOLS (Plutus engine, zero LLM cost) ────────────

        if name == "calculate_npv":
            cash_flows = params.get("cash_flows", [])
            rate = params.get("discount_rate", 0.08)
            npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
            return {
                "npv": round(npv, 2),
                "discount_rate": rate,
                "cash_flows": cash_flows,
                "periods": len(cash_flows),
                "recommendation": "accept" if npv > 0 else "reject",
            }

        elif name == "calculate_irr":
            cash_flows = params.get("cash_flows", [])
            discount_rate = params.get("discount_rate", 0.10)
            return await bridge.plutus_npv_irr(cash_flows, discount_rate)

        elif name == "dcf_valuation":
            free_cash_flows = params.get("free_cash_flows", [])
            terminal_growth = params.get("terminal_growth_rate", 0.03)
            discount_rate = params.get("discount_rate", 0.10)
            shares = params.get("shares_outstanding", 1)
            if not free_cash_flows:
                return {"error": "free_cash_flows required — provide projected FCFs (e.g., [100, 120, 140])"}
            return await bridge.plutus_dcf(free_cash_flows, terminal_growth, discount_rate, shares)

        elif name == "monte_carlo":
            initial_price = params.get("initial_price", 100.0)
            expected_return = params.get("expected_return", 0.08)
            volatility = params.get("volatility", 0.20)
            days = params.get("days", 252)
            simulations = params.get("simulations", 10000)
            return await bridge.plutus_monte_carlo(
                initial_price, expected_return, volatility, days, simulations
            )

        elif name == "black_scholes":
            S = params.get("S", params.get("stock_price", 100.0))
            K = params.get("K", params.get("strike", 100.0))
            T = params.get("T", params.get("time_to_expiry", 1.0))
            r = params.get("r", params.get("risk_free_rate", 0.05))
            sigma = params.get("sigma", params.get("volatility", 0.20))
            option_type = params.get("option_type", "call")
            return _black_scholes(S, K, T, r, sigma, option_type)

        elif name == "portfolio_risk":
            annual_return = params.get("annual_return", 0.08)
            annual_volatility = params.get("annual_volatility", 0.15)
            risk_free = params.get("risk_free_rate", 0.045)
            port_value = params.get("portfolio_value", 1_000_000)
            holding_days = params.get("holding_period_days", 1)
            return _portfolio_risk_metrics(
                annual_return, annual_volatility, risk_free, port_value, holding_days
            )

        elif name == "budget_variance":
            budgeted = params.get("budgeted", {})
            actual = params.get("actual", {})
            if not budgeted or not actual:
                return {"error": "Both 'budgeted' and 'actual' dicts are required"}
            return await bridge.plutus_budget_variance(budgeted, actual)

        # ── LIVE DATA TOOLS ───────────────────────────────────────────

        elif name == "get_market_data":
            symbol = params.get("symbol", "SPY")
            result = await bridge.plutus_query(
                f"Get current market data and analysis for {symbol}",
            )
            return {"symbol": symbol, "analysis": result.get("text", "")}

        elif name == "knowledge_search":
            query = params.get("query", "")
            results = await bridge.retrieve(query=query, top_k=5)
            return {
                "results": [
                    {"text": r.get("text", "")[:500], "score": r.get("score", 0)}
                    for r in results
                ]
                if results
                else "No results found",
            }

        # ── LLM-BACKED SPECIALIST TOOLS ───────────────────────────────

        elif name == "risk_assessment":
            symbols = params.get("symbols", [])
            period = params.get("period", "1y")
            text, _ = await bridge.generate(
                prompt=(
                    f"Provide a rigorous quantitative risk assessment for {', '.join(symbols)} "
                    f"over the {period} period. Include:\n"
                    f"- Historical volatility (annualized)\n"
                    f"- Value at Risk (VaR) at 95% and 99% confidence\n"
                    f"- Sharpe ratio (assume risk-free rate 4.5%)\n"
                    f"- Beta vs S&P 500\n"
                    f"- Maximum drawdown\n"
                    f"- Sortino ratio\n"
                    f"Show your methodology and intermediate calculations."
                ),
                agent_name="plutus",
            )
            return {"risk_analysis": text, "symbols": symbols, "period": period}

        elif name == "delegate_to_agent":
            agent_name = params.get("agent_name", "sol")
            query = params.get("query", "")
            return await bridge.delegate(
                query=query,
                agent_name=agent_name,
                session_id="agentbeats",
            )

        elif name == "contract_analysis":
            text_input = params.get("text", "")
            result = await bridge.delegate(
                query=f"Analyze this contract for risks and key terms:\n\n{text_input}",
                agent_name="athena",
                session_id="agentbeats",
            )
            return result

        elif name == "scenario_model":
            scenario_type = params.get("scenario_type", "")
            scenario_params = params.get("parameters", {})
            text, _ = await bridge.generate(
                prompt=(
                    f"Model a {scenario_type} business scenario with parameters: "
                    f"{json.dumps(scenario_params)}. Provide projections, risks, "
                    f"and recommendations with specific numbers."
                ),
                agent_name="sol",
            )
            return {"scenario": scenario_type, "analysis": text}

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        log.error(f"Tool execution failed ({name}): {e}")
        return {"error": str(e), "tool": name}
