"""Smoke tests for the A2A executor, tool registry, and Plutus math bridges."""

import asyncio
import json
import math
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Tool call parsing ─────────────────────────────────────────────────

def test_tool_call_parse_json():
    from tool_registry import parse_tool_call

    text = '{"tool": "get_market_data", "parameters": {"symbol": "AAPL"}}'
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "get_market_data"
    assert result["parameters"]["symbol"] == "AAPL"


def test_tool_call_parse_embedded():
    from tool_registry import parse_tool_call

    text = 'Let me look that up.\n{"tool": "knowledge_search", "parameters": {"query": "treasury yields"}}\n'
    result = parse_tool_call(text)
    assert result is not None
    assert result["name"] == "knowledge_search"
    assert result["parameters"]["query"] == "treasury yields"


def test_tool_call_parse_no_tool():
    from tool_registry import parse_tool_call

    text = "The NPV of this investment is $42,000 at an 8% discount rate."
    result = parse_tool_call(text)
    assert result is None


# ── NPV — pure Python ────────────────────────────────────────────────

def test_npv_calculation():
    """Direct NPV — period 0 is initial outlay."""
    import asyncio
    from unittest.mock import AsyncMock
    from tool_registry import execute_tool

    bridge = AsyncMock()
    tool_call = {
        "name": "calculate_npv",
        "parameters": {
            "cash_flows": [-1000, 200, 300, 400, 500],
            "discount_rate": 0.08,
        },
    }

    result = asyncio.run(execute_tool(tool_call, bridge))
    assert "npv" in result
    assert isinstance(result["npv"], float)
    # NPV of [-1000, 200, 300, 400, 500] at 8% ≈ 127.43
    assert 120 < result["npv"] < 140
    assert result["recommendation"] == "accept"


# ── IRR — via Plutus bridge ───────────────────────────────────────────

def test_irr_calculation():
    """IRR ≈ 16% for [-1000, 400, 450, 500] — verified analytically."""
    from solstice_bridge import SolsticeBridge

    bridge = SolsticeBridge()
    # NPV at 16%: -1000 + 400/1.16 + 450/1.3456 + 500/1.5609 ≈ 0
    result = asyncio.run(bridge.plutus_npv_irr([-1000, 400, 450, 500], 0.10))
    assert "irr" in result
    assert result["irr"] is not None
    assert 14 < result["irr"] < 19, f"IRR out of expected range: {result['irr']}"
    assert "npv" in result
    assert result["npv"] > 0  # positive NPV at 10% (IRR > 10%)


# ── DCF — via Plutus bridge ───────────────────────────────────────────

def test_dcf_valuation():
    """DCF should return enterprise value, PV of FCFs, and terminal value."""
    from solstice_bridge import SolsticeBridge

    bridge = SolsticeBridge()
    # 5-year FCFs growing from 100 to 148, WACC 10%, terminal growth 3%
    fcfs = [100, 110, 121, 133, 146]
    result = asyncio.run(bridge.plutus_dcf(fcfs, 0.03, 0.10, 1))
    assert "enterprise_value" in result
    assert "pv_terminal_value" in result
    assert "pv_projected_fcfs" in result
    # PV of terminal value should dominate total EV
    assert result["pv_terminal_value"] > result["pv_projected_fcfs"]
    assert result["enterprise_value"] > 0


# ── Monte Carlo — via Plutus bridge ──────────────────────────────────

def test_monte_carlo():
    """GBM simulation should return a valid price distribution."""
    from solstice_bridge import SolsticeBridge

    bridge = SolsticeBridge()
    result = asyncio.run(
        bridge.plutus_monte_carlo(
            initial_price=100.0,
            expected_return=0.08,
            volatility=0.20,
            time_horizon_days=252,
            num_simulations=5000,  # smaller for test speed
        )
    )
    assert result["num_simulations"] == 5000
    # Mean should be near expected value (within 3 std of simulation noise)
    assert 80 < result["mean_price"] < 140, f"Mean out of range: {result['mean_price']}"
    # Percentile ordering must hold
    assert result["percentile_5"] < result["percentile_25"] < result["mean_price"]
    assert result["mean_price"] < result["percentile_75"] < result["percentile_95"]
    # 8% expected return → probability above initial should be >50%
    assert result["probability_above_initial"] > 0.50


# ── Black-Scholes — inline scipy ─────────────────────────────────────

def test_black_scholes_call():
    """Classic at-the-money call: S=100, K=100, T=1, r=5%, sigma=20%."""
    from tool_registry import _black_scholes

    result = _black_scholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    # ATM call premium ≈ $10.45 (well-known reference value)
    assert 9.5 < result["price"] < 11.5, f"Call price unexpected: {result['price']}"
    # Delta of ATM call ≈ 0.55–0.65
    assert 0.50 < result["delta"] < 0.70
    assert result["gamma"] > 0
    assert result["vega"] > 0
    assert result["theta"] < 0  # time decay is negative for long options


def test_black_scholes_put_call_parity():
    """Put-call parity: C - P = S - K*exp(-rT)."""
    from tool_registry import _black_scholes

    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.20
    call = _black_scholes(S, K, T, r, sigma, "call")
    put = _black_scholes(S, K, T, r, sigma, "put")
    # C - P = S - K*e^(-rT)
    lhs = call["price"] - put["price"]
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.01, f"Put-call parity violated: {lhs:.4f} != {rhs:.4f}"


# ── Portfolio risk — parametric VaR ──────────────────────────────────

def test_portfolio_risk():
    """Parametric VaR: 95% 1-day VaR on a $1M portfolio."""
    from tool_registry import _portfolio_risk_metrics

    result = _portfolio_risk_metrics(
        annual_return=0.10,
        annual_volatility=0.15,
        risk_free_rate=0.045,
        portfolio_value=1_000_000,
        holding_period_days=1,
    )
    assert "var_95_1d" in result
    assert "sharpe_ratio" in result
    assert "sortino_ratio" in result
    # Sharpe = (10% - 4.5%) / 15% ≈ 0.367
    assert 0.30 < result["sharpe_ratio"] < 0.45
    # Daily VaR at 95% should be positive (dollar loss)
    assert result["var_95_1d"] > 0


# ── Budget variance — via Plutus bridge ──────────────────────────────

def test_budget_variance():
    """Over-budget revenue and under-budget marketing."""
    from solstice_bridge import SolsticeBridge

    bridge = SolsticeBridge()
    result = asyncio.run(
        bridge.plutus_budget_variance(
            budgeted={"Revenue": 1_000_000, "Marketing": 150_000, "COGS": 400_000},
            actual={"Revenue": 950_000, "Marketing": 130_000, "COGS": 420_000},
        )
    )
    assert "category_variances" in result
    assert "total_variance" in result

    rev = result["category_variances"]["Revenue"]
    assert rev["status"] == "under_budget"  # actual < budgeted → unfavorable revenue
    assert rev["variance"] == -50_000

    cogs = result["category_variances"]["COGS"]
    assert cogs["status"] == "over_budget"  # actual > budgeted → unfavorable cost


# ── Messenger utility ─────────────────────────────────────────────────

def test_merge_parts():
    from messenger import merge_parts

    class FakeTextPart:
        def __init__(self, text):
            self.text = text

    class FakePart:
        def __init__(self, text):
            self.root = FakeTextPart(text)

    parts = [FakePart("Hello"), FakePart("World")]
    result = merge_parts(parts)
    assert result == "Hello\nWorld"


# ── Bridge singleton ──────────────────────────────────────────────────

def test_bridge_singleton():
    from solstice_bridge import get_bridge

    b1 = get_bridge()
    b2 = get_bridge()
    assert b1 is b2
