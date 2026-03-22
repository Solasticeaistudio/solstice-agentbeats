"""
A2A Server entry point — Finance Agent track.

AgentBeats Phase 2, Sprint 1 (March 2-22, 2026).
Solstice AI Studio — Berkeley RDI AgentX Competition.

Usage:
    uv run src/server_finance.py --host 0.0.0.0 --port 9009
    docker run -p 9009:9009 solstice-finance-agent
"""

import argparse
import logging
import sys

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("agentbeats.server.finance")


def main():
    parser = argparse.ArgumentParser(description="Solstice Finance Purple Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    card_url = args.card_url or f"http://{args.host}:{args.port}"

    skills = [
        AgentSkill(
            id="financial_analysis",
            name="Financial Analysis & Modeling",
            description=(
                "DCF valuation, Monte Carlo simulation, portfolio optimization, "
                "market data analysis, risk assessment, and financial forecasting "
                "powered by the Plutus financial intelligence engine"
            ),
            tags=[
                "finance", "valuation", "dcf", "monte-carlo", "portfolio",
                "risk", "market-analysis", "forecasting",
            ],
            examples=[
                "Analyze AAPL stock and provide a DCF valuation",
                "Run Monte Carlo simulation for portfolio with MSFT, GOOGL, AMZN",
                "What is the risk-adjusted return for a 60/40 portfolio?",
            ],
        ),
        AgentSkill(
            id="financial_computation",
            name="Financial Computation",
            description=(
                "Multi-step financial calculations including NPV, IRR, "
                "Black-Scholes option pricing, budget variance analysis, "
                "and quantitative risk modeling"
            ),
            tags=[
                "computation", "npv", "irr", "options", "black-scholes",
                "budget", "quantitative",
            ],
            examples=[
                "Calculate NPV of cash flows [-1000, 200, 300, 400, 500] at 8% discount rate",
                "Price a European call option: S=100, K=105, T=1, r=0.05, sigma=0.2",
                "Compare IRR of two investment alternatives",
            ],
        ),
        AgentSkill(
            id="financial_reasoning",
            name="Multi-Step Financial Reasoning",
            description=(
                "Strategic financial reasoning, multi-step analysis, "
                "knowledge retrieval from financial databases, and "
                "synthesis of complex financial scenarios"
            ),
            tags=[
                "reasoning", "strategy", "analysis", "knowledge-retrieval",
                "scenarios",
            ],
            examples=[
                "Evaluate the trade-offs between debt and equity financing for a $50M expansion",
                "Analyze the impact of rising interest rates on a diversified portfolio",
                "Compare the financial health of competitors in the EV sector",
            ],
        ),
    ]

    agent_card = AgentCard(
        name="Solstice Finance Agent",
        description=(
            "Enterprise-grade financial intelligence agent with DCF modeling, "
            "Monte Carlo simulation, portfolio optimization, market analysis, "
            "and multi-step financial reasoning. Powered by the Solstice Pantheon "
            "— a 86-agent intelligence network deployed in production enterprise environments."
        ),
        url=card_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(track="finance"),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    log.info(f"Starting Solstice Finance Agent on {args.host}:{args.port}")
    log.info(f"Agent card: {card_url}/.well-known/agent.json")

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=300,
        log_level="info",
    )


if __name__ == "__main__":
    main()
