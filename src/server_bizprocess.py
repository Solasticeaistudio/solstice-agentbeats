"""
A2A Server entry point — Business Process Agent track.

AgentBeats Phase 2, Sprint 1 (March 2-22, 2026).
Solstice AI Studio — Berkeley RDI AgentX Competition.

Usage:
    uv run src/server_bizprocess.py --host 0.0.0.0 --port 9009
    docker run -p 9009:9009 solstice-bizprocess-agent
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
log = logging.getLogger("agentbeats.server.bizprocess")


def main():
    parser = argparse.ArgumentParser(description="Solstice Business Process Purple Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, default=None)
    args = parser.parse_args()

    card_url = args.card_url or f"http://{args.host}:{args.port}"

    skills = [
        AgentSkill(
            id="process_execution",
            name="Business Process Execution",
            description=(
                "Execute multi-step business workflows including customer service, "
                "procurement, onboarding, and operational task automation with "
                "policy compliance verification"
            ),
            tags=[
                "business-process", "workflow", "customer-service",
                "operations", "automation", "compliance",
            ],
            examples=[
                "Process a customer refund request for order #12345",
                "Execute the vendor onboarding workflow for Acme Corp",
                "Handle an employee offboarding checklist",
            ],
        ),
        AgentSkill(
            id="document_analysis",
            name="Business Document Analysis",
            description=(
                "Analyze contracts, compliance documents, invoices, proposals, "
                "and operational reports using legal intelligence (Athena) "
                "and financial analysis (Plutus)"
            ),
            tags=[
                "contracts", "compliance", "invoices", "reporting",
                "legal", "analysis",
            ],
            examples=[
                "Review this vendor contract for risk clauses and unfavorable terms",
                "Analyze variance in the Q3 operations report",
                "Extract key terms from this NDA",
            ],
        ),
        AgentSkill(
            id="strategic_reasoning",
            name="Strategic Business Reasoning",
            description=(
                "Multi-step reasoning for complex business decisions, "
                "scenario modeling, competitive analysis, and strategic "
                "planning backed by 86 specialized agents"
            ),
            tags=[
                "strategy", "reasoning", "scenarios", "planning",
                "competitive-analysis", "decision-making",
            ],
            examples=[
                "Evaluate trade-offs of expanding into the APAC market",
                "Model three scenarios for Q3 headcount planning",
                "Analyze the competitive landscape for enterprise AI platforms",
            ],
        ),
    ]

    agent_card = AgentCard(
        name="Solstice Business Process Agent",
        description=(
            "Enterprise business process agent with multi-step workflow execution, "
            "strategic reasoning, contract analysis, and policy compliance. "
            "Powered by the Solstice Pantheon — a 86-agent intelligence network "
            "with specialists in legal (Athena), finance (Plutus), analytics (Prism), "
            "security (Ares), and 82 more domain experts."
        ),
        url=card_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(track="bizprocess"),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    log.info(f"Starting Solstice Business Process Agent on {args.host}:{args.port}")
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
