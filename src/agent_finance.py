"""
Finance Agent — Purple agent for the AgentBeats Finance Track.

ReAct-style reasoning loop backed by Solstice-EIM's Plutus financial
intelligence, LLM service, and Pinecone RAG.
"""

import json
import logging

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from prompts.finance_system import FINANCE_SYSTEM_PROMPT, build_finance_prompt
from solstice_bridge import get_bridge
from tool_registry import execute_tool, parse_tool_call

log = logging.getLogger("agentbeats.finance")


class FinanceAgent:
    """
    Finance track purple agent.

    Implements a ReAct reasoning loop:
    1. Receive financial query
    2. Optionally search knowledge base for context
    3. Reason step-by-step, calling tools as needed
    4. Synthesize final analysis and emit as artifact
    """

    MAX_STEPS = 8

    def __init__(self):
        self.bridge = get_bridge()
        self.history: list[dict] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process an A2A message through the finance reasoning pipeline."""
        input_text = get_message_text(message)
        log.info(f"Finance query: {input_text[:120]}...")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing financial query..."),
        )

        # Optional: retrieve relevant context from knowledge base
        context = ""
        try:
            rag_results = await self.bridge.retrieve(
                query=input_text, agent="plutus", top_k=3
            )
            if rag_results:
                context = "\n".join(
                    r.get("text", "")[:300]
                    for r in rag_results
                    if isinstance(r, dict)
                )
        except Exception as e:
            log.debug(f"RAG retrieval skipped: {e}")

        # Build the full prompt (system + context + query in one shot)
        prompt = build_finance_prompt(input_text, context)
        full_prompt = f"{FINANCE_SYSTEM_PROMPT}\n\n{prompt}"

        # ReAct loop — pass prompt only (not duplicated history)
        self.history = []
        final_response = ""
        for step in range(self.MAX_STEPS):
            if step == 0:
                # First call: full prompt with system instructions
                call_prompt = full_prompt
            else:
                # Subsequent calls: just the latest observation
                call_prompt = self.history[-1]["content"]

            response_text, meta = await self.bridge.generate(
                prompt=call_prompt,
                agent_name="plutus",
                history=self.history if step > 0 else [],
            )

            # Check for tool call
            tool_call = parse_tool_call(response_text)
            if tool_call:
                log.info(f"Step {step + 1}: tool call -> {tool_call['name']}")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Step {step + 1}: executing {tool_call['name']}..."
                    ),
                )

                tool_result = await execute_tool(tool_call, self.bridge)

                # Feed observation back into the conversation
                self.history.append({"role": "assistant", "content": response_text})
                self.history.append({
                    "role": "user",
                    "content": (
                        f"Tool '{tool_call['name']}' returned:\n"
                        f"{json.dumps(tool_result, indent=2, default=str)}\n\n"
                        f"Continue your analysis. If you have enough information, "
                        f"provide your final answer directly (not as a tool call)."
                    ),
                })
                continue

            # No tool call — this is the final answer
            final_response = response_text
            break
        else:
            # Exhausted steps — use last response
            final_response = response_text

        # Emit artifact
        try:
            result_json = json.loads(final_response)
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=result_json))],
                name="FinancialAnalysis",
            )
        except (json.JSONDecodeError, TypeError):
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=final_response))],
                name="FinancialAnalysis",
            )

        await updater.update_status(TaskState.completed)
