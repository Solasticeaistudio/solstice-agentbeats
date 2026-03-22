"""
Business Process Agent — Purple agent for the AgentBeats Business Process Track.

Routes queries through Solstice-EIM's AgentOrchestrator for multi-agent
delegation, with ReAct fallback for complex multi-step reasoning.
"""

import json
import logging

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from prompts.bizprocess_system import BIZPROCESS_SYSTEM_PROMPT
from solstice_bridge import get_bridge
from tool_registry import execute_tool, parse_tool_call

log = logging.getLogger("agentbeats.bizprocess")


class BizProcessAgent:
    """
    Business Process track purple agent.

    Strategy:
    1. First try auto-delegation via AgentOrchestrator (picks the best
       specialist from the 86-agent Pantheon).
    2. If confidence is too low or delegation fails, fall back to a
       ReAct reasoning loop with tool-calling.
    """

    MAX_STEPS = 8
    AUTO_DELEGATE_CONFIDENCE_THRESHOLD = 10.0

    def __init__(self):
        self.bridge = get_bridge()
        self.history: list[dict] = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process an A2A message through the business process pipeline."""
        input_text = get_message_text(message)
        log.info(f"BizProcess query: {input_text[:120]}...")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Processing business request..."),
        )

        # Step 1: Try auto-delegation to best specialist agent
        delegation = await self.bridge.auto_delegate(
            query=input_text,
            session_id="agentbeats",
        )

        if (
            delegation.get("success")
            and delegation.get("confidence", 0) >= self.AUTO_DELEGATE_CONFIDENCE_THRESHOLD
            and delegation.get("agent") != "litellm_fallback"
        ):
            log.info(
                f"Auto-delegated to {delegation['agent']} "
                f"(confidence: {delegation['confidence']:.1f})"
            )
            final_response = delegation["text"]
        else:
            # Step 2: ReAct fallback with tool-calling
            log.info("Auto-delegation insufficient, using ReAct loop")
            final_response = await self._react_loop(input_text, updater)

        # Emit artifact
        try:
            result_json = json.loads(final_response)
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=result_json))],
                name="BusinessProcessResult",
            )
        except (json.JSONDecodeError, TypeError):
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=final_response))],
                name="BusinessProcessResult",
            )

        await updater.update_status(TaskState.completed)

    async def _react_loop(
        self, query: str, updater: TaskUpdater
    ) -> str:
        """ReAct reasoning loop for complex multi-step business queries."""
        # Retrieve relevant context
        context = ""
        try:
            rag_results = await self.bridge.retrieve(query=query, agent="sol", top_k=3)
            if rag_results:
                context = "\n".join(
                    r.get("text", "")[:300]
                    for r in rag_results
                    if isinstance(r, dict)
                )
        except Exception as e:
            log.debug(f"RAG retrieval skipped: {e}")

        prompt = query
        if context:
            prompt = f"Relevant context:\n{context}\n\nQuery: {query}"

        full_prompt = f"{BIZPROCESS_SYSTEM_PROMPT}\n\n{prompt}"
        self.history = []

        response_text = ""
        for step in range(self.MAX_STEPS):
            if step == 0:
                call_prompt = full_prompt
            else:
                call_prompt = self.history[-1]["content"]

            response_text, meta = await self.bridge.generate(
                prompt=call_prompt,
                agent_name="sol",
                history=self.history if step > 0 else [],
            )

            tool_call = parse_tool_call(response_text)
            if tool_call:
                log.info(f"Step {step + 1}: tool call -> {tool_call['name']}")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Step {step + 1}: {tool_call['name']}..."
                    ),
                )

                tool_result = await execute_tool(tool_call, self.bridge)

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

            # Final answer
            break

        return response_text
