"""
Shared AgentExecutor for both Finance and Business Process tracks.

Bridges A2A protocol to Solstice-EIM agent logic. The DefaultRequestHandler
handles task creation and terminal state guards; our executor just needs to
use the TaskUpdater to report status and artifacts.
"""

import logging
import traceback

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message

log = logging.getLogger("agentbeats.executor")


class Executor(AgentExecutor):
    """
    Bridges A2A protocol to Solstice-EIM agent logic.

    Instantiated by server.py with track="finance" or track="bizprocess".
    Each context_id gets its own Agent instance for stateless evaluation.
    """

    def __init__(self, track: str = "finance"):
        self.track = track
        self.agents: dict[str, object] = {}

    def _create_agent(self):
        """Factory: create the right agent for this track."""
        if self.track == "finance":
            from agent_finance import FinanceAgent
            return FinanceAgent()
        else:
            from agent_bizprocess import BizProcessAgent
            return BizProcessAgent()

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # DefaultRequestHandler already validates message and task state.
        # We get task_id and context_id from the RequestContext.
        task_id = context.task_id
        context_id = context.context_id or task_id

        updater = TaskUpdater(event_queue, task_id, context_id)

        try:
            # Get or create agent for this context
            agent = self.agents.get(context_id)
            if not agent:
                agent = self._create_agent()
                self.agents[context_id] = agent

            # Execute agent logic — agent calls updater to emit artifacts
            await agent.run(context.message, updater)

        except Exception as e:
            log.error(f"Agent execution failed: {e}")
            traceback.print_exc()
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {e}"),
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise NotImplementedError("Cancel not supported")
