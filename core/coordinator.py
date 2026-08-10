"""Compatibility coordinator used by the core routing integration surface.

The repository's operational coordinator lives in ``distributed.coordinator``.
This adapter preserves the older ``core.coordinator.Coordinator`` contract
expected by routing/integration callers without duplicating scheduling logic.
"""

from typing import Any, Dict, List, Optional

from distributed.coordinator import coordinator as _distributed


class Coordinator:
    """Thin compatibility adapter over the canonical distributed coordinator."""

    def get_available_agents(self) -> List[str]:
        return [
            agent_id
            for agent_id, agent in _distributed.agents.items()
            if agent.get("status") == "active"
        ]

    def is_agent_available(self, agent_id: str) -> bool:
        agent = _distributed.agents.get(agent_id)
        return bool(agent and agent.get("status") == "active")

    def assign_task(
        self,
        task_type: str,
        task_data: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        callback=None,
    ):
        return _distributed.assign_task(
            task_type,
            task_data or {},
            priority=priority,
            callback=callback,
        )

    def get_agent_responses(self, query: str):
        """Compatibility hook for callers that supply their own response executor."""
        return []

    def execute_strategy(self, query: str, agent_id: str, context=None):
        """Compatibility hook for strategy executors layered above coordination."""
        return []
