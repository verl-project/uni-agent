from llm_router.manager import LLMRouter

# Trainer config loads this via FQN as `AgentLoopManager`.
AgentLoopManager = LLMRouter

__all__ = ["LLMRouter", "AgentLoopManager"]
