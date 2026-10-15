"""Select recipe-owned server implementations without changing verl's registry."""

from verl.workers.rollout.llm_server import LLMServerManager


class InferenceServerManager(LLMServerManager):
    def __init__(self, config, **kwargs):
        if config.actor_rollout_ref.rollout.name == "sglang":
            from uni_agent.inference.sglang import SGLangInferenceReplica

            if kwargs.get("worker_group") is not None:
                raise ValueError("The Uni-Agent SGLang adapter supports standalone inference only")
            self.rollout_replica_class = SGLangInferenceReplica
        super().__init__(config, **kwargs)
