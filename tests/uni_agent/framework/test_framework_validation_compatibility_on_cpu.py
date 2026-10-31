"""Optional trainer lifecycle hooks preserve the existing custom-framework contract."""

import pytest

from uni_agent.framework.base import AgentFramework
from uni_agent.framework.entry import AgentFrameworkWorker


class ExistingFramework(AgentFramework):
    @classmethod
    def from_config(cls, **kwargs):
        return cls()

    async def generate_sequences(self, prompts):
        self.prompts = prompts


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.asyncio
async def test_existing_framework_still_runs_validation_through_worker():
    cls = AgentFrameworkWorker.__ray_metadata__.modified_class
    worker = cls.__new__(cls)
    worker.framework = ExistingFramework()
    await worker.begin_validation("legacy", 1)
    await worker.generate_sequences("prompts")
    await worker.publish_validation_metrics("legacy", {"acc": 0.5})
    assert worker.framework.prompts == "prompts"
    with pytest.raises(NotImplementedError, match="stopping validation"):
        await worker.stop_validation("legacy")
    with pytest.raises(NotImplementedError, match="validation status"):
        await worker.validation_status("legacy")
