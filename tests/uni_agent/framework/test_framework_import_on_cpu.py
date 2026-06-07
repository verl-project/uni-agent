def test_framework_imports_with_current_verl_tool_registry():
    from uni_agent.framework.framework import OpenAICompatibleAgentFramework

    assert OpenAICompatibleAgentFramework.__name__ == "OpenAICompatibleAgentFramework"
