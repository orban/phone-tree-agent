import pytest
from unittest.mock import AsyncMock
from discovery.agent import DiscoveryAgent
from discovery.phone_tree import PhoneTree
from discovery.output_generator import OutputGenerator
from call_management.call_manager import CallManager


@pytest.fixture
def mock_call_manager():
    return AsyncMock(spec=CallManager)


@pytest.fixture
def mock_output_generator():
    return AsyncMock(spec=OutputGenerator)


@pytest.fixture
def mock_openai_client():
    return AsyncMock()


@pytest.fixture
def discovery_agent(mock_call_manager, mock_output_generator, mock_openai_client):
    agent = DiscoveryAgent(mock_call_manager, mock_output_generator, mock_openai_client)
    agent.phone_tree = AsyncMock(spec=PhoneTree)
    return agent


@pytest.mark.asyncio
async def test_explore_phone_tree(discovery_agent):
    discovery_agent.phone_tree.get_unexplored_paths.side_effect = [
        [["root", "option1"]],
        [],
    ]
    discovery_agent.explore_path = AsyncMock(
        return_value={"transcription": "Test transcription"}
    )
    discovery_agent.phone_tree.extract_path = AsyncMock(
        return_value=[("decision", "choice")]
    )

    results = await discovery_agent.explore_phone_tree("1234567890")

    assert len(results) == 1
    assert results[("root", "option1")]["transcription"] == "Test transcription"
    assert discovery_agent.explore_path.call_count == 1
    assert discovery_agent.phone_tree.add_path.call_count == 1


@pytest.mark.asyncio
async def test_explore_path(discovery_agent):
    discovery_agent.generate_prompt = AsyncMock(return_value="Test prompt")
    discovery_agent.call_manager.make_call.return_value = AsyncMock(
        status="completed", transcription="Test transcription"
    )

    result = await discovery_agent.explore_path("1234567890", ["root", "option1"])

    assert result["path"] == ["root", "option1"]
    assert result["transcription"] == "Test transcription"
    discovery_agent.call_manager.make_call.assert_called_once_with(
        "1234567890", "Test prompt"
    )


@pytest.mark.asyncio
async def test_generate_prompt(discovery_agent):
    discovery_agent.output_generator.generate_mermaid_graph = AsyncMock(
        return_value="Test graph"
    )

    prompt = await discovery_agent.generate_prompt(["root", "option1"])

    assert "You are an AI assistant exploring a business phone tree" in prompt
    assert "Test graph" in prompt
