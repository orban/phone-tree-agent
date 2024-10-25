import pytest
from unittest.mock import AsyncMock, MagicMock
from discovery.agent import DiscoveryAgent

class TestDiscoveryAgent:
    @pytest.mark.asyncio
    async def test_explore_phone_tree_all_paths(self):
        call_manager = AsyncMock()
        output_generator = AsyncMock()
        agent = DiscoveryAgent(call_manager, output_generator)

        call_manager.make_call.return_value = AsyncMock(status="completed", transcription="Press 1 for sales, Press 2 for support")
        call_manager.openai.chat.completions.create.return_value = AsyncMock(choices=[MagicMock(message=MagicMock(function_call=MagicMock(name="extract_options", arguments=json.dumps({"options": ["sales", "support"]})))])

        results = await agent.explore_phone_tree("1234567890")
        assert results == {
            (): {"path": [], "transcription": "Press 1 for sales, Press 2 for support", "options": ["sales", "support"]},
            ('sales',): {"path": ["sales"], "transcription": "Press 1 for sales, Press 2 for support", "options": ["sales", "support"]},
            ('support',): {"path": ["support"], "transcription": "Press 1 for sales, Press 2 for support", "options": ["sales", "support"]}
        }

    @pytest.mark.asyncio
    async def test_explore_path_call_failure(self):
        call_manager = AsyncMock()
        output_generator = AsyncMock()
        agent = DiscoveryAgent(call_manager, output_generator)

        call_manager.make_call.return_value = AsyncMock(status="failed", message="Network error", id="123")

        result = await agent.explore_path("1234567890")
        assert result == {
            "path": [],
            "error": "Network error",
            "id": "123",
            "status": "failed"
        }

    @pytest.mark.asyncio
    async def test_extract_options_no_options_found(self):
        call_manager = AsyncMock()
        output_generator = AsyncMock()
        agent = DiscoveryAgent(call_manager, output_generator)

        call_manager.openai.chat.completions.create.return_value = AsyncMock(choices=[MagicMock(message=MagicMock(function_call=MagicMock(name="extract_options", arguments=json.dumps({"options": []})))])

        options = await agent._extract_options("No options available", [])
        assert options == []