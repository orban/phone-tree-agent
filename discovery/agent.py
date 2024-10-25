import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger
from discovery.phone_tree import PhoneTree
from call_management.call_manager import CallManager
from discovery.output_generator import OutputGenerator

class DiscoveryAgent:
    def __init__(
        self,
        call_manager: CallManager,
        output_generator: OutputGenerator,
        openai_client
    ) -> None:
        self.call_manager: CallManager = call_manager
        self.output_generator: OutputGenerator = output_generator
        self.phone_tree: PhoneTree = PhoneTree(openai_client)

    async def explore_phone_tree(
        self, phone_number: str
    ) -> Dict[Tuple[str, ...], Dict[str, Any]]:
        results = {}
        paths_to_explore = self.phone_tree.get_unexplored_paths()
        while paths_to_explore:
            for path in paths_to_explore:
                result = await self.explore_path(phone_number, path)
                results[tuple(path)] = result
                extracted_path = await self.phone_tree.extract_path(result['transcription'])
                await self.phone_tree.add_path(extracted_path)
            paths_to_explore = self.phone_tree.get_unexplored_paths()
        return results

    async def explore_path(
        self, phone_number: str, current_path: List[str] = []
    ) -> Dict[str, Any]:
        prompt = self.generate_prompt(current_path)
        call_result = await self.call_manager.make_call(phone_number, prompt)

        if call_result.status != "completed":
            return {
                "path": current_path,
                "error": call_result.message,
                "id": call_result.id,
                "status": call_result.status,
            }

        return {
            "path": current_path,
            "transcription": call_result.transcription,
        }

    def generate_prompt(self, current_path: List[str]) -> str:
        if not current_path:
            return "Start a new conversation. Ask if I'm an existing customer and proceed from there."
        else:
            path_str = " -> ".join(current_path)
            return f"""
            Continue the conversation following this path: {path_str}
            If you reach the end of this path, explore any unexplored options.
            Use the exact labels provided in the path when referring to decision points and choices.
            """

    def generate_results(self) -> Dict[Tuple[str, ...], Dict[str, Any]]:
        def dfs(node, current_path):
            results = {}
            path_tuple = tuple(current_path)
            results[path_tuple] = {
                "path": list(path_tuple),
                "context": node.content,
            }
            for child_label, child_node in node.children.items():
                results.update(dfs(child_node, current_path + [child_label]))
            return results

        return dfs(self.phone_tree.root, [])
