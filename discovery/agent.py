import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger
from discovery.phone_tree import PhoneTree
from call_management.call_manager import CallManager
from discovery.output_generator import OutputGenerator
from openai import AsyncOpenAI


class DiscoveryAgent:
    def __init__(
        self,
        call_manager: CallManager,
        output_generator: OutputGenerator,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.call_manager: CallManager = call_manager
        self.output_generator: OutputGenerator = output_generator
        self.phone_tree: PhoneTree = PhoneTree(openai_client, output_generator)

    async def explore_phone_tree(
        self, phone_number: str
    ) -> Dict[Tuple[str, ...], Dict[str, Any]]:
        results: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                paths_to_explore: List[
                    List[str]
                ] = await self.phone_tree.get_unexplored_paths()
                while paths_to_explore:
                    for path in paths_to_explore:
                        result: Dict[str, Any] = await self.explore_path(
                            phone_number, path
                        )
                        results[tuple(path)] = result
                        extracted_path: List[
                            Tuple[str, str]
                        ] = await self.phone_tree.extract_path(result["transcription"])
                        validated_path = await self.phone_tree.validate_path(
                            extracted_path
                        )
                        await self.phone_tree.add_path(validated_path)
                    paths_to_explore = await self.phone_tree.get_unexplored_paths()

                if await self.phone_tree.validate_tree():
                    break
            except Exception as e:
                logger.error(f"Error during exploration: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Exploration failed.")
                    raise

        return results

    async def explore_path(self, phone_number: str, current_path: List[str] = []):
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                prompt: str = await self.generate_prompt(current_path)
                call_result = await self.call_manager.make_call(phone_number, prompt)

                if call_result.status != "completed":
                    logger.warning(f"Call not completed: {call_result.status}")
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
            except Exception as e:
                logger.error(f"Error during path exploration: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Path exploration failed.")
                    raise

    async def generate_prompt(self, current_path: List[str]) -> str:
        base_instructions: str = (
            "You are an AI assistant exploring a business phone tree. "
            "Your goal is to map out the entire tree structure by navigating through all possible options. "
            "Keep responses brief and to the point. Do not offer additional information or services. "
            "If asked for specific details (e.g., name, address), provide fictional but realistic information. "
            "Always maintain a polite and professional tone. "
        )

        if not current_path:
            return base_instructions + (
                "Start the call and navigate through the initial options. "
                "Example interaction:\n"
                "Agent: Hello, thank you for calling. Are you an existing customer?\n"
                "Customer: Yes.\n"
                "Agent: Is this an emergency?\n"
                "Customer: No.\n"
                "Agent: What kind of issue are you facing?\n"
                "Customer: Air conditioning.\n"
                "Continue the conversation, making choices to explore a single path. "
                "You do not need to follow the exact path provided in the example, but do your best to stay on track. "
                "If you reach the end of a path, end the call by saying 'Goodbye' and wait for the agent to end the call."
            )

        else:
            path_str: str = " -> ".join(current_path)
            tree_graph: str = await self.output_generator.generate_mermaid_graph(
                self.phone_tree
            )
            return (
                base_instructions
                + f"""
                You previously explored the following path: {path_str}
                This is the current state of the phone tree:\n
                {tree_graph}\n
                Continue the conversation, making choices to explore a single unexplored path.
                Use the exact labels provided in the path when referring to decision points and choices.

                If you reach the end of this path:
                1. If there are unexplored options, choose one that hasn't been explored before.
                2. If all options have been explored or you've reached a dead end, say 'Goodbye' and wait for the agent to end the call.
                Remember:
                - Only explore new options if you've completed the given path.
                - End the call if there are no more options to explore.
                - Do not say 'END' or attempt to hang up yourself; wait for the agent to end the call after you say 'Goodbye'.
                """
            )

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

    async def simplify_tree(self):
        """
        Simplify the phone tree by merging similar nodes and removing redundancies.
        """
        await self.phone_tree.simplify_tree()
