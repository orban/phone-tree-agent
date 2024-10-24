import asyncio
from typing import List, Dict, Optional
from loguru import logger
import json
from pydantic import BaseModel
from discovery.phone_tree import PhoneTree
from call_management.call_manager import CallManager
from discovery.output_generator import OutputGenerator


class Config(BaseModel):
    OPENAI_API_KEY: str
    API_URL: str
    API_TOKEN: str
    WEBHOOK_URL: str
    AGENT_PHONE_NUMBER: str
    CONCURRENT_CALLS: int
    MAX_DEPTH: int
    TIMEOUT_SECONDS: int


class DiscoveryAgent:
    """
    Discovers the phone tree for a business by exploring all possible paths.
    """

    def __init__(
        self,
        call_manager: CallManager,
        output_generator: OutputGenerator,
    ) -> None:
        self.call_manager: CallManager = call_manager
        self.output_generator: OutputGenerator = output_generator
        self.phone_tree: PhoneTree = PhoneTree()
        self.exploration_queue: asyncio.Queue[List[str]] = asyncio.Queue()
        self.max_depth: int = self.call_manager.config.MAX_DEPTH

    async def explore_phone_tree(self, phone_number: str) -> Dict:
        """
        Explores the phone tree for a business by exploring all possible paths.
        """
        initial_result = await self.explore_path(phone_number)
        self.phone_tree.add_path([], initial_result)
        await self.output_generator.update_progress(self.phone_tree)

        await self.exploration_queue.put([])  # Start with the root path

        workers = [
            asyncio.create_task(self.worker(phone_number))
            for _ in range(
                self.call_manager.config.CONCURRENT_CALLS,
            )
        ]

        await self.exploration_queue.join()
        for worker in workers:
            worker.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

        return self.phone_tree.to_dict()

    async def worker(self, phone_number: str) -> None:
        """
        Worker function that explores the phone tree for a business.
        """
        while True:
            current_path = await self.exploration_queue.get()
            try:
                result = await self.explore_path(phone_number, current_path)
                self.phone_tree.add_path(current_path, result)
                await self.output_generator.update_progress(self.phone_tree)

                current_node = self.phone_tree.get_node(current_path)
                for option in result.get("options", []):
                    new_path = current_path + [option]
                    if (
                        len(new_path) <= self.max_depth
                        and option not in current_node.explored_options
                        and not self.phone_tree.is_fully_explored(new_path)
                    ):
                        await self.exploration_queue.put(new_path)
                        current_node.explored_options.add(option)
            except Exception as e:
                logger.error(f"Error exploring path {current_path}: {str(e)}")
            finally:
                self.exploration_queue.task_done()

    async def _extract_options(
        self, transcription: str, current_path: List[str]
    ) -> List[str]:
        function_description = {
            "name": "extract_options",
            "description": "Extract all available options from a phone tree conversation",
            "parameters": {
                "type": "object",
                "properties": {
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of all available options extracted from the transcription",
                    }
                },
                "required": ["options"],
            },
        }
        prompt = f"""
        Given the following transcription from a phone tree conversation, 
        extract ALL available options at the current decision point.
        Include both explicitly mentioned options (e.g., "Press 1 for...") and any implied options.
        If no specific options are mentioned, infer possible options based on the context.
        Consider options for both new and returning customers, emergencies and non-emergencies, etc.
        
        Examples:
        1. Transcription: "For sales, press 1. For support, press 2."
           Options: ["Press 1 for sales", "Press 2 for support"]
           
        2. Transcription: "Welcome to XYZ Corp. Are you a new or existing customer?"
           Options: ["New customer", "Existing customer"]
           
        3. Transcription: "Thank you for calling ABC Plumbing. How may we assist you today?"
            Options: []

        Transcription: {transcription}

        Current path: {' -> '.join(current_path)}

        Output the options in a consistent format, e.g., "Press X for Y" or "Say X for Y".
        """
        response = await self.call_manager.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that "
                    "extracts all possible options from phone tree conversations.",
                },
                {"role": "user", "content": prompt},
            ],
            functions=[function_description],  # type: ignore
            function_call={"name": "extract_options"},
            temperature=0,
        )
        options = json.loads(response.choices[0].message.function_call.arguments)[
            "options"
        ]
        return options

    async def explore_path(
        self,
        phone_number: str,
        current_path: Optional[List[str]] = None,
    ) -> Dict:
        """
        Explores a single path in the phone tree.
        """
        if current_path is None:
            current_path = []

        if current_path and self.detect_loop(current_path):
            logger.warning(f"Loop detected in path: {current_path}")
            return {
                "path": current_path,
                "error": "Loop detected",
                "options": []
            }

        prompt = self.generate_prompt(current_path)
        logger.info(f"Using prompt: {prompt}")
        call_result = await self.call_manager.make_call(phone_number, prompt)

        if call_result.status == "completed":
            if not call_result.transcription:
                logger.warning(f"No transcription found for call {call_result.id}")
                return {
                    "path": current_path,
                    "error": "No transcription found",
                    "id": call_result.id,
                    "status": call_result.status,
                }
            options = await self._extract_options(
                call_result.transcription, current_path
            )
            logger.info(f"Extracted options: {options}")

            # Check if the AI's response is in the extracted options
            ai_response = call_result.transcription.split()[
                -1
            ]  # Get the last word of the transcription
            if ai_response.upper() == "END":
                logger.info(f"Exploration ended for path: {current_path}")
                return {
                    "path": current_path,
                    "transcription": call_result.transcription,
                    "options": [],
                }
            if ai_response not in options:
                logger.warning(
                    f"AI's response '{ai_response}' not in extracted options. Retrying."
                )
                return await self.explore_path(
                    phone_number, current_path
                )  # Retry the same path

            result = {
                "path": current_path,
                "transcription": call_result.transcription,
                "options": options,
            }
            return result
        else:
            logger.warning(
                f"Call failed for path {current_path}: {call_result.message}"
            )
            return {
                "path": current_path,
                "error": call_result.message,
                "id": call_result.id,
                "status": call_result.status,
            }

    def generate_prompt(self, current_path: Optional[List[str]]) -> str:
        """
        Generates a prompt for the AI assistant to explore a phone tree.
        """
        base_prompt = (
            "You are an AI assistant exploring a phone tree. "
            "Your task is to navigate through all possible options to map out the entire tree structure. "
            "Listen carefully to the prompts and respond with the appropriate number or keyword for each option. "
            "If asked a question, provide a generic answer that allows you to proceed. "
            "Your goal is to explore new paths, so avoid repeating options you've already selected. "
            "If you reach an end point or a loop, say 'END' to indicate the exploration of this path is complete. "
            "Respond only with your selected option, answer, or 'END', without explanations."
        )

        if not current_path:
            return base_prompt + " Begin the call now."
        else:
            return (
                f"{base_prompt}\n"
                f"You have already navigated through the following options: {' -> '.join(current_path)}. "
                f"Continue exploring from this point by selecting a new, unexplored option or say 'END' if all options have been explored."
            )

    def detect_loop(self, current_path: List[str]) -> bool:
        if len(current_path) < 2:
            return False
        current_node = self.phone_tree.root
        for option in current_path[:-1]:
            if option not in current_node.children:
                return False
            current_node = current_node.children[option]
        return current_node.data == self.phone_tree.get_node(current_path).data
