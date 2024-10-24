import asyncio
from typing import List, Dict, Optional
from loguru import logger
import json
from pydantic import BaseModel
from textwrap import dedent
from discovery.phone_tree import PhoneTree


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

    def __init__(self, call_manager, output_generator):
        self.call_manager = call_manager
        self.output_generator = output_generator
        self.phone_tree = PhoneTree()
        self.exploration_queue = asyncio.Queue()
        self.max_depth = self.call_manager.config.MAX_DEPTH

    async def explore_phone_tree(self, phone_number: str) -> Dict:
        """
        Explores the phone tree for a business by exploring all possible paths.
        """
        initial_result = await self.explore_path(phone_number)
        self.phone_tree.add_path([], initial_result)
        await self.output_generator.update_progress(self.phone_tree)

        workers = [asyncio.create_task(self.worker(phone_number)) for _ in range(5)]
        for path in self.phone_tree.get_unexplored_paths():
            await self.exploration_queue.put(path)

        await self.exploration_queue.join()
        for worker in workers:
            worker.cancel()

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

                for option in result.get("options", []):
                    new_path = current_path + [option]
                    if not self.phone_tree.is_explored(new_path):
                        await self.exploration_queue.put(new_path)
            finally:
                self.exploration_queue.task_done()

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

        if len(current_path) >= self.max_depth:
            return {"path": current_path, "status": "max_depth_reached"}

        prompt = self.generate_prompt(current_path)
        logger.info(f"Using prompt: {prompt}")
        call_result = await self.call_manager.make_call(phone_number, prompt)

        if call_result.status == "completed":
            options = await self._extract_options(
                call_result.transcription, current_path
            )
            logger.info(f"Extracted options: {options}")
            result = {
                "path": current_path,
                "transcription": call_result.transcription,
                "options": options,
            }
            if not options:
                result["status"] = "end_node"
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
            "You are exploring a phone tree. Navigate through the options, "
            "selecting unexplored choices. Respond only with your selected "
            "option, no explanations. Begin the call now."
        )

        if not current_path:
            return (
                base_prompt
                + " Please start the call and navigate "
                + "through one of the first set of options."
            )
        else:
            return (
                f"{base_prompt}\n"
                f"You have already navigated through the following options: "
                f"{' -> '.join(current_path)}. "
                f"Please continue exploring from this point, "
                f"selecting the next unexplored option."
            )

    async def _extract_options(
        self,
        transcription: str,
        current_path: Optional[List[str]],
    ) -> List[str]:
        """
        Extracts options from a phone tree transcription.
        """
        prompt = dedent(
            f"""
        You are an AI assistant exploring a phone tree for a business. 
        Your goal is to identify and list available options from the transcription.
        Only include clear, distinct options the user can select.
        Exclude explanatory text or other information.
        Return an empty list if no options are found.
        For multiple option sets, only return those relevant to the current path.
        If no path is provided, return the LAST set of options in the transcription.
        
        For example, if given choices like 'Press 1 for existing customers' or 
        'Press 2 for new customers', simply return ['existing customers',
        'new customers'].

        Current Path: {current_path}

        Transcription:
        {transcription}

    Available options:
    """
        ).strip()

        response = await self.call_manager.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts "
                    "options from phone tree transcriptions.",
                },
                {"role": "user", "content": prompt},
            ],
            functions=[
                {
                    "name": "extract_options",
                    "description": "Extracts options from a phone tree transcription",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The list of options extracted "
                                "from the transcription",
                            }
                        },
                        "required": ["options"],
                    },
                }
            ],
            function_call={"name": "extract_options"},
        )

        function_call = response.choices[0].message.function_call
        if function_call and function_call.name == "extract_options":
            options = json.loads(function_call.arguments).get("options", [])
            return options
        else:
            logger.error("Failed to extract options from transcription")
            return []
