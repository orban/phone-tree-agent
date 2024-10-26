from typing import List, Dict, Optional, Tuple, Set
from loguru import logger
from openai import AsyncOpenAI
from collections import defaultdict
import asyncio
import json

logger = logger.bind(name="phone_tree")


class TreeNode:
    """
    Represents a node in the phone tree structure.
    """

    def __init__(self, content: str):
        """
        Initialize a TreeNode.

        Args:
            content (str): The content of the node.
        """
        self.content = content
        self.children: Dict[str, TreeNode] = {}
        self.explored = False


class LabelNormalizer:
    """
    Handles normalization and merging of labels in the phone tree.
    """

    def __init__(self, openai_client: AsyncOpenAI):
        """
        Initialize a LabelNormalizer.
        """
        self.label_groups = defaultdict(set)
        self.primary_labels = {}
        self.openai_client = openai_client
        logger.info("LabelNormalizer initialized with OpenAI client")

    async def get_normalized_label(self, label: str, context: str = "") -> str:
        """
        Get the normalized form of a label.

        Args:
            label (str): The label to normalize.
            context (str): The context to consider for fuzzy matching. Defaults to an empty string.

        Returns:
            str: The normalized label.
        """
        logger.debug(f"Normalizing label: {label} (context: {context})")
        normalized = label.lower().strip()

        # Check if the label is already in a group
        for primary, group in self.label_groups.items():
            if normalized in group:
                return primary

        # If not, use LLM to find the best match or create a new normalized label
        result = await self._normalize_or_match_with_llm(normalized, context)

        if result in self.label_groups:
            logger.info(f"LLM match found: {normalized} -> {result}")
            self.label_groups[result].add(normalized)
        else:
            logger.info(f"New normalized label created: {normalized} -> {result}")
            self.label_groups[result] = {normalized}
            self.primary_labels[result] = result

        return result

    async def _normalize_or_match_with_llm(self, label: str, context: str) -> str:
        """
        Use LLM to find the best match for a label or create a new normalized label.

        Args:
            label (str): The label to find a match for.
            context (str): The context to consider for fuzzy matching.

        Returns:
            str: The best match for the label or the new normalized label.
        """
        existing_labels = list(self.primary_labels.keys())
        labels_formatted = "\n".join(f"- {lbl}" for lbl in existing_labels)
        prompt = f"""
You are an assistant that matches labels for a phone tree system.

Given:

- A list of existing labels:
{labels_formatted}

- New label: "{label}"
- Context: "{context}"

Instructions:

1. If the new label is semantically similar to any existing label, return that existing label exactly as it appears.
2. If it is not similar to any existing label, create a new normalized label following these guidelines:

   - Use lowercase letters with underscores between words (e.g., "schedule_appointment").
   - Remove any numbering or bullets (e.g., "1.", "2.").
   - Keep it concise (maximum 2-3 words) and descriptive.
   - Use specific terms related to customer service and phone trees.
   - Prefer terms such as: existing_customer, emergency, issue_type, schedule_appointment, callback_scheduling.
   - For confirmations or denials, use a simple: yes, no. Do not use 'confirmation_yes' or 'confirmation_no'.
   - Existing customers should be labeled as 'existing_customer'.
   - Avoid overly generic terms or full sentences.
   - Ensure consistency with existing labels.

Important:

- Return only the resulting label as plain text, with no quotation marks or additional text.
"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip().lower()

        return response.choices[0].message.content.strip().lower()

    async def merge_labels(self, label1: str, label2: str) -> str:
        """
        Merge two labels, keeping the shorter one as the normalized form.

        Args:
            label1 (str): The first label to merge.
            label2 (str): The second label to merge.

        Returns:
            str: The merged label.
        """
        logger.info(f"Merging labels: {label1} and {label2}")
        norm1 = label1.lower().strip()
        norm2 = label2.lower().strip()

        group1 = next(
            (group for group in self.label_groups.values() if norm1 in group), None
        )
        group2 = next(
            (group for group in self.label_groups.values() if norm2 in group), None
        )

        if group1 is group2:  # Labels are already in the same group
            return next(
                label for label in self.primary_labels if label.lower() in group1
            )

        # Use LLM to determine which label to keep as primary
        primary_label = await self._choose_primary_label_with_llm(label1, label2)

        # Merge groups if they exist, or create a new group
        new_group = set(group1 or [norm1]) | set(group2 or [norm2])

        # Update label_groups and primary_labels
        self.label_groups = {
            k: v for k, v in self.label_groups.items() if v != group1 and v != group2
        }
        self.label_groups[primary_label.lower()] = new_group
        self.primary_labels = {
            k: v for k, v in self.primary_labels.items() if k.lower() in new_group
        }
        self.primary_labels[primary_label.lower()] = primary_label

        return primary_label

    async def _choose_primary_label_with_llm(self, label1: str, label2: str) -> str:
        """
        Use LLM to determine which label to keep as primary.

        Args:
            label1 (str): The first label to compare.
            label2 (str): The second label to compare.

        Returns:
            str: The primary label.
        """
        prompt = f"""
You are an expert in label taxonomy for a phone tree system.

Given two labels:

1. "{label1}"
2. "{label2}"

Instructions:

- Analyze the two labels to determine which one is more general or overarching in the context of customer service and phone tree navigation.
- The more general label should be selected as the primary label.
- If both labels are equally general, choose the one that is more commonly used or intuitive for users.
- Return only the chosen label as plain text, without any additional explanation or text.

Important:

- Do not include any additional text besides the chosen label.
- Ensure the label is returned exactly as it appears in the given options.
"""

        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )

        return response.choices[0].message.content.strip()


class PhoneTree:
    """
    Represents the entire phone tree structure and provides methods for manipulation and exploration.
    """

    def __init__(self, openai_client: AsyncOpenAI, output_generator):
        """
        Initialize a PhoneTree.

        Args:
            openai_client (AsyncOpenAI): The OpenAI client for API calls.
        """
        self.root = TreeNode("root")
        self.label_normalizer = LabelNormalizer(openai_client)
        self.openai_client = openai_client
        self.output_generator = output_generator
        logger.info("PhoneTree initialized with OpenAI client")

    async def add_path(self, path: List[Tuple[str, str]]) -> None:
        """
        Add a path to the phone tree with idempotency to prevent duplicates.

        Args:
            path (List[Tuple[str, str]]): A list of (decision_point, choice) tuples representing the path.
        """
        if not path:
            logger.warning("Attempted to add an empty path. Skipping.")
            return

        logger.info(f"Adding path: {path}")
        current_node = self.root
        for decision_point, choice in path:
            normalized_decision = await self.label_normalizer.get_normalized_label(decision_point)
            normalized_choice = await self.label_normalizer.get_normalized_label(choice)

            if normalized_decision == "root":
                continue  # Skip the root node as it's already the starting point

            if normalized_decision not in current_node.children:
                decision_node = TreeNode(normalized_decision)
                current_node.children[normalized_decision] = decision_node
                logger.info(f"Created new decision node: {normalized_decision}")
            else:
                decision_node = current_node.children[normalized_decision]
                logger.info(f"Using existing decision node: {normalized_decision}")

            if normalized_choice not in decision_node.children:
                choice_node = TreeNode(normalized_choice)
                decision_node.children[normalized_choice] = choice_node
                logger.info(f"Created new choice node: {normalized_choice}")
            else:
                choice_node = decision_node.children[normalized_choice]
                logger.info(f"Using existing choice node: {normalized_choice}")

            current_node = choice_node

        current_node.explored = True
        logger.info(f"Path added successfully: {path}")

    async def _is_path_explored(self, path: List[Tuple[str, str]]) -> bool:
        """
        Check if the given path has already been explored.

        Args:
            path (List[Tuple[str, str]]): The path to check.

        Returns:
            bool: True if the path is already explored, False otherwise.
        """
        current_node = self.root
        for decision_point, choice in path:
            normalized_decision = await self.label_normalizer.get_normalized_label(
                decision_point, current_node.content
            )
            normalized_choice = await self.label_normalizer.get_normalized_label(
                choice, current_node.content
            )

            if normalized_decision in current_node.children:
                current_node = current_node.children[normalized_decision]
            else:
                return False

            if normalized_choice in current_node.children:
                current_node = current_node.children[normalized_choice]
            else:
                return False

        return current_node.explored

    async def get_unexplored_paths(self) -> List[List[str]]:
        """
        Get all unexplored paths in the phone tree.

        Returns:
            List[List[str]]: A list of unexplored paths, where each path is a list of node labels.
        """

        def dfs(node: TreeNode, current_path: List[str]) -> List[List[str]]:
            if not node.children or not node.explored:
                return [current_path]
            paths = []
            for label, child in node.children.items():
                paths.extend(dfs(child, current_path + [label]))
            return paths

        return dfs(self.root, [])

    async def get_tree_structure(self) -> str:
        """
        Get a string representation of the current tree structure.
        """
        return await self.output_generator.print_tree_to_string(self.root)

    async def extract_path(self, transcript: str) -> List[Tuple[str, str]]:
        logger.info(f"Extracting path from transcript of length {len(transcript)}")
        
        if not transcript.strip():
            logger.warning("Empty transcript provided. Skipping path extraction.")
            return []
        
        tree_structure = await self.output_generator.print_tree_to_string(self.root)
        
        functions = [
            {
                "name": "extract_phone_tree_path",
                "description": "Extract the path taken in a phone tree conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "decision_point": {"type": "string"},
                                    "choice": {"type": "string"},
                                },
                                "required": ["decision_point", "choice"],
                            },
                            "description": "The extracted path as a list of decision points and choices",
                        }
                    },
                    "required": ["path"],
                },
            }
        ]

        prompt = f"""
        Given the following phone tree structure:
        {tree_structure}

        Analyze the conversation transcript below and extract the path taken.
        Use the exact labels from the phone tree structure when possible.
        If a new path is explored, use appropriate labels that fit the context of the conversation.
        Always start the path with the 'root' decision point.

        Transcript:
        {transcript}
        """

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                functions=functions,
                function_call={"name": "extract_phone_tree_path"},
            )

            function_call = response.choices[0].message.function_call
            if function_call and function_call.name == "extract_phone_tree_path":
                path_data = json.loads(function_call.arguments)
                path = [(item['decision_point'], item['choice']) for item in path_data['path']]
            else:
                logger.error("Unexpected response format from OpenAI API")
                path = []

        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}")
            path = []

        logger.info(f"Extracted path with {len(path)} elements")
        return path

    def _validate_tree_structure(self, node: TreeNode, visited: Set[TreeNode] = set()) -> bool:

        if node in visited:
            logger.error(f"Loop detected in tree structure at node: {node.content}")
            return False

        visited.add(node)

        for child in node.children.values():
            if not self._validate_tree_structure(child, visited):
                return False

        visited.remove(node)
        return True

    async def validate_tree(self) -> bool:
        is_valid = self._validate_tree_structure(self.root)
        if not is_valid:
            logger.error("Invalid tree structure detected")
        return is_valid









