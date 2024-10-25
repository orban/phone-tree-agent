from typing import List, Dict, Optional, Tuple
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class TreeNode:
    def __init__(self, content: str):
        self.content = content
        self.children: Dict[str, TreeNode] = {}
        self.explored = False


class LabelNormalizer:
    def __init__(self):
        self.label_map: Dict[str, str] = {}

    def get_normalized_label(self, label: str) -> str:
        normalized = label.lower().strip()
        if normalized not in self.label_map:
            self.label_map[normalized] = label
        return self.label_map[normalized]


class PhoneTree:
    def __init__(self, openai_client: AsyncOpenAI):
        self.root = TreeNode("root")
        self.current_node = self.root
        self.openai_client = openai_client
        self.label_normalizer = LabelNormalizer()

    async def add_path(self, path: List[Tuple[str, str]]) -> None:
        current_node = self.root
        for decision_point, choice in path:
            normalized_decision = self.label_normalizer.get_normalized_label(decision_point)
            normalized_choice = self.label_normalizer.get_normalized_label(choice)
            
            if normalized_decision not in current_node.children:
                current_node.children[normalized_decision] = TreeNode(normalized_decision)
            current_node = current_node.children[normalized_decision]
            
            if normalized_choice not in current_node.children:
                current_node.children[normalized_choice] = TreeNode(normalized_choice)
            current_node = current_node.children[normalized_choice]
        
        current_node.explored = True

    def get_unexplored_paths(self) -> List[List[str]]:
        def dfs(node: TreeNode, current_path: List[str]) -> List[List[str]]:
            if not node.children or not node.explored:
                return [current_path]
            paths = []
            for label, child in node.children.items():
                paths.extend(dfs(child, current_path + [label]))
            return paths
        
        return dfs(self.root, [])

    async def extract_path(self, transcript: str) -> List[Tuple[str, str]]:
        prompt = f"""
        Analyze the following conversation transcript and extract the path taken.
        Return the path as a list of (decision_point, choice) tuples.
        Use the following format:
        1. (decision_point_1, choice_1)
        2. (decision_point_2, choice_2)
        
        Example:
        1. (is_existing_customer, yes)
        2. (is_emergency, no)
        3. (describe_issue, air_conditioning)
        4. (schedule_appointment, yes)
        5. (end_call, goodbye)
        ...

        Transcript:
        {transcript}
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        
        path_str = response.choices[0].message.content.strip()
        path = []
        for line in path_str.split('\n'):
            if '(' in line and ')' in line:
                decision_point, choice = line.split('(')[1].split(')')[0].split(',')
                path.append((decision_point.strip(), choice.strip()))
        
        return path

    def mark_path_explored(self, path: List[str]) -> None:
        node = self.root
        for content in path:
            if content in node.children:
                node = node.children[content]
            else:
                return
        node.explored = True

    def print_tree(self, node: Optional[TreeNode] = None, indent: str = "", last: bool = True) -> None:
        if node is None:
            node = self.root
        print(indent, end="")
        if last:
            print("└── ", end="")
            indent += "    "
        else:
            print("├── ", end="")
            indent += "│   "
        print(node.content)
        child_count = len(node.children)
        for i, (content, child) in enumerate(node.children.items()):
            self.print_tree(child, indent, i == child_count - 1)
