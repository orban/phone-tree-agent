from typing import List, Dict, Set


class TreeNode:
    def __init__(self, option: str):
        self.option = option
        self.children: Dict[str, TreeNode] = {}
        self.data: Dict = {}
        self.explored_options: Set[str] = set()


class PhoneTree:
    def __init__(self):
        self.root = TreeNode("root")

    def add_path(self, path: List[str], data: Dict):
        current = self.root
        for option in path:
            if option not in current.children:
                current.children[option] = TreeNode(option)
            current = current.children[option]
        current.data = data
        # Only update explored_options for the last node in the path
        current.explored_options.add(path[-1] if path else "root")

    def is_fully_explored(self, path: List[str]) -> bool:
        current = self.root
        for option in path:
            if option not in current.children:
                return False
            current = current.children[option]
        return set(current.data.get("options", [])) == current.explored_options

    def get_unexplored_paths(self) -> List[List[str]]:
        def dfs(node: TreeNode, current_path: List[str]) -> List[List[str]]:
            if not node.data:
                return [current_path]

            paths = []
            for option in node.data.get("options", []):
                if option not in node.explored_options:
                    paths.append(current_path + [option])

            for child in node.children.values():
                paths.extend(dfs(child, current_path + [child.option]))

            return paths

        return dfs(self.root, [])

    def to_dict(self) -> Dict:
        def node_to_dict(node: TreeNode) -> Dict:
            result = {
                "option": node.option,
                "data": node.data,
                "explored_options": list(node.explored_options),
            }
            if node.children:
                result["children"] = {
                    k: node_to_dict(v) for k, v in node.children.items()
                }
            return result

        return node_to_dict(self.root)

    def get_node(self, path: List[str]) -> TreeNode:
        current = self.root
        for option in path:
            if option not in current.children:
                raise ValueError(f"Path {path} does not exist in the tree")
            current = current.children[option]
        return current
