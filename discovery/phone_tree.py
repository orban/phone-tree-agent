from typing import List, Dict


class TreeNode:
    def __init__(self, option: str):
        self.option = option
        self.children: Dict[str, TreeNode] = {}
        self.data: Dict = {}
        self.explored: bool = False


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
        current.explored = True

    def is_explored(self, path: List[str]) -> bool:
        current = self.root
        for option in path:
            if option not in current.children:
                return False
            current = current.children[option]
        return current.explored

    def to_dict(self) -> Dict:
        def node_to_dict(node: TreeNode) -> Dict:
            result = {
                "option": node.option,
                "data": node.data,
                "explored": node.explored,
            }
            if node.children:
                result["children"] = {
                    k: node_to_dict(v) for k, v in node.children.items()
                }
            return result

        return node_to_dict(self.root)

    def get_unexplored_paths(self) -> List[List[str]]:
        def dfs(node: TreeNode, current_path: List[str]) -> List[List[str]]:
            if not node.explored and not node.children:
                return [current_path]

            paths = []
            for option, child in node.children.items():
                paths.extend(dfs(child, current_path + [option]))
            return paths

        return dfs(self.root, [])
