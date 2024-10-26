import json
from loguru import logger
from discovery.phone_tree import PhoneTree
from discovery.phone_tree import TreeNode




class OutputGenerator:
    """
    Generates and updates a JSON file with the discovered phone tree.
    """

    def __init__(self, output_file: str = "phone_tree_progress.json"):
        self.output_file = output_file
        self.tree_structure = {}

    async def update_progress(self, phone_tree: PhoneTree):
        """
        Updates the progress with new data.
        """
        logger.info("Updating progress")

        # Convert the phone tree to a dictionary
        self.tree_structure = await self._to_dict(phone_tree.root)

        # Write updated tree to file
        await self._write_to_file(self.tree_structure)

        logger.info("Progress updated and written to file")

    async def _write_to_file(self, phone_tree_dict: dict):
        """
        Writes the discovered tree to a file.
        """
        try:
            with open(self.output_file, "w") as f:
                json.dump(phone_tree_dict, f, indent=2)
        except IOError as e:
            logger.error(f"Error writing to file: {str(e)}")

    async def generate_mermaid_graph(self, phone_tree: PhoneTree) -> str:
        """
        Generates a Mermaid graph from the discovered tree.
        """

        def traverse(node, parent=None, depth=0):
            lines = []
            option = node.get("option", "root")
            if parent:
                lines.append(f"    {'    ' * depth}{parent} --> {option}")
            if "children" in node:
                for child in node["children"].values():
                    lines.extend(traverse(child, option, depth + 1))
            return lines

        graph_lines = ["graph TD"]
        graph_lines.extend(traverse(await self._to_dict(phone_tree.root)))
        return "\n".join(graph_lines)

    async def generate_json_dataset(self, phone_tree: PhoneTree):
        async def node_to_dict(node, path):
            node_data = {
                "id": "_".join(path) if path else "root",
                "label": node.option,
                "path": path,
                "data": node.data,
                "explored": node.explored,
            }
            if node.children:
                node_data["children"] = [
                    node_to_dict(child, path + [child.option])
                    for child in node.children.values()
                ]
            return node_data

        dataset = node_to_dict(phone_tree.root, [])
        with open("phone_tree_dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)
        logger.info("JSON dataset generated and saved as phone_tree_dataset.json")

    def generate_summary_report(self, phone_tree: PhoneTree):
        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count

        def max_depth(node):
            if not node.children:
                return 1
            return 1 + max(max_depth(child) for child in node.children.values())

        total_nodes = count_nodes(phone_tree.root)
        tree_depth = max_depth(phone_tree.root) - 1  # Subtract 1 to exclude root
        end_nodes = sum(
            1 for node in phone_tree.root.children.values() if not node.children
        )

        summary = f"""
            Phone Tree Summary:
            -------------------
            Total nodes: {total_nodes}
            Tree depth: {tree_depth}
            End nodes: {end_nodes}
            """
        with open("phone_tree_summary.txt", "w") as f:
            f.write(summary)
        logger.info("Summary report generated and saved as phone_tree_summary.txt")

    async def print_tree(self, node: TreeNode, prefix: str = "", is_last: bool = True):
        stack = [(node, prefix, is_last)]
        while stack:
            current_node, current_prefix, current_is_last = stack.pop()
            if isinstance(current_node, dict):
                items = list(current_node.items())
                for index, (key, value) in enumerate(reversed(items)):
                    is_last_item = index == 0
                    print(f"{current_prefix}{'└── ' if current_is_last else '├── '}{key}")
                    stack.append((value, current_prefix + ('    ' if current_is_last else '│   '), is_last_item))
            elif isinstance(current_node, TreeNode):
                print(f"{current_prefix}{'└── ' if current_is_last else '├── '}{current_node.content}")
                children = list(current_node.children.values())
                for index, child in enumerate(reversed(children)):
                    is_last_item = index == 0
                    stack.append((child, current_prefix + ('    ' if current_is_last else '│   '), is_last_item))
                
            

    async def _to_dict(self, node):
        result = {}
        for child in node.children.values():
            result[child.option] = self._to_dict(child)
        return result

    async def print_tree_to_string(self, node: TreeNode, prefix: str = "", is_last: bool = True) -> str:
        result = []
        result.append(f"{prefix}{'└── ' if is_last else '├── '}{node.content}")
        
        child_count = len(node.children)
        for index, (child_key, child) in enumerate(node.children.items()):
            is_last_item = index == child_count - 1
            new_prefix = prefix + ('    ' if is_last else '│   ')
            result.append(await self.print_tree_to_string(child, new_prefix, is_last_item))
        
        return "\n".join(result)
