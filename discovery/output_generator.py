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

    async def update_progress(self, phone_tree: PhoneTree):
        """
        Updates the progress with new data.
        """
        logger.info("Updating progress")

        # Write updated tree to file
        await self._write_to_file(phone_tree.to_dict())

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

    def generate_mermaid_graph(self, phone_tree: PhoneTree) -> str:
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
        graph_lines.extend(traverse(phone_tree.to_dict()))
        return "\n".join(graph_lines)

    def generate_json_dataset(self, phone_tree: PhoneTree):
        def node_to_dict(node, path):
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

    def print_tree(self, node: TreeNode, prefix="", is_last=True):
        print(prefix + ("└── " if is_last else "├── ") + str(node.option))
        child_count = len(node.children)
        for i, (key, child) in enumerate(node.children.items()):
            is_last_child = i == child_count - 1
            self.print_tree(
                child, prefix + ("    " if is_last else "│   "), is_last_child
            )
