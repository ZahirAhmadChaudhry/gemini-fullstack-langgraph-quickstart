import os
import sys

def write_tree(root, buffer, prefix=""):
    files = sorted(os.listdir(root))
    for idx, name in enumerate(files):
        path = os.path.join(root, name)
        connector = "└── " if idx == len(files) - 1 else "├── "
        buffer.append(f"{prefix}{connector}{name}")
        if os.path.isdir(path):
            extension = "    " if idx == len(files) - 1 else "│   "
            write_tree(path, buffer, prefix + extension)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    root_name = os.path.basename(os.path.abspath(folder)) or folder
    lines = [f"{root_name}/"]
    write_tree(folder, lines)
    tree_md = "\n".join(lines)
    with open("directory_structure.md", "w", encoding="utf-8") as f:
        f.write(tree_md)
    print(tree_md)
    print("\nSaved as directory_structure.md")
