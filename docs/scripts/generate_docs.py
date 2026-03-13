import os
import shutil
from typing import Any

import yaml

# Directories to include in documentation
SOURCE_DIRS = ["src/picsellia_cv_engine"]

# Directories to exclude from documentation
EXCLUDE_DIRS = [
    "src/picsellia_cv_engine/core/logging",
]

# Specific files to exclude
EXCLUDE_FILES = [
    "enums.py",
    "logger.py",
    "src/picsellia_cv_engine/core/data/coco_file_manager.py",
]

DOCS_DIR = "docs/api"
USAGE_DIR = "docs/usage"
MKDOCS_CONFIG_FILE = "mkdocs.yml"

# Template for Markdown doc pages
MKDOCS_TEMPLATE = """# {title}

::: {module}
    handler: python
    options:
        show_submodules: false
        show_if_no_docstring: true
        show_root_heading: true
"""


def should_exclude(path, filename=None):
    """Check if a path or specific file should be excluded."""
    if any(path.startswith(exclude) for exclude in EXCLUDE_DIRS):
        return True

    if filename:
        full_file_path = os.path.join(path, filename).replace("\\", "/")
        return any(
            full_file_path.endswith(exclude_file) or full_file_path == exclude_file
            for exclude_file in EXCLUDE_FILES
        )

    return False


def generate_markdown():
    os.makedirs(DOCS_DIR, exist_ok=True)
    generated_files = []

    for source_dir in SOURCE_DIRS:
        for root, _, files in os.walk(source_dir):
            if should_exclude(root):
                continue

            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    if should_exclude(root, file):
                        continue

                    module_path = (
                        os.path.join(root, file)
                        .replace("/", ".")
                        .replace("\\", ".")
                        .replace(".py", "")
                    )
                    module_path = module_path.replace("src.", "")
                    module_name = module_path.replace("picsellia_cv_engine.", "")

                    title = module_name

                    relative_path = (
                        root.replace(source_dir, "").strip(os.sep).replace(os.sep, "/")
                    )
                    output_dir = os.path.join(DOCS_DIR, relative_path)
                    os.makedirs(output_dir, exist_ok=True)

                    md_filename = os.path.join(output_dir, f"{file.replace('.py', '.md')}")
                    md_content = MKDOCS_TEMPLATE.format(title=title, module=module_path)

                    with open(md_filename, "w") as md_file:
                        md_file.write(md_content)

                    generated_files.append((relative_path, file.replace(".py", ".md")))
                    print(f"✅ Generated: {md_filename}")

    return generated_files


def _display_name_from_filename(filename: str) -> str:
    name = filename.replace(".md", "").replace("_", " ").replace("-", " ").strip()
    if name.lower() == "index":
        return "Overview"
    return " ".join(w.capitalize() for w in name.split())


def _sorted_md_files(files: list[str]) -> list[str]:
    """Sort md files with index.md first, then alphabetical."""
    md = [f for f in files if f.endswith(".md")]
    md.sort(key=lambda f: (0 if f == "index.md" else 1, f))
    return md


def generate_usage_nav() -> dict[str, list]:
    """
    Auto-generate the 'Usage Guide' navigation from docs/usage.

    Rules:
    - index.md becomes "Overview"
    - index.md appears first in a folder
    - other pages sorted alphabetically
    - directories become nested sections
    """
    if not os.path.isdir(USAGE_DIR):
        return {"Usage Guide": []}

    structure: dict[str, Any] = {}

    for root, dirs, files in os.walk(USAGE_DIR):
        # ignore hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        rel_dir = os.path.relpath(root, "docs").replace("\\", "/")  # e.g. "usage/commands"
        rel_from_usage = os.path.relpath(root, USAGE_DIR).replace("\\", "/")  # e.g. "commands" or "."
        rel_from_usage = "" if rel_from_usage == "." else rel_from_usage

        target = structure
        if rel_from_usage:
            for part in rel_from_usage.split("/"):
                target = target.setdefault(_display_name_from_filename(part), {})

        for f in _sorted_md_files(files):
            # ignore hidden/partials
            if f.startswith(".") or f.startswith("_"):
                continue
            display = _display_name_from_filename(f)
            target[display] = f"{rel_dir}/{f}"  # path relative to docs/

    def build_nav(struct: dict) -> list:
        nav = []
        for key, value in sorted(struct.items(), key=lambda kv: kv[0].lower()):
            if isinstance(value, dict):
                nav.append({key: build_nav(value)})
            else:
                nav.append({key: value})
        return nav

    return {"Usage Guide": build_nav(structure)}


def update_mkdocs_nav(generated_files):
    if not os.path.exists(MKDOCS_CONFIG_FILE):
        print(f"❌ Error: {MKDOCS_CONFIG_FILE} not found!")
        return

    with open(MKDOCS_CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    nav = config.get("nav", [])

    # Remove previous API entries
    nav = [
        item
        for item in nav
        if not isinstance(item, dict) or "API Reference" not in item
    ]

    # Remove previous Usage Guide entries (so it's fully regenerated)
    nav = [
        item
        for item in nav
        if not isinstance(item, dict) or "Usage Guide" not in item
    ]

    # Re-inject Usage Guide (auto)
    usage_section = generate_usage_nav()
    nav.append(usage_section)

    # Re-inject API Reference (auto)
    api_section: dict[str, list] = {"API Reference": [{"Overview": "api/index.md"}]}
    structure: dict[str, Any] = {}

    for path, file in generated_files:
        sections = path.split("/") if path else []
        target = structure

        for section in sections:
            target = target.setdefault(section, {})

        display_name = file.replace(".md", "").replace("_", " ")
        target[display_name] = f"api/{path}/{file}" if path else f"api/{file}"

    def build_nav(struct: dict) -> list:
        out = []
        for key, value in sorted(struct.items()):
            if isinstance(value, dict):
                out.append({key: build_nav(value)})
            else:
                out.append({key: value})
        return out

    api_section["API Reference"].extend(build_nav(structure))
    nav.append(api_section)

    config["nav"] = nav

    with open(MKDOCS_CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("✅ Updated mkdocs.yml with new Usage Guide + API Reference navigation.")


def clean_docs_dir():
    """Remove all files and folders in DOCS_DIR except index.md."""
    if not os.path.exists(DOCS_DIR):
        return

    for item in os.listdir(DOCS_DIR):
        item_path = os.path.join(DOCS_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)


if __name__ == "__main__":
    clean_docs_dir()
    generated_files = generate_markdown()
    update_mkdocs_nav(generated_files)
