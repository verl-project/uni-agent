import os
import re

pr_title = os.environ.get("PR_TITLE", "").strip()

allowed_modules = [
    "core",
    "interaction",
    "model",
    "env",
    "tools",
    "deployment",
    "reward",
    "dashboard",
    "docs",
    "examples",
    "data",
    "train",
    "ci",
    "build",
    "deps",
    "misc",
]
allowed_types = ["feat", "fix", "refactor", "chore", "test"]


progress_match = re.match(r"^\[\d/[\dNn]\]\s*(.+)$", pr_title, re.IGNORECASE)
if progress_match:
    pr_title = progress_match.group(1).strip()


breaking_match = re.match(r"^\[BREAKING\]\s*(.+)$", pr_title, re.IGNORECASE)
if breaking_match:
    core_pr_title = breaking_match.group(1).strip()
    is_breaking = True
else:
    core_pr_title = pr_title
    is_breaking = False


re_modules_pattern = re.compile(r"^\[([a-z_,\s]+)\]", re.IGNORECASE)
re_modules = re_modules_pattern.match(core_pr_title)
if not re_modules:
    print(f"Invalid PR title: '{pr_title}'")
    print("Expected format: [BREAKING][module] type: description")
    print(f"Allowed modules: {', '.join(allowed_modules)}")
    raise Exception("Invalid PR title")

modules = re.findall(r"[a-z_]+", re_modules.group(1).lower())
if not modules:
    print(f"Invalid PR title: '{pr_title}'")
    print("At least one module must be specified")
    raise Exception("Invalid PR title")

if not all(module in allowed_modules for module in modules):
    invalid_modules = [module for module in modules if module not in allowed_modules]
    print(f"Invalid modules: {', '.join(invalid_modules)}")
    print(f"Allowed modules: {', '.join(allowed_modules)}")
    raise Exception("Invalid PR title")

types_pattern = "|".join(re.escape(t) for t in allowed_types)
re_types_pattern = re.compile(rf"^\[[a-z_,\s]+\]\s+({types_pattern}):\s+.+$", re.IGNORECASE)
match = re_types_pattern.match(core_pr_title)
if not match:
    print(f"Invalid PR title: '{pr_title}'")
    print("Expected format: [BREAKING][module] type: description")
    print(f"Allowed types: {', '.join(allowed_types)}")
    raise Exception("Invalid PR title")

change_type = match.group(1).lower()
breaking_info = " (BREAKING CHANGE)" if is_breaking else ""
print(f"PR title is valid: {pr_title}, modules: {modules}, type: {change_type}{breaking_info}")
