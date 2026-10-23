#!/usr/bin/env python3
"""Build sidecar filesystem images using a shared, dependency-free entry point."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_RECIPES = Path(__file__).parent / "sample" / "recipes.json"


def load_recipe(name: str, recipe_file: str | Path | None = None) -> dict:
    """Read a recipe and resolve context/dockerfile relative to its manifest."""
    source = Path(recipe_file or DEFAULT_RECIPES).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("recipes"), dict):
        raise ValueError("recipe file must contain schema_version=1 and a recipes object")
    if name not in data["recipes"]:
        raise ValueError(f"unknown recipe {name!r}; choose from: {', '.join(data['recipes'])}")
    if not isinstance(data["recipes"][name], dict):
        raise TypeError("recipe must be a JSON object")
    recipe = dict(data["recipes"][name])
    for key in ("context", "dockerfile", "image", "tag", "platform", "mount_target", "version_arg"):
        if not isinstance(recipe.get(key), str) or not recipe[key]:
            raise ValueError(f"recipe field {key!r} must be a non-empty string")
    for key in ("entrypoint", "probe_argv"):
        if not isinstance(recipe.get(key), list) or not recipe[key]:
            raise ValueError(f"recipe field {key!r} must be a non-empty argv array")
        if any(not isinstance(item, str) or not item for item in recipe[key]):
            raise ValueError(f"recipe field {key!r} contains an invalid argument")
    build_args = recipe.get("build_args")
    if not isinstance(build_args, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in build_args.items()
    ):
        raise TypeError("build_args must be a string-to-string object")
    if recipe["version_arg"] not in build_args or "BASE_IMAGE" not in build_args:
        raise ValueError("build_args must declare BASE_IMAGE and version_arg")
    if recipe["platform"] != "linux/amd64":
        raise ValueError("the current sidecar contract supports only linux/amd64")
    if not recipe["mount_target"].startswith("/") or recipe["mount_target"] == "/":
        raise ValueError("mount_target must be an absolute non-root sandbox path")
    for key in ("context", "dockerfile"):
        recipe[key] = str((source.parent / recipe[key]).resolve())
    if not Path(recipe["context"]).is_dir() or not Path(recipe["dockerfile"]).is_file():
        raise ValueError("recipe context directory or Dockerfile does not exist")
    return recipe


def _public_value(key: str, value: str) -> None:
    # Build arguments and registry URLs become part of image history and logs.
    if re.search(r"TOKEN|PASSWORD|SECRET|CREDENTIAL|(?:^|_)KEY$", key, re.IGNORECASE):
        raise ValueError("credentials must not be supplied as build arguments")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{key} must not contain line breaks")
    if "://" in value:
        url = urlsplit(value)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError(f"{key} must be a public URL without credentials, query or fragment")


def resolve_recipe(args: argparse.Namespace) -> dict:
    recipe = load_recipe(args.recipe, args.recipe_file)
    build_args = dict(recipe["build_args"])
    for item in args.build_arg:
        key, separator, value = item.partition("=")
        if not separator or key not in build_args:
            raise ValueError("--build-arg expects a KEY=VALUE declared in the recipe")
        build_args[key] = value
    if args.base_image is not None:
        build_args["BASE_IMAGE"] = args.base_image
    if args.version is not None:
        build_args[recipe["version_arg"]] = args.version
    pip_index_arg = "INSTALL_PIP_INDEX_URL" if "INSTALL_PIP_INDEX_URL" in build_args else "PIP_INDEX_URL"
    for option, key in ((args.pip_index, pip_index_arg), (args.npm_registry, "NPM_REGISTRY")):
        if option is not None:
            if key not in build_args:
                raise ValueError(f"this recipe does not declare {key}")
            build_args[key] = option
    for key, value in build_args.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError("build argument names must use uppercase letters, digits and underscores")
        _public_value(key, value)
    image = args.image or recipe["image"]
    tag = args.tag or recipe["tag"].format_map(build_args)
    registry = args.registry.rstrip("/")
    if not re.fullmatch(r"(?:[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/)?[a-z0-9]+(?:[._/-][a-z0-9]+)*", image):
        raise ValueError("--image must be a lowercase repository name (optional registry host:port, no tag or digest)")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError("--tag is not a valid image tag")
    if registry and not re.fullmatch(
        r"[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*", registry
    ):
        raise ValueError("--registry expects a host[/namespace] without a URL scheme or credentials")
    if args.push and not registry:
        raise ValueError("--push requires --registry; authenticate separately with docker login")
    image_ref = f"{registry + '/' if registry else ''}{image}:{tag}"
    command = ["docker", "build", "--platform", recipe["platform"], "-f", recipe["dockerfile"], "-t", image_ref]
    if args.no_cache:
        command.append("--no-cache")
    if args.pull:
        command.append("--pull")
    for key, value in build_args.items():
        command.extend(("--build-arg", f"{key}={value}"))
    command.append(recipe["context"])
    recipe.update(build_args=build_args, image=image, tag=tag, image_ref=image_ref, build_command=command)
    return recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", help="recipe name declared in the manifest")
    parser.add_argument("--recipe-file", type=Path, help="JSON manifest; paths are relative to this file")
    parser.add_argument("--image", help="repository name, optionally registry-qualified (no tag or digest)")
    parser.add_argument("--tag", help="image tag; defaults to the tool version")
    parser.add_argument("--registry", default="", help="registry host[/namespace]; does not push by itself")
    parser.add_argument("--push", action="store_true", help="push after a successful build; requires --registry")
    parser.add_argument("--version", help="override the recipe's primary tool version")
    parser.add_argument("--base-image", help="override BASE_IMAGE (a digest may be used)")
    parser.add_argument(
        "--build-arg", action="append", default=[], metavar="KEY=VALUE", help="override a declared build argument"
    )
    parser.add_argument("--pip-index", help="public pip index URL (Python recipe only)")
    parser.add_argument("--npm-registry", help="public npm registry URL (native recipe only)")
    parser.add_argument("--no-cache", action="store_true", help="rebuild every Docker layer")
    parser.add_argument("--pull", action="store_true", help="check for a newer builder base image")
    parser.add_argument("--dry-run", action="store_true", help="print commands without invoking Docker")
    parser.add_argument("--describe", action="store_true", help="print resolved recipe JSON without invoking Docker")
    args = parser.parse_args(argv)
    try:
        recipe = resolve_recipe(args)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    if args.describe:
        print(json.dumps(recipe, indent=2))
        return 0
    commands = [recipe["build_command"]]
    if args.push:
        commands.append(["docker", "push", recipe["image_ref"]])
    if not args.dry_run and not shutil.which("docker"):
        parser.error("docker was not found; install Docker or use --describe/--dry-run")
    try:
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        return error.returncode
    if not args.dry_run:
        print(f"Image ready: {recipe['image_ref']}")
        print(f"Mount target: {recipe['mount_target']}")
        print(f"Entrypoint: {shlex.join(recipe['entrypoint'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
