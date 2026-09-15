"""MkDocs hooks for repository-level documentation assets."""

from pathlib import Path

from mkdocs.structure.files import File, Files

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPOSITORY_ROOT / "assets"


def on_files(files: Files, *, config) -> Files:
    """Include the repository's shared assets in the generated site."""
    known_files = {file.src_uri for file in files}

    for asset in ASSETS_DIR.rglob("*"):
        if not asset.is_file():
            continue

        src_uri = asset.relative_to(REPOSITORY_ROOT).as_posix()
        if src_uri in known_files:
            continue

        files.append(
            File(
                path=src_uri,
                src_dir=str(REPOSITORY_ROOT),
                dest_dir=config.site_dir,
                use_directory_urls=config.use_directory_urls,
            )
        )

    return files
