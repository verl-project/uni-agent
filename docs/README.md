# Uni-Agent Documentation

This directory contains the Markdown source files for the **Uni-Agent**
documentation, built with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Install Dependencies

From the repository root, install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

## Build the Docs

Generate the static HTML site from the `docs/` directory:

```bash
cd docs
make html
```

After the build completes, open the generated homepage:

```text
_build/html/index.html
```

## Preview Locally

Run `make serve` and open
[http://localhost:8000](http://localhost:8000). MkDocs automatically reloads
the page when documentation files change.

## Add New Content

- Add a Markdown (`.md`) file under `source/`.
- Add the page to `nav` in the repository-level `mkdocs.yml`.
- Shared images live in the repository-level `assets/` directory.