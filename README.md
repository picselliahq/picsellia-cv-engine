# Picsellia CV Engine

**Picsellia CV Engine** is a modular Python engine used to build **training** and **processing** computer vision pipelines, fully integrated with the Picsellia platform.

👉 **Full documentation (recommended starting point):**  
**https://picselliahq.github.io/picsellia-cv-engine/**

It provides the building blocks to write clean, reusable pipelines:
- a **pipeline** abstraction (`@pipeline`)
- composable **steps** (`@step`)
- shared **contexts**, **parameters**, and runtime helpers
- logging and execution utilities aligned with Picsellia jobs

> If you want to *generate / test / dockerize / deploy* a pipeline, you’ll typically use the **Picsellia Pipelines CLI** on top of this engine.

---

## Table of contents

### Main workflows
- [What’s a pipeline?](#whats-a-pipeline)
- [Getting started](#getting-started)
- [Create, test, and deploy a pipeline](#create-test-and-deploy-a-pipeline)

### Ecosystem
- [Picsellia CV ecosystem](#-picsellia-cv-ecosystem)

### Contributing
- [Local development](#local-development)
- [Documentation](#documentation)
  - [Where to edit documentation](#where-to-edit-documentation)
  - [Regenerating the documentation](#regenerating-the-documentation)
  - [Publishing documentation](#publishing-github-pages)
- [Releasing the package (PyPI)](#releasing-the-package-pypi)

---

## 🔗 Picsellia CV ecosystem

This repository is part of the Picsellia Computer Vision ecosystem.
Each component has a clear responsibility:

- **Picsellia CV Engine** (this repo)

    → The core Python engine used to build training and processing pipelines
    (pipeline abstraction, steps, contexts, logging, execution)


- **Picsellia CV Pipelines**

    → A collection of ready-to-use pipeline implementations built on top of the engine
    
    👉 https://github.com/picselliahq/picsellia-cv-pipelines


- **Picsellia Pipelines CLI**

    → The developer-facing CLI to generate, test, dockerize and deploy pipelines
    
    👉 https://github.com/picselliahq/picsellia-pipelines-cli

## What’s a pipeline?

A pipeline is simply:
- a Python entrypoint (`pipeline.py`)
- calling one or more **steps** (`steps.py`)
- configured via `run_config.toml`
- runnable locally or on Picsellia infrastructure

A **step** is a small, focused function decorated with `@step` (e.g. data preparation, training, evaluation, export).
A **pipeline** is a function decorated with `@pipeline` that orchestrates step calls.

---

## Getting Started

Install from PyPI:

- With uv:

```bash
uv add picsellia-cv-engine
uv add picsellia-pipelines-cli
```

 - With pip:

```bash
pip install picsellia-cv-engine
pip install picsellia-pipelines-cli
```

## Create, test, and deploy a pipeline

The recommended workflow is to use the Picsellia Pipelines CLI (it scaffolds templates and manages the lifecycle).

### 1. **Init — generate a pipeline project**

```bash
pxl-pipeline init my_pipeline --type training --template yolov8
```

This generates a ready-to-use pipeline folder (code templates, config, Dockerfile, etc.).

### 2. **Customize — implement steps & parameters**

You’ll typically edit:

- `steps.py` to implement your processing/training logic

- `pipeline.py` to orchestrate steps

- `utils/parameters.py` to define runtime parameters

In most cases, if you chose the right template, you only need to adapt the existing default step.

### 3. **Test — run locally**

```bash
pxl-pipeline test my_pipeline
```

### 4. **Deploy — publish to Picsellia**

```bash
pxl-pipeline deploy my_pipeline
```

🔎 Want real examples?
Explore the [pipeline usage templates](https://picselliahq.github.io/picsellia-cv-engine/usage/) for training and processing workflows.

## Local Development

To contribute or explore the code:

### 1. Clone the repository

```bash
git clone https://github.com/picselliahq/picsellia-cv-engine.git
cd picsellia-cv-engine
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the documentation locally

```bash
uv run mkdocs serve -a 127.0.0.1:8080
```
Then open http://127.0.0.1:8080 in your browser.

## Documentation

The documentation is built with MkDocs and published on GitHub Pages.

### Where to edit documentation

There are two kinds of docs in this repository:

- **API Reference (auto-generated from docstrings)**

  Generated Markdown lives under `docs/api/` (do not edit these files manually).
 

- **Editorial / Usage docs (hand-written)**

  You can edit or add pages under:

  - `docs/usage/`
  - `docs/index.md`
  - `docs/installation.md`
  - and more generally any `.md` file under `docs/`

### Regenerating the documentation

Whenever you:

- add or modify files under `docs/` (including `docs/usage/`)
- add or modify Python code / docstrings that should appear in the API reference

you must regenerate the documentation from the repository root:

```bash
python docs/scripts/generate_docs.py
```

Then you can serve the docs locally:

```bash
uv run mkdocs serve -a 127.0.0.1:8080
```

Open http://127.0.0.1:8080 -> in your browser.

### Publishing (GitHub Pages)

Documentation is automatically deployed to GitHub Pages on every push to `main`.

CI workflow: **Deploy MkDocs to GitHub Pages**

- runs `docs/scripts/generate_docs.py `
- deploys with `mkdocs gh-deploy`

So once your changes are merged into `main`, the website updates automatically.

### Releasing the package (PyPI)

Releases are published to PyPI via CI when a version tag is pushed.

CI workflow: **Publish to PyPI**

- triggers on tags matching `v*.*.*` (e.g. `v0.1.0`)
- builds the package with `uv build` 
- publishes with `uv publish` using the `PYPI_TOKEN` secret

### Publish a new version

1. Update the version (following the project’s versioning rules).

2. Create and push a tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

This will trigger the PyPI publishing workflow automatically.

--------------------------------

Made with ❤️ by the Picsellia team.