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

### 🚀 Main workflows
- [What’s a pipeline?](#whats-a-pipeline)
- [Getting started](#getting-started)
- [Create, test, and deploy a pipeline](#create-test-and-deploy-a-pipeline)

### 🧑‍💻 Contributing
- [Local development](#local-development)

---

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

### 1. **Init — generate a pipeline projectInit — generate a pipeline project**

```bash
pxl-pipeline init my_pipeline --type training --template yolov8
```

This generates a ready-to-use pipeline folder (code templates, config, Dockerfile, etc.).

### 2. **Customize — implement steps & parameters**

You’ll typically edit:

- steps.py to implement your processing/training logic

- pipeline.py to orchestrate steps

- utils/parameters.py to define runtime parameters

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
