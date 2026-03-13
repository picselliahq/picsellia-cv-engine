# Dataset Version Creation Pipeline


his guide explains how to create, customize, test, and deploy a dataset processing pipeline using the `dataset_version_creation` template with the Picsellia Pipelines CLI.

This template is designed for pipelines that produce a new dataset version by modifying images and/or COCO annotations, for example:

- applying augmentations
- filtering or remapping classes
- resizing images
- cleaning or restructuring annotations

---

## Table of Contents

### Getting started
- [Overview](#overview)
- [Login to Picsellia](#login-to-picsellia)
- [Initialize the pipeline](#1-initialize-the-pipeline)
- [Understand the project structure](#2-understand-the-project-structure)

### Configure before running
- [Configure execution with `run_config.toml`](#3-configure-execution-with-run_configtoml)
  - [Run config example](#example)
  - [Required fields](#required-fields)

### Implement your logic
- [Customize the processing logic](#4-customize-the-processing-logic)
  - [Processing step (`steps.py`)](#stepspy)
  - [Image & annotation logic (`utils/processing.py`)](#utilsprocessingpy)
- [Define pipeline parameters](#5-define-pipeline-parameters)
- [Manage dependencies](#6-manage-dependencies)

### Run and validate
- [Run locally (test)](#7-run-locally-test)
- [Validate in Docker (smoke test)](#8-validate-in-docker-smoke-test)

### Publish and execute
- [Deploy to Picsellia](#9-deploy-to-picsellia)
- [Launch a real job (optional)](#10-launch-a-real-job-optional)


## Overview

The `dataset_version_creation` template generates a processing pipeline that:

- takes an existing dataset version as input

- processes images and annotations

- uploads a new dataset version to Picsellia

Execution is config-first: all runs rely on a run_config.toml file, reused across local, Docker, and Picsellia executions.

## Login to Picsellia

Before running any command, make sure you are logged in:

```bash
pxl-pipeline login
```

If you are already logged in, you can skip this step.

## **1. Initialize the pipeline**

Create a new processing pipeline:

```sh
pxl-pipeline init my_custom_pipeline --type processing --template dataset_version_creation
```

⚠️ Naming recommendation

Use underscores (_) in pipeline names instead of dashes (-) to avoid issues with directories, imports, and Docker paths.

## 2. Understand the project structure

The command above generates the following structure:

- `pipeline.py` — single entrypoint

- `steps.py` — default processing step

- `utils/processing.py` — image & annotation logic

- `utils/parameters.py` — pipeline parameters

- `runs/run_config.toml` — run configuration template

- `config.toml`, `Dockerfile`, `.dockerignore`, dependencies

Example structure:

```
my_custom_pipeline/
├── pipeline.py
├── steps.py
├── utils/
│   ├── processing.py
│   └── parameters.py
├── config.toml
├── Dockerfile
├── .dockerignore
├── runs/
│   └── run_config.toml
└── pyproject.toml
```

## 3. Configure execution with `run_config.toml`

The run config file defines everything required to execute the pipeline.

📍 Location:

```bash
my_custom_pipeline/runs/run_config.toml
```

### Example

```toml
override_outputs = true

[job]
type = "DATASET_VERSION_CREATION"

[input.dataset_version]
id = ""

[output.dataset_version]
name = "test_my_custom_pipeline"

[parameters]
datalake = "default"
data_tag = "processed"
```

### Required fields

Before running, you must set:

```toml
[input.dataset_version]
id = "DATASET_VERSION_ID"
```

💡 The same run config file is used for:

- local testing

- Docker smoke tests

- real execution on Picsellia

## 4. Customize the processing logic

### `steps.py`

The default process step:

- loads parameters from the execution context

- initializes a new output dataset

- calls `process_images()` from `utils/processing.py`

- returns the processed dataset

In most cases, you do not need to change this step.


### `utils/processing.py`

This is where you implement:

- image transformations

- annotation updates

- dataset filtering or remapping logic

Your function must:

- save processed images to `output_images_dir`

- fully populate `output_coco["images"]` and `output_coco["annotations"]`

## 5. Define pipeline parameters

Define custom parameters in `utils/parameters.py`:

```python
class ProcessingParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)
        self.blur = self.extract_parameter(["blur"], expected_type=bool, default=False)
```

These parameters:

- are read from `run_config.toml`

- are injected automatically at runtime

- are logged and tracked by Picsellia

### Accessing parameters in your code

At runtime, parameters are accessible via the active context:

```python
context = Pipeline.get_active_context()
threshold = context.parameters.threshold
```

Parameters values come from `run_config.toml`.

## 6. Manage dependencies

Dependencies are declared in `pyproject.toml`.

Add packages with:

```bash
uv add opencv-python --project my_custom_pipeline
uv add git+https://github.com/picselliahq/picsellia-cv-engine.git --project my_custom_pipeline
```

## 7. Run locally (test)

Run the pipeline locally using the run config file:

```sh
pxl-pipeline test my_custom_pipeline \
  --run-config-file my_custom_pipeline/runs/run_config.toml
```

This:

- executes the pipeline in the local virtual environment
- uses real Picsellia objects
- uploads the output dataset version

To reuse the same folder and avoid re-downloading assets or model files, use:

```bash
pxl-pipeline test my_custom_pipeline --reuse-dir \
  --run-config-file my_custom_pipeline/runs/run_config.toml
```

## 8. Validate in Docker (smoke test)

Before deploying, validate the pipeline inside Docker:

```bash
pxl-pipeline smoke-test my_custom_pipeline \
  --run-config-file my_custom_pipeline/runs/run_config.toml
```

This ensures:

- the Dockerfile is correct

- dependencies are installed properly

- runtime paths and imports work as expected

## 9. Deploy to Picsellia

Publish the pipeline:

```bash
pxl-pipeline deploy my_custom_pipeline
```

This builds and pushes the Docker image and registers the pipeline in Picsellia.

## 10. Launch a real job (optional)

Trigger a real execution on Picsellia:

```bash
pxl-pipeline launch my_custom_pipeline \
  --run-config-file my_custom_pipeline/runs/run_config.toml
```

This behaves exactly like launching a job from the UI.