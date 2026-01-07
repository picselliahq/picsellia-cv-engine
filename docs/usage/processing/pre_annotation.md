# Pre-Annotation Pipeline

This page describes how to create, configure, run, and deploy a **pre-annotation processing pipeline** using the `pre_annotation` template with the **Picsellia Pipelines CLI**.

Pre-annotation pipelines automatically annotate a dataset by applying an existing model (e.g. YOLOv8, GroundingDINO), producing a new dataset version with generated annotations.

---

## Table of Contents

### Getting started
- [Overview](#overview)
- [Login to Picsellia](#login-to-picsellia)
- [Initialize the pipeline](#1-initialize-the-pipeline)
- [Understand the project structure](#2-understand-the-project-structure)

### Configure before running
- [Configure execution with `run_config.toml`](#3-configure-execution-with-run_configtoml)
  - [Run config example](#run-config-example)
  - [Required fields](#required-fields)

### Implement your logic
- [Customize the pre-annotation logic](#4-customize-the-pre-annotation-logic)
  - [Processing step (`steps.py`)](#processing-step-stepspy)
  - [Model inference & annotation logic (`utils/processing.py`)](#model-inference--annotation-logic-utilsprocessingpy)
- [Define pipeline parameters](#5-define-pipeline-parameters)
- [Manage dependencies](#6-manage-dependencies)

### Run and validate
- [Run locally (test)](#7-run-locally-test)
- [Validate in Docker (smoke test)](#8-validate-in-docker-smoke-test)

### Publish and execute
- [Deploy to Picsellia](#9-deploy-to-picsellia)
- [Launch a real job (optional)](#10-launch-a-real-job-optional)

---

## Overview

The `pre_annotation` template generates a **processing pipeline** that:

- takes an existing dataset version as input
- applies a trained model to images
- generates annotations automatically (boxes, masks, keypoints, etc.)
- uploads a new annotated dataset version to Picsellia

Like all processing pipelines, execution is **config-first** and driven by a `run_config.toml` file reused across environments.

## Login to Picsellia

Before running any command, make sure you are logged in:

```bash
pxl-pipeline login
```

If you are already logged in, you can skip this step.

## 1. Initialize the pipeline

Create a new pre-annotation pipeline:

```bash
pxl-pipeline init my_preannotation_pipeline --type processing --template pre_annotation
```

⚠️ Naming recommendation

Use underscores (_) in pipeline names. Avoid dashes (-) to prevent issues with directories, imports, and Docker paths.

## 2. Understand the project structure

The initialization command generates the following structure:

```
my_preannotation_pipeline/
├── pipeline.py                 # Entrypoint
├── steps.py                    # Pipeline steps
├── utils/
│   ├── processing.py           # Model inference & COCO generation
│   └── parameters.py           # Pipeline parameters
├── config.toml                 # Pipeline metadata & execution info
├── Dockerfile
├── .dockerignore
├── runs/
│   └── run_config.toml         # Execution configuration (template)
└── pyproject.toml
```

## 3. Configure execution with `run_config.toml`

All executions rely on a run configuration file, generated as a template at:

```bash
my_preannotation_pipeline/runs/run_config.toml
```

### Run config example

```toml
override_outputs = true

[job]
type = "PRE_ANNOTATION"

[input.dataset_version]
id = ""

[input.model_version]
id = ""

[output.dataset_version]
name = "preannotated_my_dataset"

[parameters]
threshold = 0.1
```

### Required fields

Before running, you must set:

```toml
[input.dataset_version]
id = "DATASET_VERSION_ID"

[input.model_version]
id = "MODEL_VERSION_ID"
```

💡 The same run config file is reused for:

- local testing

- Docker smoke tests

- real execution on Picsellia

## 4. Customize the pre-annotation logic

### Processing step (`steps.py`)

The default process step:

- retrieves the model and dataset from the execution context
- loads parameters
- delegates inference and annotation logic to `utils/processing.py`
- returns a fully annotated dataset

In most cases, you do not need to change the pipeline structure.

### Model inference & annotation logic (`utils/processing.py`)

This is where you implement:

- model inference of images
- bounding box / mask / keypoint generation
- confidence filtering
- formatting annotations into COCO

You can:

- swap the model backend (YOLOv8, GroundingDINO, custom model)
- post-process predictions 
- remap or filter classes

## 5. Define pipeline parameters

Define custom parameters in `utils/parameters.py`:

```python
class ProcessingParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)
        self.threshold = self.extract_parameter(
            ["threshold"],
            expected_type=float,
            default=0.1,
        )
```

These parameters:

- are read from `run_config.toml`

- are injected automatically at runtime

- are logged and tracked by Picsellia

### Accessing parameters in your code

At runtime, parameters are accessible via the active context:

```bash
context = Pipeline.get_active_context()
threshold = context.parameters.threshold
```

Parameters values come from `run_config.toml`.

## 6. Manage dependencies

Dependencies are declared in `pyproject.toml`.

Add packages with:

```bash
uv add opencv-python --project my_preannotation_pipeline
uv add ultralytics --project my_preannotation_pipeline
```

Dependencies are automatically resolved locally and inside Docker.

## 7. Run locally (test)

Run the pipeline locally using the run config file:

```bash
pxl-pipeline test my_preannotation_pipeline \
  --run-config-file my_preannotation_pipeline/runs/run_config.toml
```

This:

- runs the pipeline in the local virtual environment
- loads real datasets and models from Picsellia
- uploads the annotated dataset version

To reuse the same folder and avoid re-downloading assets or model files:

```bash
pxl-pipeline test my_preannotation_pipeline --reuse-dir \
  --run-config-file my_preannotation_pipeline/runs/run_config.toml
```

## 8. Validate in Docker (smoke test)

Before deployment, validate the pipeline inside Docker:

```bash
pxl-pipeline smoke-test my_preannotation_pipeline \
  --run-config-file my_preannotation_pipeline/runs/run_config.toml
```

This ensures:

- the Docker image builds correctly
- dependencies and models are available
- inference works in the container runtime

## 9. Deploy to Picsellia

Publish the pipeline:

```bash
pxl-pipeline deploy my_preannotation_pipeline
```

This builds and pushes the Docker image and registers the pipeline in Picsellia under Processings → Dataset.

## 10. Launch a real job (optional)

Trigger a real execution on Picsellia:

```bash
pxl-pipeline launch my_preannotation_pipeline \
  --run-config-file my_preannotation_pipeline/runs/run_config.toml
```

This behaves exactly like launching a pre-annotation job from the UI.