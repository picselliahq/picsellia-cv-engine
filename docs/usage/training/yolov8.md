# YOLOv8 Training Pipeline

This page describes how to create, configure, run, and deploy a **training pipeline** using the `yolov8` template with the **Picsellia Pipelines CLI**.

This template is designed to train YOLOv8 models while integrating tightly with Picsellia for:
- dataset handling
- experiment tracking
- model versioning
- artifact logging

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
  - [Dataset inputs (train / val / test)](#dataset-inputs-train--val--test)
  - [Required fields](#required-fields)

### Implement your logic
- [Customize the training logic](#4-customize-the-training-logic)
  - [Training step (`steps.py`)](#training-step-stepspy)
- [Define training hyperparameters](#5-define-training-hyperparameters)
- [Manage dependencies](#6-manage-dependencies)

### Run and validate
- [Run locally (test)](#7-run-locally-test)
- [Validate in Docker (smoke test)](#8-validate-in-docker-smoke-test)

### Publish and execute
- [Deploy to Picsellia](#9-deploy-to-picsellia)
- [Launch a real training job (optional)](#10-launch-a-real-training-job-optional)

---

## Overview

The `yolov8` template generates a **training pipeline** that:

- trains a YOLOv8 model on one or more datasets
- logs metrics, checkpoints, and artifacts to Picsellia
- updates or creates a model version
- tracks experiments consistently across environments

Execution is **config-first** and driven by a `run_config.toml` file reused across local, Docker, and Picsellia executions.

---

## Login to Picsellia

Before running any command, make sure you are logged in:

```bash
pxl-pipeline login
```

If you are already logged in, you can skip this step.

## 1. Initialize the pipeline

Create a new YOLOv8 training pipeline:

```bash
pxl-pipeline init my_yolov8_training --type training --template yolov8
```

⚠️ Naming recommendation
Use underscores (_) in pipeline names. Avoid dashes (-) to prevent issues with directories, imports, and Docker paths.

During initialization, you will be prompted to:

- select or create a model version

- associate the pipeline with that model version

## 2. Understand the project structure

The initialization command generates the following structure:

```
my_yolov8_training/
├── pipeline.py                 # Entrypoint
├── steps.py                    # Training step
├── utils/
│   └── parameters.py           # Training hyperparameters
├── config.toml                 # Pipeline metadata & execution info
├── Dockerfile
├── .dockerignore
├── runs/
│   └── run_config.toml         # Execution configuration (template)
└── pyproject.toml
```

## 3. Configure execution with run_config.toml

All executions rely on a run configuration file, generated as a template at:

```bash
my_yolov8_training/runs/run_config.toml
```

This file defines:

- which datasets are attached to the experiment

- which model version is trained

- which hyperparameters are used

- how the experiment is created on Picsellia

### Run config example

```toml
override_outputs = true

[job]
type = "TRAINING"

[hyperparameters]
epochs = 50
batch_size = 8
image_size = 640
device = "cuda:0"

[input.train_dataset_version]
id = "TRAIN_DATASET_VERSION_ID"

# Optional
[input.val_dataset_version]
id = "VAL_DATASET_VERSION_ID"

# Optional
[input.test_dataset_version]
id = "TEST_DATASET_VERSION_ID"

[input.model_version]
id = "MODEL_VERSION_ID"

[output.experiment]
name = "yolov8_exp1"
project_name = "my_project"
```

### Dataset inputs (train / val / test)

Training datasets are attached to the output experiment using aliases:

- train_dataset_version → required

- val_dataset_version → optional but recommended

- test_dataset_version → optional

Each dataset is attached to the experiment with its corresponding role (train, val, test).

At minimum, you must provide:

```toml
[input.train_dataset_version]
id = "TRAIN_DATASET_VERSION_ID"
```

### Required fields

Before running, you must set:

```toml
[input.train_dataset_version]
id = "TRAIN_DATASET_VERSION_ID"

[input.model_version]
id = "MODEL_VERSION_ID"

[output.experiment]
name = "experiment_name"
project_name = "project_name"
```

Hyperparameters are optional and default to the values defined in utils/parameters.py.

💡 The same run config file is reused for:

- local testing

- Docker smoke tests

- real training jobs on Picsellia

## 4. Customize the training logic

### Training step (`steps.py`)

The default training step:

- retrieves datasets, model, and hyperparameters from the execution context

- instantiates the YOLOv8 model

- launches training

- logs metrics and artifacts to Picsellia

In most cases, you do not need to change the pipeline structure.

You may customize:

- preprocessing logic

- training arguments

- checkpoint saving

- model export formats

. Define training hyperparameters

Training hyperparameters are defined in `utils/parameters.py`:

```python
class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data):
        super().__init__(log_data=log_data)
        self.epochs = self.extract_parameter(["epochs"], int, default=50)
        self.batch_size = self.extract_parameter(["batch_size"], int, default=8)
        self.image_size = self.extract_parameter(["image_size"], int, default=640)
```

### Accessing hyperparameters in your code

At runtime, hyperparameters are accessible via the active execution context:

```python
from picsellia_cv_engine.decorators.pipeline_decorator import Pipeline

context = Pipeline.get_active_context()
epochs = context.hyperparameters.epochs
batch_size = context.hyperparameters.batch_size
```

Hyperparameter values come from `run_config.toml`.

⚠️ Make sure your hyperparameter class stays in sync with the model version’s expected configuration.

## 6. Manage dependencies

Dependencies are declared in `pyproject.toml`.

Add packages with:

```bash
uv add albumentations --project my_yolov8_training
```

To add Git-based dependencies:

```bash
uv add git+https://github.com/picselliahq/picsellia-cv-engine.git --project my_yolov8_training
```

Dependencies are automatically resolved locally and inside Docker.

## 7. Run locally (test)

Run the training pipeline locally using the run config file:

```bash
pxl-pipeline test my_yolov8_training \
  --run-config-file my_yolov8_training/runs/run_config.toml
```

This:

- runs the training locally in the virtual environment

- logs metrics and artifacts to Picsellia

- attaches results to the specified experiment

## 8. Validate in Docker (smoke test)

Before deployment, validate the pipeline inside Docker:

```bash
pxl-pipeline smoke-test my_yolov8_training \
  --run-config-file my_yolov8_training/runs/run_config.toml
```

This ensures:

- the Docker image builds correctly

- dependencies and CUDA runtime are available

- training runs correctly in the container

## 9. Deploy to Picsellia

Publish the training pipeline:

```bash
pxl-pipeline deploy my_yolov8_training
```

This will:

- build and push the Docker image

- register or update the training pipeline in Picsellia

- associate it with the selected model version

## 10. Launch a real training job (optional)

Trigger a real training job on Picsellia:

```bash
pxl-pipeline launch my_yolov8_training \
  --run-config-file my_yolov8_training/runs/run_config.toml
```

This behaves exactly like launching a training experiment from the UI.