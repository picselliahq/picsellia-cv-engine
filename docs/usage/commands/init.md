# `pxl-pipeline init`

The `init` command bootstraps a new **pipeline project** (either processing or training) with the standard folder structure, dependencies, and metadata required to run pipelines in Picsellia.

## Usage

```bash
pxl-pipeline init PIPELINE_NAME \
  --type [processing|training] \
  --template TEMPLATE_NAME \
  [OPTIONS]
```

### Arguments

| Argument        | Description                                    | Required |
|-----------------|------------------------------------------------|----------|
| `PIPELINE_NAME` | Name of the pipeline project (and its folder). | ✅        |

---

### Options

| Option           | Description                                                         | Default       |
|------------------|---------------------------------------------------------------------|---------------|
| `--type`         | Pipeline type: `processing` or `training`.                          | ✅ Required   |
| `--template`     | Template name.                         | ✅ Required   |
| `--output-dir`   | Target directory where the pipeline will be created.                | `.` (current) |
| `--use-pyproject`| Generate a `pyproject.toml` for dependency management (via `uv`).   | `True`        |

## Templates

### Processing
- `dataset_version`: process a dataset version with custom logic, inputs, and parameters.

- `datalake`: process data in a datalake (tagging, filtering, etc.).

- `model_version`: process a model version (conversion, compression, etc.).

### Training
- `simple_training`: A minimal, framework-agnostic training pipeline scaffold that you can extend with any ML framework.

## Behavior

### Processing pipelines

- Generate the full pipeline scaffold:

    - `config.toml`
    - `steps.py`
    - `utils/parameters.py`
    - `utils/inputs.py` (for processing pipelines that declare inputs)
    - `.venv/` (with dependencies installed via `uv`)

### Training pipelines

- Prompt for organization and environment if not set via env vars:

```bash
export PICSELLIA_ORGANIZATION=my-org
export PICSELLIA_ENV=STAGING
```

- Prompt for model version:

    - Reuse an existing private/public model version

    - Or create a new one (define model name, version, framework, inference type)

- Save model metadata into config.toml:

```toml
[model_version]
name = "v1"
origin_name = "MyModel"
framework = "ONNX"
inference_type = "OBJECT_DETECTION"
```

## Examples

### Create a dataset processing pipeline

```bash
pxl-pipeline init my_dataset_pipeline \
  --type processing \
  --template dataset_version
```

### Create a training pipeline
```bash
pxl-pipeline init my_training_pipeline \
  --type training \
  --template simple_training
```

👉 During setup, the CLI will prompt:

- Organization name (if not set in env vars)
- Picsellia environment (prod/staging/local)
- Model version reuse/creation

### Project Structure

```
my_pipeline/
├── config.toml
├── pyproject.toml
├── uv.lock
├── Dockerfile

├── pipeline.py
├── steps.py
├── utils/
│   ├── parameters.py
│   └── inputs.py

├── runs/
│   └── run1/
│       └── run_config.toml

└── .venv/
```
