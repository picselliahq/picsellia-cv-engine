# 🧠 How the pxl-pipeline cli works

## 🚀 Pipeline lifecycle

- `init` → generate template
- `test` → runs the pipeline locally in `.venv/`
- `deploy` → builds & pushes Docker image + registers in Picsellia
- `smoke-test` → runs pipeline in a container before deploying

## 📂 Project structure

Here is a typical pipeline folder structure:

```
my_pipeline/
├── config.toml
├── pyproject.toml
├── uv.lock
├── Dockerfile

├── pipeline.py
├── steps.py
├── utils/
│ ├── parameters.py
│ └── inputs.py

├── runs/
│ └── run1/
│ └── run_config.toml

└── .venv/
```


### Key files:

- **`config.toml`**
  Describes the pipeline metadata, entrypoint files, requirements file, and model metadata.
  ➕ This makes pipelines easily portable and shareable.

- **`pyproject.toml` / `uv.lock`**
  Managed by [`uv`](https://github.com/astral-sh/uv) to declare dependencies.
  You don’t need to manually install anything — just run the CLI.

- **`pipeline.py`**
  Entrypoint for the pipeline, used both locally and on Picsellia.

- **`steps.py`**
  Contains `@step`-decorated functions that define the logic of your pipeline.

- **`utils/parameters.py`**
  Contains the parameter class (`TrainingHyperParameters`, `ProcessingParameters`, etc.) used to extract configuration at runtime.

- **`utils/inputs.py`**
  Contains the inputs class (`ProcessingInputs`) that declares which inputs the processing expects. These are synced to the platform on deploy.

- **`.venv/`**
  Created automatically by the CLI when you run `pxl-pipeline test`.

## 🔐 Environment variables

The CLI requires:

```bash
PICSELLIA_API_TOKEN
PICSELLIA_ORGANIZATION_NAME
PICSELLIA_HOST  # optional, defaults to https://app.picsellia.com
```

They are:

- Prompted once during init, test, or deploy
- Saved in: `~/.config/picsellia/.env`
- Automatically loaded on future runs


You can:

- Manually edit that file
- Or override any value in the current terminal session with export VAR=...


## 🧰 Dependency management with uv

Each pipeline uses `uv` as the dependency manager. It handles package resolution and installation via `pyproject.toml`, without needing pip or poetry.

### 📦 What happens during pxl-pipeline test?

When you run:

```bash
pxl-pipeline test my_pipeline
```

The following is automatically done for you:

- `uv lock` resolves all dependencies and generates/updates `uv.lock`
- `uv sync`  installs packages into `.venv/` based on the lock file

You don't need to install or activate anything manually — the CLI ensures the right environment is built.

### ➕ Adding dependencies

To install a PyPI package:

```bash
uv add opencv-python --project my_pipeline
```

To add a Git-based package:

```bash
uv add git+https://github.com/picselliahq/picsellia-cv-engine.git --project my_pipeline
```

This updates the pyproject.toml and uv.lock files inside your pipeline folder.

💡 Tip: the `--project` flag ensures the package is added to the correct pipeline folder.

## 📁 How `runs/` work

Each test run creates a new directory under runs/:

```
├── runs/
│   ├── run1/
│   ├── run2/
│   └── run3/
│       └── run_config.toml
```

Inside each run folder:

- `run_config.toml` stores the parameters used for that run (e.g. `experiment_id`, `model_version_id`, etc.)
- The dataset and model will be downloaded into this folder
- Logs, annotations, and any outputs will be saved here

### Reusing configurations

- If a previous run exists, the CLI will prompt:

```bash
📝 Reuse previous config? experiment_id=... [Y/n]
```

- Choosing Y reuses the last config (but creates a new folder and re-downloads assets).

- Use the flag `--reuse-dir` to reuse the same directory and config, without downloading again.

## Working with pipeline parameters

### ➕ Adding a custom parameter

Each pipeline includes a `utils/parameters.py` file containing a parameter class that extracts and validates values from Picsellia metadata (experiment or processing).

#### 1. Locate your parameters file

```
my_pipeline/
├── utils/
│   └── parameters.py  ← edit this file
```

#### 2. Edit the parameter class
Inside `parameters.py`, you’ll find a class that inherits from:

- `Parameters` (for processing pipelines)

- `HyperParameters` (for training pipelines)

Add your new fields by calling `self.extract_parameter(...)` in the constructor.

```python
from picsellia_cv_engine.core.parameters import Parameters

class ProcessingParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        # Add your custom parameters here 👇
        self.threshold = self.extract_parameter(
            keys=["threshold"],
            expected_type=float,
            default=0.5,
        )

        self.use_filter = self.extract_parameter(
            keys=["use_filter"],
            expected_type=bool,
            default=True,
        )
```

3. Link the class in `config.toml`

Make sure the class is declared in your pipeline’s `config.toml`:

```toml
[execution]
parameters_class = "utils/parameters.py:ProcessingParameters"
```

#### ✅ What you can define

Each parameter can include:

| Field          | Description                                                           |
|----------------|-----------------------------------------------------------------------|
| `keys`         | One or more fallback keys (e.g. `["lr", "learning_rate"]`)            |
| `expected_type`| Type validation (`int`, `float`, `bool`, `str`, `Optional[...]`)      |
| `default`      | Optional default value (or `...` to mark as required)                 |
| `range_value`  | Value bounds: `(min, max)` for numeric parameters                     |


Advanced use cases (enums, optional types, dynamic validation) are documented in the base Parameters class via extract_parameter(...).

## Working with pipeline inputs

Inputs define what resources or values a processing pipeline expects at launch time. Unlike parameters (which are configuration values like thresholds or batch sizes), inputs represent the actual data the pipeline will work with — such as additional dataset versions, model versions, or free-form text and numbers.

When you deploy a pipeline, the CLI automatically syncs your declared inputs to the Picsellia platform. The platform then validates that all required inputs are provided when a user launches the processing.

### 1. Locate your inputs file

```
my_pipeline/
├── utils/
│   └── inputs.py  ← edit this file
```

### 2. Define your inputs

Inside `inputs.py`, you'll find a class that inherits from `PipelineInputs`. Declare each input by calling `self.define_input(...)` in the constructor.

```python
from picsellia.types.enums import ProcessingInputType
from picsellia_pipelines_cli.utils.inputs import PipelineInputs


class ProcessingInputs(PipelineInputs):
    def __init__(self):
        super().__init__()
        self.define_input(
            name="reference_dataset",
            input_type=ProcessingInputType.DATASET_VERSION,
            required=True,
        )
        self.define_input(
            name="prompt",
            input_type=ProcessingInputType.TEXT,
            required=False,
        )
```

### 3. Link the class in `config.toml`

Make sure the inputs class is declared in your pipeline's `config.toml`:

```toml
[execution]
inputs_class = "utils/inputs.py:ProcessingInputs"
```

### Available input types

Each input must specify a `ProcessingInputType` from the Picsellia SDK:

| Type              | Description                                      |
|-------------------|--------------------------------------------------|
| `DATASET_VERSION` | A reference to a dataset version                 |
| `MODEL_VERSION`   | A reference to a model version                   |
| `DATALAKE`        | A reference to a datalake                        |
| `TEXT`            | A free-form text value                           |
| `NUMBER`          | A numeric value                                  |

### Optional constraints

For inputs of type `MODEL_VERSION` or `DATASET_VERSION`, you can optionally enforce constraints on the framework or inference type:

```python
from picsellia.types.enums import Framework, InferenceType, ProcessingInputType

self.define_input(
    name="detection_model",
    input_type=ProcessingInputType.MODEL_VERSION,
    required=True,
    inference_type_constraint=InferenceType.OBJECT_DETECTION,
    framework_constraint=Framework.PYTORCH,
)
```

These constraints are enforced by the platform when a user launches the processing.

### 4. Add inputs to `run_config.toml` for local testing

When testing locally, input values are read from the `[inputs]` section of `run_config.toml`:

```toml
[inputs]
reference_dataset = "01892b88-8dac-7664-96ee-16d05aa599c9"
prompt = "detect all cats"
```

If you run `pxl-pipeline test` and some declared inputs are missing from the run config, the CLI will scaffold them automatically with empty defaults.

### What happens on deploy?

When you run `pxl-pipeline deploy`, the CLI:

1. Reads the inputs class from `config.toml`
2. Extracts the list of declared inputs
3. Compares them against the inputs already registered on the platform
4. Adds new inputs, updates existing ones, and removes inputs that are no longer declared

This keeps the platform definition always in sync with your code.

## ✅ Summary

- Pipelines are self-contained and shareable via config.toml

- Dependencies are isolated and reproducible with uv

- CLI stores runs in runs/, with config and outputs

- Parameters are centralized and easy to extend

- Inputs define what resources and values a processing expects, and are synced to the platform on deploy

- You can deploy to Picsellia with `pxl-pipeline deploy ...`

For template-specific usage, see:

- [Processing - Dataset Version](processing/dataset_version.md)

- [Processing - Datalake](processing/datalake.md)

- [Processing - Model Version](processing/model_version.md)

- [Training - Simple Training](training/simple_training.md)
