# Dataset Version Processing Pipeline

This guide explains how to create, customize, test, and deploy a processing pipeline that runs on a **Dataset Version** using the `pxl-pipeline` cli with the `dataset_version` template.

These pipelines receive a dataset version as their target and can apply transformations, filters, or any custom logic to the data.

---

## 1. Initialize your pipeline

```sh
pxl-pipeline init my_dataset_pipeline --type processing --template dataset_version
```

This generates a pipeline folder with standard files. See [project structure](../cli_overview.md#project-structure) for details.

## 2. Customize your pipeline logic

### `steps.py`

Contains the `process()` step where your core logic lives. The context gives you access to the target dataset version, parameters, and any inputs you've declared.

```python
@step
def process():
    context: PicselliaDatasetProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters
    dataset_version = context.target

    # If you want to process only selected assets:
    asset_ids_to_process = context.asset_ids

    # Your logic goes here ...
```

### `utils/parameters.py`

Define custom parameters using a class that inherits from `Parameters`:

```python
class ProcessingParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)
        self.example_parameter = self.extract_parameter(
            ["example_parameter"], expected_type=str, default="default"
        )
```

See [Working with pipeline parameters](../cli_overview.md#working-with-pipeline-parameters) for more.

### `utils/inputs.py`

Define the inputs your processing expects. Inputs are registered on the Picsellia platform when you deploy and are validated at launch time.

```python
from picsellia.types.enums import ProcessingInputType
from picsellia_pipelines_cli.utils.inputs import PipelineInputs


class ProcessingInputs(PipelineInputs):
    def __init__(self):
        super().__init__()
        self.define_input(
            name="example_input",
            input_type=ProcessingInputType.TEXT,
            required=True,
        )
```

See [Working with pipeline inputs](../cli_overview.md#working-with-pipeline-inputs) for the full guide.

## 3. Configure `run_config.toml` for local testing

When you run `pxl-pipeline init`, a `run_config.toml` is generated. It contains the target, inputs, and parameters needed to run locally:

```toml
override_outputs = true

target_id = ""

[job]
type = "DATASET_VERSION_CREATION"

[inputs]
example_input = "example_value"

[parameters]
example_parameter = "default"
```

Fill in the `target_id` with the UUID of the dataset version you want to process.

## 4. Manage dependencies with `uv`

```bash
uv add opencv-python --project my_dataset_pipeline
```

Dependencies are declared in `pyproject.toml`. See [dependency management with uv](../cli_overview.md#dependency-management-with-uv).

## 5. Test your pipeline locally

```sh
pxl-pipeline test my_dataset_pipeline
```

This will:

- Prompt for the target dataset version if not set in the run config
- Scaffold any missing inputs with empty defaults
- Run the pipeline via `pipeline.py --mode local`
- Save everything under `runs/runX/`

To reuse the same folder and avoid re-downloading assets:

```bash
pxl-pipeline test my_dataset_pipeline --reuse-dir
```

See [how runs/ work](../cli_overview.md#how-runs-work) for more details.

## 6. Deploy to Picsellia

```sh
pxl-pipeline deploy my_dataset_pipeline
```

This will:

- Build and push the Docker image
- Register the pipeline in Picsellia
- Sync the declared inputs to the platform (add new inputs, update existing ones, remove stale ones)

See [deployment lifecycle](../cli_overview.md#pipeline-lifecycle).
