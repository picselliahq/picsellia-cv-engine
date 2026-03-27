# Datalake Processing Pipeline

This guide explains how to create, customize, test, and deploy a processing pipeline that runs on a **Datalake** using the `pxl-pipeline` cli with the `datalake` template.

These pipelines receive a datalake as their target and can apply tagging, filtering, or any custom logic to the data it contains.

---

## 1. Initialize your pipeline

```sh
pxl-pipeline init my_datalake_pipeline --type processing --template datalake
```

This generates a pipeline folder with standard files. See [project structure](../cli_overview.md#project-structure) for details.

## 2. Customize your pipeline logic

### `steps.py`

Contains the `process()` step where your core logic lives. The context gives you access to the target datalake, parameters, and any inputs you've declared.

```python
@step
def process():
    context: PicselliaDatalakeProcessingContext = Pipeline.get_active_context()
    parameters = context.processing_parameters
    datalake = context.target

    # If you want to process only selected data:
    data_ids_to_process = context.data_ids

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

When you run `pxl-pipeline init`, a `run_config.toml` is generated:

```toml
override_outputs = true

target_id = ""

[job]
type = "DATA_AUTO_TAGGING"

[inputs]
example_input = "example_value"

[parameters]
example_parameter = "default"
```

Fill in the `target_id` with the UUID of the datalake you want to process.

## 4. Manage dependencies with `uv`

```bash
uv add transformers --project my_datalake_pipeline
```

Dependencies are declared in `pyproject.toml`. See [dependency management with uv](../cli_overview.md#dependency-management-with-uv).

## 5. Test your pipeline locally

```sh
pxl-pipeline test my_datalake_pipeline
```

This will:

- Prompt for the target datalake if not set in the run config
- Scaffold any missing inputs with empty defaults
- Run the pipeline via `pipeline.py --mode local`
- Save everything under `runs/runX/`

To reuse the same folder and avoid re-downloading assets:

```bash
pxl-pipeline test my_datalake_pipeline --reuse-dir
```

See [how runs/ work](../cli_overview.md#how-runs-work) for more details.

## 6. Deploy to Picsellia

```sh
pxl-pipeline deploy my_datalake_pipeline
```

This will:

- Build and push the Docker image
- Register the pipeline in Picsellia
- Sync the declared inputs to the platform (add new inputs, update existing ones, remove stale ones)

See [deployment lifecycle](../cli_overview.md#pipeline-lifecycle).
