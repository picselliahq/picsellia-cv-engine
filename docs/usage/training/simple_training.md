# Simple Training Pipeline

This guide explains how to create, customize, test, and deploy a training pipeline using the `simple` template from the `pxl-pipeline` cli.

This is the recommended starting point for training pipelines. It provides a minimal, framework-agnostic scaffold that you can extend with any ML framework of your choice.

---

## 1. Initialize your pipeline

```bash
pxl-pipeline init my_training_pipeline --type training --template simple
```

This generates a pipeline folder with standard files. See [project structure](../cli_overview.md#project-structure) for details.

During init, you'll be prompted to:

- Create a new model version or select an existing one
- If you create one, default parameters from `TrainingHyperParameters` will be used
- If using an existing model, ensure the parameter class matches the version's expected inputs

## 2. Customize your pipeline

### `pipeline.py`

The entry point creates a training context from your config and runs the pipeline:

```python
context = create_training_context_from_config(
    hyperparameters_cls=TrainingHyperParameters,
    augmentation_parameters_cls=AugmentationParameters,
    export_parameters_cls=ExportParameters,
    mode=args.mode,
    config_file_path=args.config_file,
)

@pipeline(context=context, log_folder_path="logs/", remove_logs_on_completion=False)
def my_training_pipeline():
    datasets = list_training_datasets()
    print(context.hyperparameters.epochs)
    # Your training code goes here ...
```

### `steps.py`

Contains your training steps. The template includes a simple step to list attached datasets:

```python
@step()
def list_training_datasets() -> list[DatasetVersion]:
    context = Pipeline.get_active_context()
    experiment = context.experiment
    datasets = experiment.list_attached_dataset_versions()
    return datasets
```

You can add more steps for data preprocessing, model building, training loops, evaluation, and artifact saving.

### `utils/parameters.py`

This file defines the training hyperparameters for the pipeline:

```python
class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.epochs = self.extract_parameter(["epochs"], expected_type=int, default=3)
        self.batch_size = self.extract_parameter(["batch_size"], expected_type=int, default=8)
        self.image_size = self.extract_parameter(["image_size"], expected_type=int, default=640)
```

To add a new hyperparameter (e.g., learning rate):

```python
self.learning_rate = self.extract_parameter(["lr"], expected_type=float, default=0.001)
```

See [Working with pipeline parameters](../cli_overview.md#working-with-pipeline-parameters) for more advanced usage.

### `pyproject.toml`: Customize your dependencies

Dependencies are managed with uv. To add a new package to the pipeline environment:

```bash
uv add torch --project my_training_pipeline
```

To install a Git-based package:

```bash
uv add git+https://github.com/picselliahq/picsellia-cv-engine.git --project my_training_pipeline
```

This updates the `pyproject.toml` and `uv.lock`.
The CLI will automatically install everything on the next test or deploy.

See [dependency management with uv](../cli_overview.md#dependency-management-with-uv) for full details.

## 3. Configure `run_config.toml` for local testing

When you run `pxl-pipeline init`, a `run_config.toml` is generated:

```toml
override_outputs = true

[job]
type = "TRAINING"

[input.train_dataset_version]
id = ""

[input.model_version]
id = ""

[output.experiment]
name = "my_training_pipeline_exp1"
project_name = "my_training_pipeline"

[hyperparameters]
epochs = 3
batch_size = 8
image_size = 640
```

Fill in the `input.train_dataset_version.id` and `input.model_version.id` with the UUIDs from your Picsellia workspace. The `output.experiment` section defines where the experiment results will be stored.

> **Note:** `run_config.toml` drives **local test** runs only. It does not configure `deploy`.

## 4. Test your pipeline locally

```bash
pxl-pipeline test my_training_pipeline
```

This will:

- Create a `.venv` in the pipeline folder
- Install dependencies using uv
- Prompt for an `experiment_id`

You must create the experiment manually in the Picsellia UI and attach the correct model version and training datasets.

Outputs will be saved under:

```
my_training_pipeline/runs/<runX>/
├── run_config.toml
├── dataset/
└── models/
```

See [how runs/ work](../cli_overview.md#how-runs-work) for details on configuration reuse.

## 5. Configure `config.toml` for deploy

Before `pxl-pipeline deploy`, `config.toml` must declare where the training Docker image should be registered on Picsellia.

Use **either** a single `[model_version]` **or** several `[[model_versions]]` rows — not both:

```toml
[model_version]
origin_name = "my-training-model"
name = "v1"
framework = "YOLOV8"
inference_type = "OBJECT_DETECTION"
```


| Field            | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| `origin_name`    | Model name on the platform (created on deploy if missing).        |
| `name`           | Version name (created on deploy if missing).                      |
| `framework`      | `Framework` enum from the Picsellia SDK (e.g. `YOLOV8`, `ONNX`).  |
| `inference_type` | `InferenceType` enum (e.g. `OBJECT_DETECTION`, `CLASSIFICATION`). |


Interactive `init` fills this section for you. If you used `init --run-config-file`, add these fields manually — an `id` alone is not enough for deploy.

To attach the same training pipeline to multiple model versions:

```toml
[[model_versions]]
origin_name = "my-training-model"
name = "v1"
framework = "YOLOV8"
inference_type = "OBJECT_DETECTION"

[[model_versions]]
origin_name = "my-training-model"
name = "v2"
framework = "YOLOV8"
inference_type = "OBJECT_DETECTION"
```

See [deploy — Training pipelines](../commands/deploy.md#training-pipelines) for the full deploy flow.

## 6. Deploy to Picsellia

```bash
pxl-pipeline deploy my_training_pipeline --organization my-org --env STAGING
```

This will:

1. Validate deploy targets from `[model_version]` or `[[model_versions]]` in `config.toml`
2. Ensure each target model/version exists on the platform (create if needed)
3. Build a Docker image (based on your Dockerfile) and push it to your registry
4. Update **each** target model version with the image, tag, and default hyperparameters

Your `Dockerfile` installs `picsellia-cv-engine` and any other dependencies from `pyproject.toml`.

After deploy, users can launch training from the Picsellia UI on any of the registered model versions.
