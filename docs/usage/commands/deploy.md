# `pxl-pipeline deploy`

The `deploy` command builds a Docker image for your pipeline, pushes it to the registry,
and registers (or updates) the pipeline in Picsellia.

---

## Usage

```bash
pxl-pipeline deploy PIPELINE_NAME --organization ORG_NAME [--env ENV]
```

### Example:

```bash
pxl-pipeline deploy dataset_version_creation --organization my-org --env STAGING
```

### Options

| Option            | Description                                   | Default       |
|-------------------|-----------------------------------------------|---------------|
| `PIPELINE_NAME`   | Name of the pipeline project (folder).        | ✅ Required   |
| `--organization`  | Picsellia organization name.                  | ✅ Required   |
| `--env`           | Target environment: `PROD`, `STAGING`, `LOCAL`| `PROD`        |


## What happens during deploy?

**1. Pipeline details**

- Reads pipeline metadata from `config.toml` (type, description, etc.).

**2. Docker image name**

- If not already set in `config.toml`, prompts you to provide one.

- Format: `user/pipeline_name`.

**3. Version bump**

- Prompts for the next version bump (`patch`, `minor`, `major`, `rc`, `final`).

- Updates config.toml with the new version and image tag.

**4. Resource allocation**

- Prompts for default CPU and GPU allocation if missing.

- Saves values in the docker section of config.toml.

**5. Build & push Docker image**

- Builds the Docker image for the pipeline.

- Pushes tags: the new version and either latest (or test if RC).

**6. Environment setup**

- Resolves the target environment (PROD, STAGING, or LOCAL).

- Loads API token and org name from config or env vars.

**7. Register/update in Picsellia**

- If the pipeline does not exist, it is created.

- If it already exists, it is updated with the new image + resources.

**8. Sync inputs** *(processing pipelines only)*

- If an `inputs_class` is defined in `config.toml`, the CLI reads the declared inputs and syncs them with the platform.

- New inputs are added, existing inputs are updated, and inputs that are no longer declared in the class are removed.

- This ensures the platform always reflects the inputs defined in your code.

- See [Working with pipeline inputs](../cli_overview.md#working-with-pipeline-inputs) for details on defining inputs.

---

## Training pipelines

Training `deploy` registers the pipeline as a **training Docker image** on one or more **model versions** on Picsellia. Unlike processing deploy (which creates/updates a Processing asset and syncs `inputs_class`), training deploy:

1. Reads **deploy targets** from `config.toml` (see below).
2. Pre-checks that each target model/version exists (creates them if missing).
3. Builds and pushes the Docker image.
4. Updates **each** target model version with:
   - Docker image name and tag
   - Default hyperparameters from `parameters_class`
   - GPU docker flags (`--gpus all`, `--ipc host`, `--name training`)

### Model deploy targets in `config.toml`

Define **exactly one** of the following — not both:

**Single target** — one model version receives the training image:

```toml
[model_version]
origin_name = "MyModel"
name = "v1"
framework = "YOLOV8"
inference_type = "OBJECT_DETECTION"
```

**Multiple targets** — the same training image is deployed to several model versions (each row must be complete):

```toml
[[model_versions]]
origin_name = "MyModel"
name = "v1"
framework = "ONNX"
inference_type = "OBJECT_DETECTION"

[[model_versions]]
origin_name = "MyModel"
name = "v2"
framework = "ONNX"
inference_type = "OBJECT_DETECTION"
```

| Field | Description |
|-------|-------------|
| `origin_name` | Model name on the platform (created if it does not exist). |
| `name` | Model version name (created if it does not exist). |
| `framework` | SDK `Framework` enum value (e.g. `YOLOV8`, `ONNX`, `PYTORCH`). Defaults to `NOT_CONFIGURED` if omitted. |
| `inference_type` | SDK `InferenceType` enum value (e.g. `OBJECT_DETECTION`, `CLASSIFICATION`). Defaults to `NOT_CONFIGURED` if omitted. |

If the same `(origin_name, name)` pair appears more than once, the last entry wins.

### `config.toml` vs `run_config.toml`

These files serve different purposes for training:

| File | Used by | Purpose |
|------|---------|---------|
| `config.toml` | `init`, `deploy` | Pipeline metadata, Docker image, **`[model_version]` / `[[model_versions]]` deploy targets** |
| `runs/.../run_config.toml` | `test`, `launch` | Local run inputs: dataset version IDs, experiment, hyperparameters |

`[input.model_version].id` in `run_config.toml` is only for **local testing** — it is **not** used by `deploy`. Deploy always reads deploy fields from `config.toml`.

During interactive `init`, the CLI prompts for a model and writes the full `[model_version]` block into `config.toml`. With `init --run-config-file`, only a model version **id** may be stored for reference; you must still add deploy fields before running `deploy`.

### Example

```bash
pxl-pipeline deploy my_training_pipeline --organization my-org --env STAGING
```

Ensure `config.toml` contains `[model_version]` (or `[[model_versions]]`) with all required fields before deploying.
