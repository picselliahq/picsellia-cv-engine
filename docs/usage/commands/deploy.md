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
