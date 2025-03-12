# Picsellia CV Engine

## Installation & Setup guide

This guide will help you set up Picsellia CV Engine and Pipeline CLI to start building and testing your own dataset processing pipelines.

### 1. Clone the required repositories

You need to clone two repositories:

```bash
git clone https://github.com/picselliahq/picsellia-cv-engine.git
git clone https://github.com/picselliahq/picsellia-pipelines-cli.git
```

Navigate to the picsellia-cv-engine directory:

```bash
cd picsellia-cv-engine
```

### 2. Install dependencies with Poetry

We use Poetry to manage dependencies. If you haven't installed Poetry yet, run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then, install the dependencies:

```bash
poetry install
```

This installs:

- Picsellia SDK
- Pipeline CLI (linked in dev mode)
- NumPy, Tabulate, and other required packages

### 4. Running the docs

To explore the documentation, run:

```bash
poetry run mkdocs serve -a 127.0.0.1:8080
```

Then open `http://127.0.0.1:8080/` in your browser to see all available documentation.

### 5. Start building your pipeline

Once everything is set up, you can create your first pipeline:

```bash
pipeline-cli init my_custom_pipeline
```

Modify the process_dataset.py file inside your pipeline folder, add any necessary dependencies to requirements.txt, and test your pipeline locally:

```python
pipeline-cli test my_custom_pipeline
```

If everything works correctly, deploy your pipeline to Picsellia:

```bash
pipeline-cli deploy my_custom_pipeline
```

### Need More Info?
For a detailed guide on building your pipeline, visit http://127.0.0.1:8080/usage once the docs are running.
