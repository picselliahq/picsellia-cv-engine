# Picsellia CV Engine

**Picsellia CV Engine** is a modular engine for building, testing, and deploying computer vision pipelines — fully integrated with the Picsellia platform.

Whether you're transforming datasets, training models, or tracking experiments, this engine helps you organize everything into **clean, reusable components**.

## 🧠 What’s a pipeline?

A pipeline is a structured sequence of actions — like:

- 🧼 Preprocessing images
- 🧪 Training a model
- 📊 Evaluating predictions
- ☁️ Uploading results to Picsellia

Each action is implemented as a step — a small, focused function decorated with @step.

You can chain together these steps inside a @pipeline, and run it locally or on Picsellia.

## 🚀 Quickstart guide

### 1. Clone the repository

```bash
git clone https://github.com/picselliahq/picsellia-cv-engine.git
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

### 3. Link the Pipeline CLI

`picsellia-pipelines-cli` is not included by default, you must add it:

```bash
poetry add git+https://github.com/picselliahq/picsellia-pipelines-cli.git
```

## 📘 Explore the documentation

Start the docs server:

```bash
poetry run mkdocs serve -a 127.0.0.1:8080
```

Then open `http://127.0.0.1:8080/` in your browser to see all available documentation.

You’ll find:

- [📦 Installation Guide](http://127.0.0.1:8080/installation/)
- [🛠 Usage Examples](http://127.0.0.1:8080/usage/)
- [📖 API Reference](http://127.0.0.1:8080/api/)

## 🛠 Build your first pipeline

Create a pipeline (training or processing):

```bash
pipeline-cli training init my_pipeline --template simple
```
Then edit the generated files (e.g. `steps.py`, `requirements.txt`, `parameters.py`), and test locally:

```bash
pipeline-cli training test my_pipeline
```

Once everything works:
```bash
pipeline-cli training deploy my_pipeline
```

### 🙋 Need help?

- Follow the [Usage Guide](http://127.0.0.1:8080/usage)
- Browse the full [API Reference](http://127.0.0.1:8080/api/)
- Or reach out on the [Picsellia Platform](https://app.picsellia.com/signup)
