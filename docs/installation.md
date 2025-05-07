# 📦 Installation

## Prerequisites
- Python **>=3.10**
- [Poetry](https://python-poetry.org/docs/) installed
- [Git](https://git-scm.com/downloads) installed

## Installation & Setup guide

You can either install this project locally for development, or use it directly as a dependency via Git.

### Option 1 — Install via Git Only (No Local Clone)

Use this method if you want to consume the CV Engine and Pipeline CLI without modifying the code.

✅ Using Poetry

```bash
poetry add git+https://github.com/picselliahq/picsellia-cv-engine.git@main
poetry add git+https://github.com/picselliahq/picsellia-pipelines-cli.git@main
```

✅ Using uv

```bash
uv add git+https://github.com/picselliahq/picsellia-cv-engine.git@main
uv add git+https://github.com/picselliahq/picsellia-pipelines-cli.git@main

```

✅ Using pip

```bash
pip install git+https://github.com/picselliahq/picsellia-cv-engine.git@main
pip install git+https://github.com/picselliahq/picsellia-pipelines-cli.git@main
```

### Option 2 — Develop Locally

Use this option if you want to contribute or make local changes.

#### 1. Clone the repository

```bash
git clone https://github.com/picselliahq/picsellia-cv-engine.git
cd picsellia-cv-engine
```

#### 2. Install dependencies with Poetry

```bash
poetry install
```

#### 3. Add the Pipeline CLI (manually)

`picsellia-pipelines-cli` is not included by default, you must add it:

```bash
poetry add git+https://github.com/picselliahq/picsellia-pipelines-cli.git
```
