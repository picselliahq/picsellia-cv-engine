# 📦 Installation

## Prerequisites
- Python **>=3.10**
- [Poetry](https://python-poetry.org/docs/) installed
- [Git](https://git-scm.com/downloads) installed

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
