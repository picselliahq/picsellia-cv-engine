# Picsellia CV Engine Documentation

Welcome to the Picsellia CV Engine documentation.

This engine is the core execution layer used to build training and processing computer vision pipelines on top of the Picsellia platform.

It provides:

- a pipeline abstraction (`@pipeline`)

- composable steps (`@step`)

- shared execution contexts

- reusable dataset, model, and framework components

This documentation helps you understand how pipelines are built, reuse existing components, and extend the engine when needed.

## Where should I start?

Choose the path that best matches what you want to do.

---

### I want to run or build a pipeline

Start here if your goal is to create, customize, or run pipelines.


- [Installation](installation.md)  
  Install the engine and the `pxl-pipeline` CLI.

- [Pipeline usage examples](usage/index.md)  
  Real-world training and processing pipelines you can reuse or adapt.

- [CLI-based workflow](usage/cli_overview.md)  
  Understand how pipelines are generated, tested, and deployed.

Understand how pipelines are generated, tested, and deployed.

---

### I want to understand how pipelines work internally

Start here if you want to understand the engine concepts or debug pipeline behavior.

- [What’s a pipeline?](usage/index.md)  
  High-level explanation of pipelines, steps, and execution flow.

- [Execution contexts](api/core/contexts/common/picsellia_context.md)  
  How configuration, datasets, and metadata are injected into pipelines.

- [Base dataset loading steps](api/steps/base/dataset/loader.md)  
  How datasets are loaded and prepared inside pipelines.

How datasets are loaded and prepared inside pipelines.

---

### I want to reuse or extend engine components

Start here if you want to reuse existing logic or add new capabilities.

- [API Reference](api/index.md)  
  Full reference for contexts, decorators, steps, services, and frameworks.

- [Base steps](api/index.md#base-steps)  
  Reusable dataset, model, and datalake steps.

- [Framework-specific extensions](api/index.md#framework-specific-extensions)  
  Ultralytics and other framework integrations.

---

### I want to contribute to the engine

Start here if you want to modify the engine or its documentation.

- [Local development & contribution guide](../README.md#local-development)  
  Set up the repository and understand the contribution workflow.

- [Documentation workflow](../README.md#documentation)  
  How API docs and usage docs are generated and published.

---

## Core concepts at a glance

- **Pipeline**

  A Python function decorated with @pipeline that defines an execution flow.

- **Step**

  A focused unit of work decorated with @step (dataset loading, training, evaluation, etc.).

- **Context**

  A shared object injected into the pipeline, carrying configuration, datasets, models, and metadata.

Pipelines are composed by chaining steps together, allowing complex workflows to remain modular, readable, and reusable.

---

## 🔗 Related documentation

- [Pipeline usage examples](usage/index.md)  
  End-to-end training and processing workflows.

- [API reference](api/index.md)  
  Detailed documentation for all engine components.

- [Picsellia platform documentation](https://documentation.picsellia.com/docs/welcome)  
  Learn more about experiments, datasets, and models in Picsellia.

---


## 👋 New to Picsellia?

- Create a Picsellia account  
  https://app.picsellia.com/signup

- Explore the broader ecosystem:

  - Picsellia Pipelines CLI – pipeline lifecycle tooling  
    https://github.com/picselliahq/picsellia-pipelines-cli

  - Picsellia CV Pipelines – ready-to-use reference pipelines  
    https://github.com/picselliahq/picsellia-cv-pipelines
