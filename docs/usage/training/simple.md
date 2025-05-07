# 🏋️ Simple training pipeline template

This guide shows you how to create, customize, test, and deploy a training pipeline using the simple template from `pipeline-cli`.

The pipeline uses Ultralytics for training, with built-in integration for datasets, model logging, and experiment management on Picsellia.


## 1. Initialize your pipeline

To generate a training pipeline project:

```bash
pipeline-cli training init test_training --template simple
```

This creates the pipeline under the pipelines/ folder:

```
pipelines/
└── test_training/
    ├── Dockerfile
    ├── requirements.txt
    ├── local_training_pipeline.py
    ├── training_pipeline.py
    ├── steps.py
    ├── utils/
    │   ├── parameters.py
    │   └── data.py
```

#### 🛠 During init, you will be prompted to:
When initializing the pipeline, you’ll be asked:

- Choose between creating a new model version or using an existing one.
- If you create a new one, the model will be registered with the default parameters `SimpleHyperParameters` from `utils/parameters.py`.
- If you use an existing model, make sure that `SimpleHyperParameters` locally matches the model's configuration.

## 2. Customize your pipeline

### Training logic: `steps.py`

Your model training logic lives in `steps.py`:

```python
from picsellia_cv_engine import step
from picsellia_cv_engine.core import Model, DatasetCollection, YoloDataset

@step()
def train(picsellia_model: Model, picsellia_datasets: DatasetCollection[YoloDataset]):
    ...
```

You can customize:
- Preprocessing
- Model instantiation
- Training arguments
- Logging logic (e.g., best model, metrics)

Just make sure each function is decorated with @step() to be tracked by the engine.

### Parameters: `utils/parameters.py`:

Defines the pipeline's hyperparameters. Minimal default:


```python
from picsellia_cv_engine.core.parameters import HyperParameters
from picsellia.types.schemas import LogDataType

class SimpleHyperParameters(HyperParameters):
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

Use it in `steps.py`:

```python
ultralytics_model.train(
    ...,
    lr0=context.hyperparameters.learning_rate,
)
```

❗Keep your model version and your code in sync: If you change the code or the model config, make sure they match.

A sync feature will be available soon to help push/pull parameter definitions automatically.

## 3. Test your pipeline locally

Before deploying, always run a local test:

```bash
pipeline-cli training test test_training
```

This will:

1. Create a `.venv` in the pipeline folder
2. Install `requirements.txt`
3. Prompt for the experiment ID

📌 You must create the experiment manually in the Picsellia UI and attach:

- The correct model version
- The appropriate datasets

✅ Outputs are stored under:

```python
pipelines/test_training/tests/<experiment-id>/
├── dataset/
├── models/
```

#### ⚠️ Local testing tip:
If you update your parameters in `SimpleHyperParameters`, remember to update them in the experiment configuration in the UI — especially if you are reusing the same experiment.

## 4. Deploy to Picsellia

When you're ready to deploy:

```bash
pipeline-cli training deploy test_training
```

This will:

1. Build the Docker image
2. Push it to your registry
3. Register the pipeline to the associated model version

Your `Dockerfile` installs:

1. `picsellia-cv-engine `
2. Torch + CUDA (via pre-built wheels)
3. Any custom requirements (`requirements.txt`)
