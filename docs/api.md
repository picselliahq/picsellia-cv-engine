# 📖 API Reference

The **Picsellia CV Engine API** provides a modular framework for defining dataset processing, model training, and data validation.

---

## **Available Components**

### **Processing Steps**

#### **Dataset Steps**
- [Dataset Loader](api/steps/dataset/loader.md)
- [Dataset Preprocessor](api/steps/dataset/preprocessor.md)
- [Dataset Uploader](api/steps/dataset/uploader.md)
- [Dataset Validator](api/steps/dataset/validator.md)

#### **Datalake Steps**
- [Datalake Loader](api/steps/datalake/loader.md)

#### **Model Steps**
- [Model Loader](api/steps/model/loader.md)

### **Decorators**
- [Pipeline Decorator](api/decorators/pipeline_decorator.md)
- [Step Decorator](api/decorators/step_decorator.md)

### **Data Models**
#### **Processing Contexts**
- [Picsellia Context](api/models/contexts/common/picsellia_context.md)
- [Dataset Processing Context](api/models/contexts/processing/dataset/picsellia_processing_context.md)
- [Model Processing Context](api/models/contexts/processing/model/picsellia_model_processing_context.md)
- [Training Context](api/models/contexts/training/picsellia_training_context.md)

#### **Data Handling**
- [COCO Dataset Context](api/models/data/dataset/coco_dataset_context.md)
- [Dataset Collection](api/models/data/dataset/dataset_collection.md)
- [Datalake Collection](api/models/data/datalake/datalake_collection.md)

#### **Model Management**
- [Model Context](api/models/model/model_context.md)
- [Model Downloader](api/models/model/model_downloader.md)
- [Picsellia Prediction](api/models/model/picsellia_prediction.md)

---

## **Using the API**
To integrate **Picsellia CV Engine** into your project:

```python
from picsellia_cv_engine.decorators.pipeline_decorator import pipeline
from picsellia_cv_engine.steps.dataset.loader import load_coco_datasets

@pipeline
def my_pipeline():
    dataset_collection = load_coco_datasets()
    return dataset_collection
```
