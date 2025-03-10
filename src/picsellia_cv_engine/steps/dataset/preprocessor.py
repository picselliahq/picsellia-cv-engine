import os

from picsellia_cv_engine.decorators.step_decorator import step
from picsellia_cv_engine.models.data.dataset.dataset_collection import (
    DatasetCollection,
)
from picsellia_cv_engine.models.steps.data.dataset.preprocessing.classification_dataset_context_preparator import (
    ClassificationBaseDatasetContextPreparator,
)


@step
def prepare_classification_datasets(
    dataset_collection: DatasetCollection,
    destination_dir: str,
) -> DatasetCollection:
    """
    Prepares a classification dataset by organizing image files into category-based subdirectories.

    This function processes a dataset collection by sorting images into directories named after their respective
    class labels (categories). The dataset is restructured into a format that is compatible with model training
    for classification tasks, where each category of images is placed into its own folder.

    Args:
        dataset_collection (DatasetCollection): The dataset collection to prepare, which includes images and
            the corresponding class labels.
        destination_dir (str): The destination directory where the prepared dataset will be saved, with
            category-based subdirectories for each class.

    Returns:
        DatasetCollection: A dataset collection with images organized into subdirectories, each named after the corresponding class labels.

    Examples:
        **Before Preparation:**
        ```
        dataset/
        ├── train/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   ├── image3.jpg
        ├── val/
        │   ├── image4.jpg
        │   ├── image5.jpg
        │   ├── image6.jpg
        └── test/
            ├── image7.jpg
            ├── image8.jpg
            └── image9.jpg
        ```

        **After Preparation:**
        ```
        dataset/
        ├── train/
        │   ├── category1/
        │   │   ├── image1.jpg
        │   │   └── image3.jpg
        │   └── category2/
        │       └── image2.jpg
        ├── val/
        │   ├── category1/
        │   │   └── image4.jpg
        │   └── category2/
        │       ├── image5.jpg
        │       └── image6.jpg
        └── test/
            ├── category1/
            │   └── image7.jpg
            └── category2/
                ├── image8.jpg
                └── image9.jpg
        ```
    """
    for dataset_context in dataset_collection:
        organizer = ClassificationBaseDatasetContextPreparator(
            dataset_context=dataset_context,
            destination_dir=os.path.join(destination_dir, dataset_context.dataset_name),
        )
        dataset_collection[dataset_context.dataset_name] = organizer.organize()
    return dataset_collection
